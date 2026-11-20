from copy import deepcopy
from math import log
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from src.models.sequence_objective import SequenceRunner, last_valid_state, multi_positive_loss


class TinySequence(nn.Module):
    """Small differentiable encoder with the same tied head contract as the adapters."""

    def __init__(self):
        super().__init__()
        self.item_embedding = nn.Embedding(9, 3, padding_idx=0)
        self.encoder = nn.Linear(3, 3)
        self.dropout = nn.Dropout(0.25)
        self.output_bias = nn.Parameter(torch.linspace(-0.1, 0.1, 9))

    def encode(self, ids, lengths):
        history = self.item_embedding(ids) * ids.ne(0).unsqueeze(2)
        return self.dropout(self.encoder(history.cumsum(dim=1)))

    def score_items(self, states, item_ids):
        return states @ self.item_embedding(item_ids).T + self.output_bias[item_ids]


def toy_data():
    train = pd.DataFrame({"query_id": ["a", "b", "c", "d"], "catalog_id": ["early", "late", "early", "late"],
                          "split": "train"})
    return SimpleNamespace(
        vocab=pd.DataFrame({"item_id": list(range(1, 9)), "stock_code": list("ABCDEFGH")}),
        train_queries=train, val_queries=pd.DataFrame({"query_id": ["val"], "catalog_id": ["oct"]}),
        catalog_ids={"early": [4, 2], "late": [3, 4, 2], "oct": [6, 2, 7, 1, 3, 4]},
        histories_by_query={"a": [7, 8, 1, 2], "b": [2, 4], "c": [1, 1], "d": [1],
                            "val": [3, 7], "val-alt": [3, 6], "empty": []},
        positives={"a": [2, 2], "b": [3], "c": [4], "d": [2, 4]},
    )


def test_query_macro_loss_uses_only_distinct_positives_and_current_catalog_columns():
    # Raw logits correspond to an unnormalized ratio 1:2:4. Repeated 3 and
    # a target outside this cutoff cannot give the second query extra weight.
    raw = torch.tensor([[1., 2., 4.], [1., 2., 4.]], dtype=torch.float64)
    logits = raw.log().requires_grad_()
    loss = multi_positive_loss(logits, torch.tensor([4, 2, 3]), [[4], [3, 2, 3, 5]])
    assert loss.item() == pytest.approx((log(7) + (log(7 / 4) + log(7 / 2)) / 2) / 2)
    loss.backward()
    assert torch.allclose(logits.grad.sum(dim=1), torch.zeros(2, dtype=torch.float64))
    # Against the softmax gradient, each query sums to one equally weighted target mass.
    targets = torch.tensor([[1., 0., 0.], [0., .5, .5]], dtype=torch.float64)
    assert torch.allclose(logits.grad, (raw / 7 - targets) / 2)
    for catalog, positives in [([0, 2, 3], [[3]]), ([4, 4, 3], [[3]]), ([4, 2, 3], [[7]])]:
        with pytest.raises(ValueError):
            multi_positive_loss(logits[:1], catalog, positives)


@pytest.mark.parametrize("padding,expected", [("left", [[4], [8], [0]]),
                                              ("right", [[1], [6], [0]])])
def test_valid_positions_and_empty_history(padding, expected):
    states = torch.arange(1, 13).reshape(3, 4, 1)
    assert last_valid_state(states, torch.tensor([1, 2, 0]), padding).tolist() == expected
    with pytest.raises(ValueError):
        last_valid_state(states, torch.tensor([1, 5, 0]), padding)


@pytest.mark.parametrize("padding", ["left", "right"])
def test_actual_batch_support_fallback_on_input_and_output_preserves_training_weights(padding):
    torch.manual_seed(102)
    data = toy_data()
    runner = SequenceRunner(TinySequence(), data, padding=padding, max_length=2, batch_size=2)
    original = runner.network.encoder.weight.detach().clone()
    assert np.isfinite(runner.train_epoch())
    assert runner.last_encoder_grad_norm > 0 and runner.last_head_grad_norm > 0
    assert not torch.equal(original, runner.network.encoder.weight)
    assert runner.input_support.nonzero().flatten().tolist() == [1, 2, 4]
    assert runner.catalog_support.nonzero().flatten().tolist() == [2, 3, 4]
    assert runner.supported_items == {1, 2, 3, 4} and runner.support_summary()["item_bias_rows"] == 3
    q = next(data.val_queries.itertuples(index=False))
    candidates = data.catalog_ids["oct"]
    real_weights = deepcopy(runner.network.state_dict())
    actual = runner.score(q, candidates)
    chunked = np.r_[runner.score(q, candidates[:2]), runner.score(q, candidates[2:])]
    np.testing.assert_allclose(actual, chunked, atol=1e-6)
    # Independent reference replaces rows in a disposable network. ID 1 appears
    # only as a history input; its tied embedding is learned, its scoring bias is not.
    ref = deepcopy(runner.network).eval()
    with torch.no_grad():
        ref.item_embedding.weight[[5, 6, 7, 8]] = ref.item_embedding.weight[[1, 2, 3, 4]].mean(0)
        ref.output_bias[[1, 5, 6, 7, 8]] = ref.output_bias[[2, 3, 4]].mean()
        hidden = ref.encode(torch.tensor([[3, 7]]), torch.tensor([2]))[:, -1]
        expected = ref.score_items(hidden, torch.tensor(candidates))
    np.testing.assert_allclose(actual, expected.numpy()[0], atol=1e-6)
    other = SimpleNamespace(query_id="val-alt", catalog_id="oct")
    np.testing.assert_allclose(runner.score(other, candidates), actual, atol=1e-6)
    assert actual[candidates.index(6)] == pytest.approx(actual[candidates.index(7)])
    for key, value in runner.network.state_dict().items():
        assert torch.equal(value, real_weights[key])
    empty = SimpleNamespace(query_id="empty", catalog_id="oct")
    np.testing.assert_allclose(runner.score(empty, candidates), ref.output_bias.detach()[candidates].numpy())
    for invalid in [[0], [8]]:
        with pytest.raises(ValueError):
            runner.score(q, invalid)


def test_optimizer_shuffle_dropout_and_supported_rows_resume_at_epoch_boundary(tmp_path):
    torch.manual_seed(23)
    data = toy_data()
    runner = SequenceRunner(TinySequence(), data, padding="right", max_length=2, batch_size=2, seed=59)
    runner.train_epoch()
    q = next(data.val_queries.itertuples(index=False))
    candidates = data.catalog_ids["oct"]
    before = runner.score(q, candidates)
    path = tmp_path / "checkpoint.pt"
    runner.save(path)
    alternate = SequenceRunner(TinySequence(), data, padding="right", max_length=2, batch_size=2, seed=9)
    alternate.restore(path)
    np.testing.assert_array_equal(before, alternate.score(q, candidates))
    assert alternate.seed == 59 and alternate.support_summary() == runner.support_summary()
    next_loss = runner.train_epoch()
    torch.rand(17)  # Unrelated global RNG use must not change resumed dropout or ordering.
    assert alternate.train_epoch() == next_loss
    assert alternate.epoch == runner.epoch == 2
    assert alternate.optimizer.state_dict()["state"].keys() == runner.optimizer.state_dict()["state"].keys()
    for name, state in runner.network.state_dict().items():
        assert torch.equal(alternate.network.state_dict()[name], state)
    np.testing.assert_array_equal(runner.score(q, candidates), alternate.score(q, candidates))
    incompatible = SequenceRunner(TinySequence(), data, padding="left", max_length=2, batch_size=2)
    with pytest.raises(ValueError, match="incompatible"):
        incompatible.restore(path)
