"""Exercise the actual pinned LRURec encoder, monthly training and candidate scoring."""

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models import lrurec_source
from src.models.lrurec_adapter import LRURec
from src.models.sequence_objective import SequenceRunner, multi_positive_loss


def test_verified_source_cache_reuse_and_tamper_detection(tmp_path, monkeypatch):
    # Bootstrap from the same verified source used in training, then exercise a fresh offline cache.
    _, original = lrurec_source.read_source()
    monkeypatch.delenv("LRUREC_REFERENCE_FILE", raising=False)
    monkeypatch.setenv("LRUREC_CACHE_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        lrurec_source.read_source(offline=True)

    def fetch(url, *, timeout):
        assert url == lrurec_source.SOURCE_URL
        return BytesIO(original)

    monkeypatch.setattr(lrurec_source, "urlopen", fetch)
    path, content = lrurec_source.read_source()
    assert content == original == path.read_bytes()

    def unexpected_fetch(*args, **kwargs):
        pytest.fail("A cached source must not trigger a download")

    monkeypatch.setattr(lrurec_source, "urlopen", unexpected_fetch)
    assert lrurec_source.read_source(offline=True) == (path, original)
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        lrurec_source.load_lru_class()


def test_raw_scores_against_pinned_upstream():
    torch.manual_seed(23)
    model = LRURec(11, hidden_dim=8, num_layers=2, dropout=0).eval()
    assert model.upstream.__class__.__name__ == "LRU"
    ids = torch.tensor([[0, 0, 1, 2, 1], [0, 0, 0, 0, 3], [5, 2, 3, 4, 6]])
    candidates = torch.arange(1, 12)
    with torch.no_grad():
        reference, _ = model.upstream(ids)
        last = model.encode(ids, ids.ne(0).sum(1))[:, -1]
        chunks = [model.score_items(last, chunk) for chunk in candidates.split(4)]
    # 3 different histories x 11 real items, including purchases appearing in the histories.
    torch.testing.assert_close(torch.cat(chunks, dim=1), reference[:, -1, 1:], rtol=2e-5, atol=2e-6)


def test_padding_ragged_batch_repeats_and_checkpoint(tmp_path):
    torch.manual_seed(19)
    model = LRURec(11, hidden_dim=8, num_layers=2, dropout=0.2).eval()
    ids = torch.tensor([[0, 0, 0, 0, 1, 2, 1], [0, 3, 7, 1, 8, 8, 6], [0] * 7])
    lengths = torch.tensor([3, 6, 0])
    states = model.encode(ids, lengths)
    single = model.encode(torch.tensor([[1, 2, 1]]), torch.tensor([3]))
    torch.testing.assert_close(states[0, -3:], single[0], rtol=2e-5, atol=2e-6)
    assert torch.count_nonzero(states[0, :-3]) == 0 and torch.count_nonzero(states[2]) == 0
    items = torch.tensor([1, 2, 6, 11])
    scores = model.score_items(states[:, -1], items)
    path = tmp_path / "lrurec.pt"
    torch.save(model.state_dict(), path)
    restored = LRURec(11, hidden_dim=8, num_layers=2, dropout=0.2).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    recovered = restored.score_items(restored.encode(ids, lengths)[:, -1], items)
    torch.testing.assert_close(scores, recovered, rtol=0, atol=0)
    with pytest.raises(ValueError, match="zeros on the left"):
        model.encode(torch.tensor([[1, 2, 0]]), torch.tensor([2]))


def test_monthly_loss_backpropagates_to_author_encoder_tied_weights_and_bias():
    torch.manual_seed(31)
    model = LRURec(6, hidden_dim=8, num_layers=1, dropout=0)
    ids = torch.tensor([[0, 1, 2, 1], [2, 4, 3, 5]])
    state = model.encode(ids, torch.tensor([3, 4]))[:, -1]
    scores = model.score_items(state, torch.arange(1, 7))
    loss = multi_positive_loss(scores, list(range(1, 7)), [[1, 3], [4]])
    loss.backward()
    recurrence = model.upstream.model.lru_blocks[0].lru_layer
    for param in (recurrence.params_log, recurrence.in_proj.weight, model.item_embedding.weight, model.output_bias):
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    assert torch.count_nonzero(model.item_embedding.weight.grad[0]) == 0


def test_chunked_runner_reuses_encoding_and_matches_author_with_item_fallback():
    # Items 1 and 6 are history-only and need bias fallback; 7 and 8 need both fallbacks.
    candidates = [8, 1, 3, 4, 2, 7]
    data = SimpleNamespace(
        vocab=pd.DataFrame({"item_id": range(1, 9)}),
        train_queries=pd.DataFrame({"query_id": ["a"], "catalog_id": ["train"], "split": ["train"]}),
        catalog_ids={"train": [2, 3, 4], "val": candidates},
        histories_by_query={"a": [1, 6], "b": [2, 7, 2]}, positives={"a": [2, 3]},
    )
    torch.manual_seed(17)
    model = LRURec(8, hidden_dim=8, num_layers=1, dropout=0)
    runner = SequenceRunner(model, data, padding="left", max_length=5)
    runner.train_epoch()
    assert runner.supported_items == {1, 2, 3, 4, 6}
    assert set(runner.catalog_support.nonzero().flatten().tolist()) == {2, 3, 4}

    # Build a full-forward oracle with the two different support masks measured during training.
    ref = deepcopy(model.upstream).eval()
    with torch.no_grad():
        weights = ref.embedding.token.weight
        weights[[5, 7, 8]] = weights[[1, 2, 3, 4, 6]].mean(0)
        ref.model.bias[[1, 5, 6, 7, 8]] = ref.model.bias[[2, 3, 4]].mean()
        reference, _ = ref(torch.tensor([[0, 0, 2, 7, 2]]))

    encoded = []
    hook = model.upstream.embedding.register_forward_hook(lambda *args: encoded.append(1))
    query = SimpleNamespace(query_id="b", catalog_id="val")
    try:
        actual = np.r_[runner.score(query, candidates[:3]), runner.score(query, candidates[3:])]
    finally:
        hook.remove()
    assert len(encoded) == 1
    np.testing.assert_allclose(actual, reference[0, -1, candidates].numpy(), rtol=2e-5, atol=2e-6)
