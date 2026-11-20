"""LRURec accuracy, left padding and parity with the pinned research implementation."""

import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from src.models.lrurec_adapter import LRURec
from src.models.sequence_objective import multi_positive_loss

REFERENCE_SHA256 = "149e0ccfe03c7eba7a51bf405d179f8a3f395011c0f730169aec2b59c9c156b1"


def test_scan_matches_direct_complex_recurrence():
    torch.manual_seed(7)
    model = LRURec(9, hidden_dim=5, num_layers=1, dropout=0).eval()
    layer = model.blocks[0]
    x = torch.randn(2, 7, 5)
    valid = torch.tensor([[False, False, False, False, True, True, True], [True] * 7])
    decay, phase, scale = layer.log_dynamics.exp().unbind()
    transition = torch.exp(torch.complex(-decay, phase))
    state = torch.zeros(2, 10, dtype=torch.complex64)
    values = []
    for t in range(x.size(1)):
        add = layer.input_projection(x[:, t].to(torch.complex64)) * scale
        state = torch.where(valid[:, t, None], transition * state + add, 0)
        values.append(layer.output_projection(state).real)
    residual = layer.recurrent_norm(x + torch.stack(values, dim=1))
    expected = layer.feedforward_norm(residual + layer.contract(torch.nn.functional.gelu(layer.expand(residual))))
    expected = expected.masked_fill(~valid.unsqueeze(-1), 0)
    torch.testing.assert_close(layer(x, valid), expected, rtol=2e-5, atol=2e-6)


def test_padding_ragged_batch_repeats_and_checkpoint(tmp_path):
    torch.manual_seed(19)
    model = LRURec(11, hidden_dim=8, num_layers=2, dropout=0.2).eval()
    ids = torch.tensor([[0, 0, 0, 0, 1, 2, 1], [0, 3, 7, 1, 8, 8, 6], [0] * 7])
    states = model.encode(ids, torch.tensor([3, 6, 0]))
    single = model.encode(torch.tensor([[1, 2, 1]]), torch.tensor([3]))
    torch.testing.assert_close(states[0, -3:], single[0], rtol=2e-5, atol=2e-6)
    assert torch.count_nonzero(states[0, :-3]) == 0 and torch.count_nonzero(states[2]) == 0
    items = torch.tensor([1, 2, 6, 11])
    scores = model.score_items(states[:, -1], items)
    expected = states[:, -1] @ model.item_embedding.weight[items].T + model.output_bias[items]
    torch.testing.assert_close(scores, expected)
    # Repeated item 1 in the history keeps its raw score in column 0.
    torch.testing.assert_close(scores[:1], model.score_items(single[:, -1], items), rtol=2e-5, atol=2e-6)
    path = tmp_path / "lrurec.pt"
    torch.save(model.state_dict(), path)
    restored = LRURec(11, hidden_dim=8, num_layers=2, dropout=0.2).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    recovered = restored.score_items(restored.encode(ids, torch.tensor([3, 6, 0]))[:, -1], items)
    torch.testing.assert_close(scores, recovered, rtol=0, atol=0)


def test_monthly_loss_backpropagates_to_recurrence_tied_weights_and_bias():
    torch.manual_seed(31)
    model = LRURec(6, hidden_dim=8, num_layers=1, dropout=0)
    ids = torch.tensor([[0, 1, 2, 1], [2, 4, 3, 5]])
    state = model.encode(ids, torch.tensor([3, 4]))[:, -1]
    scores = model.score_items(state, torch.arange(1, 7))
    loss = multi_positive_loss(scores, list(range(1, 7)), [[1, 3], [4]])
    loss.backward()
    for p in (model.blocks[0].log_dynamics, model.blocks[0].input_projection.weight,
              model.item_embedding.weight, model.output_bias):
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert torch.count_nonzero(model.item_embedding.weight.grad[0]) == 0


def test_raw_scores_against_pinned_upstream():
    """Supply the authors' original model/lru.py through LRUREC_REFERENCE_FILE."""
    location = os.environ.get("LRUREC_REFERENCE_FILE")
    if not location:
        pytest.skip("Set LRUREC_REFERENCE_FILE to the pinned authors' model/lru.py for parity testing.")
    path = Path(location)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256
    spec = importlib.util.spec_from_file_location("pinned_lrurec_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    torch.manual_seed(23)
    args = SimpleNamespace(num_items=11, bert_hidden_units=8, bert_num_blocks=2,
                           bert_dropout=0.0, bert_attn_dropout=0.0, dataset_code="retail")
    official = reference.LRU(args).eval()
    model = LRURec(11, hidden_dim=8, num_layers=2, dropout=0).eval()
    with torch.no_grad():
        model.item_embedding.load_state_dict(official.embedding.token.state_dict())
        model.input_norm.load_state_dict(official.embedding.layer_norm.state_dict())
        model.output_bias.copy_(official.model.bias)
        for target, origin in zip(model.blocks, official.model.lru_blocks, strict=True):
            target.log_dynamics.copy_(origin.lru_layer.params_log)
            for ours, theirs in (
                (target.input_projection, origin.lru_layer.in_proj),
                (target.output_projection, origin.lru_layer.out_proj),
                (target.recurrent_norm, origin.lru_layer.layer_norm),
                (target.expand, origin.feed_forward.w_1),
                (target.contract, origin.feed_forward.w_2),
                (target.feedforward_norm, origin.feed_forward.layer_norm),
            ):
                ours.load_state_dict(theirs.state_dict())
        ids = torch.tensor([[0, 0, 1, 2, 1], [0, 0, 0, 0, 3], [5, 2, 3, 4, 6]])
        valid_items = torch.arange(1, 12)
        expected, _ = official(ids)
        actual = model.score_items(model.encode(ids, ids.ne(0).sum(1))[:, -1], valid_items)
        torch.testing.assert_close(actual, expected[:, -1, 1:], rtol=2e-5, atol=2e-6)
