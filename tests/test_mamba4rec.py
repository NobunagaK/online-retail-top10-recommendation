"""CPU capability reporting and real CUDA checks for the Mamba-1 recommendation adapter."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_reports_an_unavailable_gpu(tmp_path):
    output = tmp_path / "gpu-preflight.json"
    result = subprocess.run(
        [sys.executable, "-m", "src.models.mamba4rec_adapter", "--preflight", "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode != 0
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["error"]
    assert "mamba-ssm" in report["packages"]
    assert "encoder_and_output_updated" not in report


@pytest.fixture
def cuda_torch():
    torch = pytest.importorskip("torch", reason="GPU tests need PyTorch with CUDA")
    if not torch.cuda.is_available():
        pytest.skip("Native Mamba-1 forward/backward requires an NVIDIA GPU")
    return torch


def test_native_gradients_padding_and_reload(cuda_torch):
    from src.models.mamba4rec_adapter import run_gpu_preflight

    measured = run_gpu_preflight()
    assert measured["encoder_and_output_updated"]
    assert measured["grad_norms"]["output_only_item_4"] > 0
    assert measured["grad_norms"]["mamba_transition_log"] > 0


@pytest.mark.parametrize("num_layers", [1, 2])
def test_original_residual_rule_scoring_and_empty_history(cuda_torch, num_layers):
    from src.models.mamba4rec_adapter import Mamba4Rec
    from src.models.sequence_objective import last_valid_state

    torch = cuda_torch
    torch.manual_seed(42)
    model = Mamba4Rec(5, hidden_dim=16, num_layers=num_layers, dropout=0.0).cuda().eval()
    ids = torch.tensor([[1, 2, 1, 0], [5, 0, 0, 0], [0, 0, 0, 0]], device="cuda")
    lengths = torch.tensor([3, 1, 0], device="cuda")
    candidates = torch.tensor([5, 1, 2], device="cuda")
    with torch.no_grad():
        states = model.encode(ids, lengths)
        # Rebuild the authors' one-layer / stacked residual and FFN from the actual CUDA operators.
        reference = model.input_norm(model.item_embedding(ids))
        for block in model.blocks:
            native = block.mamba(reference)
            after_mixer = block.mixer_norm(native if num_layers == 1 else native + reference)
            reference = block.output_norm(after_mixer + block.contract(block.activation(block.expand(after_mixer))))
        selected = last_valid_state(states, lengths, "right")
        torch.testing.assert_close(selected[:2], reference[[0, 1], [2, 0]], atol=1e-5, rtol=1e-5)
        assert torch.count_nonzero(selected[2]) == 0
        logits = model.score_items(selected, candidates)
        direct = selected @ model.item_embedding.weight[candidates].T
        torch.testing.assert_close(logits, direct)
        # SKU 1 occurs twice in the first history and remains a candidate with its raw dot product.
        torch.testing.assert_close(logits[0, 1], (selected[0] * model.item_embedding.weight[1]).sum())
    with pytest.raises(ValueError, match="real items first"):
        model.encode(torch.tensor([[0, 5, 0]], device="cuda"), torch.tensor([1], device="cuda"))
