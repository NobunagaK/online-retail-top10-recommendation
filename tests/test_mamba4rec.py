"""CPU capability reporting and real CUDA checks for the Mamba-1 recommendation adapter."""

import hashlib
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
    assert "recbole" in report["packages"]
    assert report["recbole_imports"]["status"] in {"passed", "failed"}
    assert report["upstream_sha256"] == "5cfb15241ad3608ef21e7abd9739ad03b363c10759dcdf136aa02305392c91f2"
    assert "encoder_and_output_updated" not in report


def test_vendored_recommender_and_license_match_pinned_upstream(tmp_path):
    source_dir = ROOT / "third_party" / "mamba4rec"
    source = json.loads((source_dir / "source.json").read_text())
    assert source["commit"] == "44a0c8432b34e3b89d46035b10e8525b1bd368dc"
    for name, expected in source["sha256"].items():
        contents = (source_dir / name).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == expected
        (tmp_path / name).write_bytes(contents)
    pytest.importorskip("torch")
    from src.models.mamba4rec_adapter import verify_upstream_files

    verify_upstream_files(tmp_path)
    with (tmp_path / "mamba4rec.py").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_upstream_files(tmp_path)


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
    assert measured["upstream_full_sort_max_abs_error"] < 1e-4
    assert measured["grad_norms"]["output_only_item_4"] > 0
    assert measured["grad_norms"]["mamba_transition_log"] > 0


@pytest.mark.parametrize("num_layers", [1, 2])
def test_official_full_sort_with_repeats_padding_and_empty_history(cuda_torch, num_layers):
    from src.models.mamba4rec_adapter import Mamba4Rec
    from third_party.mamba4rec.mamba4rec import Mamba4Rec as AuthorsMamba4Rec
    from recbole.model.abstract_recommender import SequentialRecommender

    torch = cuda_torch
    torch.manual_seed(42)
    model = Mamba4Rec(5, hidden_dim=16, num_layers=num_layers, dropout=0.0).cuda().eval()
    assert type(model.upstream) is AuthorsMamba4Rec
    assert isinstance(model.upstream, SequentialRecommender)
    ids = torch.tensor([[1, 2, 1, 0], [5, 0, 0, 0], [0, 0, 0, 0]], device="cuda")
    lengths = torch.tensor([3, 1, 0], device="cuda")
    candidates = torch.tensor([5, 1, 2], device="cuda")
    with torch.no_grad():
        state = model.encode(ids, lengths)
        scores = model.score_items(state, candidates)
        official = model.upstream
        official_state = official(ids[:2], lengths[:2])
        official_scores = official.full_sort_predict({official.ITEM_SEQ: ids[:2], official.ITEM_SEQ_LEN: lengths[:2]})
        torch.testing.assert_close(state[:2], official_state, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[:2], official_scores[:, candidates], atol=1e-5, rtol=1e-5)
        assert torch.count_nonzero(state[2]) == torch.count_nonzero(scores[2]) == 0
    with pytest.raises(ValueError, match="real items first"):
        model.encode(torch.tensor([[0, 5, 0]], device="cuda"), torch.tensor([1], device="cuda"))
