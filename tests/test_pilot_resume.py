"""Recover a real MF pilot after a committed epoch and reject incompatible histories."""

import json
from argparse import Namespace

import pytest

from scripts import train


def test_interrupted_epoch_and_changed_sequence_policy(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    args = Namespace(model="mf", config=None, seed=42, smoke=True, device="cpu", threads=1,
                     data_dir=train.ROOT / "data/processed/d2", resume=False)
    original = train.write_atomic

    def interrupt_after_checkpoint(path, writer):
        if path.name == "metrics.csv":
            raise OSError("Interrupted after committing the epoch")
        original(path, writer)

    with monkeypatch.context() as patch:
        patch.setattr(train, "write_atomic", interrupt_after_checkpoint)
        with pytest.raises(OSError, match="Interrupted"):
            train.run_model(args, tmp_path)
    assert json.loads((tmp_path / "progress.json").read_text())["best_epoch"] == 1

    args.resume = True
    with monkeypatch.context() as patch:
        digest = train.digest
        patch.setattr(train, "digest", lambda p: "changed" if p.name == "sequences.py" else digest(p))
        with pytest.raises(ValueError, match="same model, code, inputs"):
            train.run_model(args, tmp_path)
    result = train.run_model(args, tmp_path)
    assert result["status"] == "complete" and result["epochs_completed"] == 1
    assert result["reload_identical_top10"] and result["reload_max_score_delta"] == 0
    assert train.run_model(args, tmp_path)["output_sha256"] == result["output_sha256"]
