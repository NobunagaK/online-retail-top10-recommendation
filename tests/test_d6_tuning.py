"""D6 registration, October-only selection, baseline reuse and a synthetic locked November."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts import build_queries, evaluate, freeze_experiment, train, tune
from test_monthly_queries import ledger_fixture


def test_budget_length_comparison_and_hardware_ties():
    plan = tune.registration()
    assert list(plan["configs"]) == list(train.MODULES)
    assert plan["seeds"] == [42, 2026, 3407] and plan["finalists"] == 2
    for model, configs in plan["configs"].items():
        assert len(configs) == 8
        assert configs["01"] == yaml.safe_load((tune.ROOT / "configs" / f"{model}.yaml").read_text())
        if model in {"lrurec", "mamba4rec"}:
            assert [configs[c]["max_length"] for c in ("02", "03", "04")] == [64, 128, 256]
            assert {k for k in configs["02"] if configs["02"][k] != configs["04"][k]} == {"max_length"}
            assert {k for k in configs["03"] if configs["03"][k] != configs["05"][k]} == {"lr_scheduler"}
            baseline = {**configs["01"], **train.optimizer_settings(model, configs["01"])}
            assert {k for k in baseline if baseline[k] != configs["08"][k]} == {"optimizer", "lr_scheduler"}
            assert configs["08"]["lr_scheduler"] == "cosine"
            assert configs["02"]["optimizer"] == "adamw" and configs["02"]["patience"] == 15
    a = dict(id="first", ndcg_at_10=.2, recall_at_10=.3, train_seconds=6, hardware_keys=["T4"])
    b = dict(a, id="second", train_seconds=3)
    assert tune.rank([a, b])[0]["id"] == "second"
    assert tune.rank([a, dict(b, hardware_keys=["L4"])])[0]["id"] == "first"
    assert tune.rank([a, dict(b, ndcg_at_10=.200000000001, recall_at_10=.1)])[0]["id"] == "second"


def test_restart_freeze_and_score_without_real_november(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    # All transactions, including the held-out purchases in this test, are synthetic.
    ledger, credits = ledger_fixture()
    extra = ledger.iloc[:2].copy()
    extra["ExcelRow"] = [101, 102]
    extra["InvoiceNo"] = ["101", "102"]
    extra["InvoiceDate"] = pd.Timestamp("2011-11-29")
    extra["StockCode"] = ["A", "NOV_ONLY"]
    ledger = pd.concat([ledger, extra], ignore_index=True)
    root, data = tmp_path / "results", tmp_path / "data/processed/d2"
    root.mkdir()
    data.mkdir(parents=True)
    ledger.to_csv(tmp_path / build_queries.LEDGER, index=False)
    credits.to_csv(tmp_path / build_queries.ALLOCATIONS, index=False)
    source = {p: {"sha256": train.digest(tmp_path / p)} for p in (build_queries.LEDGER, build_queries.ALLOCATIONS)}
    train.write_json(tmp_path / build_queries.D1_MANIFEST, {"recommendation_quantity_policy": build_queries.QUANTITY_POLICY,
                     "training_snapshot": {"end_exclusive": "2011-10-01"}, "outputs": source})
    build_queries.run(tmp_path)
    plan = tune.registration()
    configs = {c: dict(plan["configs"]["mf"][c], epochs=2) for c in ["01", "02", "03"]}
    plan["configs"] = {"mf": configs}
    monkeypatch.setattr(tune, "registration", lambda: plan)
    cfg = tmp_path / "baseline.yaml"
    cfg.write_text(yaml.safe_dump(configs["01"]))
    baseline = tmp_path / "pilot"
    baseline.mkdir()
    train.run_model(Namespace(model="mf", config=cfg, data_dir=data, seed=42, threads=2,
                              device="cpu", smoke=False, resume=False), baseline)
    args = Namespace(results_dir=root, data_dir=data, device="cpu", model="mf",
                     baseline_run=baseline, pilots_dir=None, stage="plan")
    tune.tune(args)
    with pytest.raises(ValueError, match="Missing full October"):
        freeze_experiment.freeze(root, data)
    with pytest.raises(ValueError, match="November is sealed"):
        evaluate.evaluate_locked(root, data, tmp_path, "mf", "cpu")
    assert not (root / "d6_test").exists()
    # Refuse an incompatible pilot before creating a formal result.
    path = baseline / "run.json"
    original = tune.read_json(path)
    args.stage = "screen"
    train.write_json(path, dict(original, smoke=True))
    with pytest.raises(ValueError, match="full October trial"):
        tune.tune(args)
    train.write_json(path, original)
    tune.tune(args)
    ledger_path = root / "tuning.csv"
    assert set(pd.read_csv(ledger_path).status) == {"failed", "complete"}
    successful = pd.read_csv(ledger_path).query("status == 'complete'")
    assert successful.optimizer.eq("adam").all() and successful.lr_scheduler.eq("none").all()
    imported = tune.read_json(tune.run_dir(root, "mf", "01", 42) / "run.json")
    assert imported["optimizer"] == "adam" and imported["lr_scheduler"] == "none"
    assert len(successful) == 3 and successful.origin.eq("verified_baseline").sum() == 1
    args.stage = "confirm"
    tune.tune(args)
    weights = tune.run_dir(root, "mf", "01", 42) / "best.pt"
    assert train.digest(weights) == train.digest(baseline / "best.pt")
    ledger_before = ledger_path.read_bytes()
    tune.tune(args)
    assert ledger_path.read_bytes() == ledger_before

    frozen = freeze_experiment.freeze(root, data)
    assert len(frozen["models"]["mf"]["weights"]) == 3
    assert freeze_experiment.freeze(root, data) == frozen
    with pytest.raises(ValueError, match="frozen"):
        tune.tune(args)
    assert not (root / "d6_test").exists()
    changed = deepcopy(frozen)
    changed["selected_model"] = "changed"
    train.write_json(root / "selection.json", changed)
    with pytest.raises(ValueError, match="differ from"):
        evaluate.evaluate_locked(root, data, tmp_path, "mf", "cpu")
    train.write_json(root / "selection.json", frozen)
    result = evaluate.evaluate_locked(root, data, tmp_path, "mf", "cpu")
    test_csv = Path(result["test_metrics"])
    before = test_csv.read_bytes()
    tables = root / "d6_test/tables"
    assert pd.read_parquet(tables / "queries.parquet").history_end.eq(pd.Timestamp("2011-10-01")).all()
    assert set(pd.read_parquet(tables / "labels.parquet").StockCode) == {"A", "NOV_ONLY"}
    assert "NOV_ONLY" not in set(pd.read_parquet(tables / "catalogs.parquet").StockCode)
    assert set(pd.read_parquet(root / "d6_test/mf/42/top10.parquet").split) == {"test"}
    assert pd.read_csv(test_csv).primary_queries.tolist() == [1]
    evaluate.evaluate_locked(root, data, tmp_path, "mf", "cpu")
    assert test_csv.read_bytes() == before
    assert not (root / "d6_test/tables/validation.parquet").exists()
