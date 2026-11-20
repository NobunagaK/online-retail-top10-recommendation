"""Score the locked November suite from the twelve October-selected checkpoints."""

import argparse
import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import train, tune
from scripts.build_queries import load_inputs
from scripts.freeze_experiment import summarize_seeds
from src.data.monthly_queries import HISTORY_END, TRAIN_START, VALIDATION_END, build_query_window, pair_tables
from src.data.purchase_snapshot import purchase_snapshot
from src.models.pilot_data import PilotData


def frozen_inputs(root, data_dir):
    marker, checksum, registration = (root / n for n in ("selection.json", "selection_digest.json", "d6_registration.json"))
    if not all(p.is_file() for p in (marker, checksum, registration)):
        raise ValueError("November is sealed. Complete the October search and run freeze_experiment.py first.")
    key = train.digest(marker)
    frozen = tune.read_json(marker)
    plan = tune.registration()
    reg = tune.registration_record(plan, tune.development_inputs(data_dir))
    if (tune.read_json(checksum) != {"sha256": key} or train.digest(registration) != frozen["registration_sha256"]
            or reg != tune.read_json(registration) or frozen["split_for_selection"] != "validation"
            or set(frozen["models"]) != set(plan["configs"]) or frozen["seeds"] != plan["seeds"]):
        raise ValueError("Selection, code or development data differ from the frozen experiment.")
    for model, case in frozen["models"].items():
        cid = case["config_id"]
        if (case["config"] != plan["configs"][model][cid]
                or [w["seed"] for w in case["weights"]] != plan["seeds"]):
            raise ValueError("Incomplete frozen configuration or seeds.")
        for weight in case["weights"]:
            expected = tune.run_dir(root, model, cid, weight["seed"]) / tune.best_file(model)
            if weight["path"] != str(expected.relative_to(root)) or train.digest(expected.parent / "run.json") != weight["run_sha256"]:
                raise ValueError(f"Frozen run changed: {expected.parent}")
    return frozen, key


def november_tables(project, data):
    ledger, alloc, inputs = load_inputs(project)
    if inputs != data.manifest["input_sha256"]:
        raise ValueError("D1 inputs differ from the frozen development manifest.")
    end = pd.Timestamp("2011-12-01")
    _, history = purchase_snapshot(ledger, alloc, cutoff=HISTORY_END)
    _, targets = purchase_snapshot(ledger, alloc, cutoff=end)
    window = build_query_window("test", HISTORY_END, VALIDATION_END, end,
                                history.loc[history.basket_time.ge(TRAIN_START)], targets,
                                data.vocab, data.train_queries.customer_id)
    catalog = window["catalogs"]
    fixed = data.catalogs.loc[data.catalogs.history_end.eq(HISTORY_END), ["StockCode", "item_id"]]
    if list(catalog[["StockCode", "item_id"]].itertuples(index=False, name=None)) != list(fixed.itertuples(index=False, name=None)):
        raise ValueError("November candidates differ from the fixed validation catalog.")
    histories = data.read("histories")
    histories = histories.loc[histories.history_end.eq(HISTORY_END)].set_index("customer_id")
    queries = window["queries"]
    queries["history_sha256"] = queries.customer_id.map(histories.history_sha256)
    if queries.history_sha256.isna().any():
        raise ValueError("A test query has no frozen history.")
    _, window["auc_pairs"] = pair_tables(queries, window["labels"], catalog, data.manifest["train_negative_seed"])
    return window


def verify_files(root, receipt):
    for name, expected in receipt["output_sha256"].items():
        if Path(name).name != name or train.digest(root / name) != expected:
            raise ValueError(f"Frozen test output changed: {root / name}")


def test_tables(folder, key, project, data):
    receipt_path = folder / "manifest.json"
    if not receipt_path.exists():
        window = november_tables(project, data)
        folder.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=folder) as tmp:
            stage = Path(tmp)
            files = {}
            for name, table in window.items():
                path = stage / f"{name}.parquet"
                table.to_parquet(path, index=False)
                files[path.name] = train.digest(path)
            for name in files:
                (stage / name).replace(folder / name)
            train.write_json(receipt_path, {"selection_sha256": key, "output_sha256": files})
    receipt = tune.read_json(receipt_path)
    if receipt["selection_sha256"] != key:
        raise ValueError("Test labels belong to another frozen selection.")
    verify_files(folder, receipt)
    return {Path(name).stem: pd.read_parquet(folder / name) for name in receipt["output_sha256"]}


def score_seed(root, output, model, case, weight, device, data, window, key):
    folder = output / model / str(weight["seed"])
    marker = folder / "run.json"
    if marker.exists():
        record = tune.read_json(marker)
        if (record["selection_sha256"] != key or record["checkpoint_sha256"] != weight["sha256"]
                or record["seed"] != weight["seed"] or record["model"] != model):
            raise ValueError("An existing test run belongs to a different model, seed or selection.")
        verify_files(folder, record)
        return record
    model_module = importlib.import_module(f"src.models.{train.MODULES[model]}")
    predictor = model_module.make_model(data, {**case["config"], "seed": weight["seed"]}, device)
    predictor.restore(root / weight["path"])
    labels, cohorts = window["labels"], window["cohorts"]
    positives = labels.loc[labels.in_catalog & labels.eligible].groupby("query_id").item_id.agg(list).to_dict()
    metrics, top, sampled, rows = train.score_queries(predictor, data, window["queries"], positives,
                                                      window["auc_pairs"], cohorts.loc[cohorts.supplemental_eligible & ~cohorts.eligible])
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, frame in [("top10.parquet", top), ("auc_scores.parquet", sampled), ("query_metrics.parquet", rows)]:
        train.write_atomic(folder / name, lambda p: frame.to_parquet(p, index=False))
        files[name] = train.digest(folder / name)
    record = {"split": "test", "model": model, "seed": weight["seed"], "selection_sha256": key,
              "checkpoint_sha256": weight["sha256"], "device": device, "metrics": metrics, "output_sha256": files}
    train.write_json(marker, record)
    return record


def evaluate_locked(root, data_dir, project, model, device):
    frozen, key = frozen_inputs(root, data_dir)
    case = frozen["models"][model]
    for weight in case["weights"]:
        if train.digest(root / weight["path"]) != weight["sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {weight['path']}")
    data = PilotData(data_dir)
    output = root / "d6_test"
    with train.run_lock(output / "tables"):
        window = test_tables(output / "tables", key, project, data)
    histories = data.read("histories")
    lookup = histories.loc[histories.history_end.eq(HISTORY_END)].set_index("customer_id").history_item_ids
    data.histories_by_query.update(zip(window["queries"].query_id, window["queries"].customer_id.map(lookup)))
    if model != "lightgbm":
        import torch
        torch.set_num_threads(2)
    with train.run_lock(output / model):
        for weight in case["weights"]:
            score_seed(root, output, model, case, weight, device, data, window, key)
    # Publish the all-model table only when all twelve scores are available.
    with train.run_lock(output):
        all_rows = []
        for name, selected in frozen["models"].items():
            metrics = []
            for weight in selected["weights"]:
                folder = output / name / str(weight["seed"])
                if not (folder / "run.json").is_file():
                    return {"scored": model, "remaining_suite": True}
                rec = tune.read_json(folder / "run.json")
                if (rec["selection_sha256"] != key or rec["checkpoint_sha256"] != weight["sha256"]
                        or rec["model"] != name or rec["seed"] != weight["seed"]):
                    raise ValueError("Cannot combine test results from different frozen runs.")
                verify_files(folder, rec)
                metrics.append(rec["metrics"])
            all_rows.append({"model": name, "config_id": selected["config_id"], **summarize_seeds(metrics)})
        train.write_atomic(root / "test_metrics.csv", lambda p: pd.DataFrame(all_rows).to_csv(p, index=False))
    return {"scored": model, "test_metrics": str(root / "test_metrics.csv")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=train.MODULES, required=True)
    parser.add_argument("--results-dir", type=Path, default=tune.ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=tune.ROOT / "data/processed/d2")
    parser.add_argument("--project-root", type=Path, default=tune.ROOT, help="Project containing the original D1 artifacts")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    device = args.device or ("cuda" if args.model in {"lrurec", "mamba4rec"} else "cpu")
    print(json.dumps(evaluate_locked(args.results_dir, args.data_dir, args.project_root, args.model, device)))


if __name__ == "__main__":
    main()
