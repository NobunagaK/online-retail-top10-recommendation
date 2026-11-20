"""Freeze four configurations and their twelve checkpoints using October validation alone."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import train, tune

METRIC_FIELDS = (*tune.METRICS, "supplementary_recall_at_10")
COUNT_FIELDS = ("primary_queries", "supplementary_queries", "auc_valid_queries", "auc_skipped_queries",
                "auc_positive_items", "auc_negative_items")


def summarize_seeds(records):
    stats = {}
    for key in METRIC_FIELDS:
        values = [r[key] for r in records]
        if all(v is None for v in values):
            stats[f"{key}_mean"] = stats[f"{key}_sd"] = None
        elif any(v is None for v in values):
            raise ValueError(f"Evaluation coverage differs across seeds for {key}")
        else:
            stats[f"{key}_mean"] = mean(values)
            stats[f"{key}_sd"] = stdev(values)
    for key in COUNT_FIELDS:
        values = {r[key] for r in records}
        if len(values) != 1:
            raise ValueError(f"Training seeds differ on evaluation denominator: {key}")
        stats[key] = values.pop()
    return stats


def option(cid, runs):
    stats = summarize_seeds([r["validation_metrics"] for r in runs])
    return {"id": cid, "ndcg_at_10": stats["ndcg_at_10_mean"], "recall_at_10": stats["recall_at_10_mean"],
            "train_seconds": mean(r["train_seconds"] for r in runs),
            "hardware_keys": [tune.hardware_key(r) for r in runs], "summary": stats}


def freeze(root, data_dir):
    plan = tune.registration()
    reg_path = root / "d6_registration.json"
    input_record = tune.development_inputs(data_dir)
    reg = tune.registration_record(plan, input_record)
    if not reg_path.is_file() or tune.read_json(reg_path) != reg:
        raise ValueError("A matching D6 registration is required. Run the registered October search first.")
    source = reg["trainer_sha256"]
    seeds = plan["seeds"]
    models, choices, validation = {}, [], []
    for model, configs in plan["configs"].items():
        def checked(cid, seed):
            path = tune.run_dir(root, model, cid, seed)
            if not (path / "run.json").is_file():
                raise ValueError(f"Missing full October result for {model}/{cid}/{seed}. Finish tuning before freezing.")
            return tune.verify_run(path, model, cid, seed, configs[cid], input_record, source)
        screen = {cid: checked(cid, seeds[0]) for cid in configs}
        finalists = tune.screen_order(screen)[:plan["finalists"]]
        replicated = {cid: [screen[cid], *[checked(cid, s) for s in seeds[1:]]] for cid in sorted(finalists)}
        winner = tune.rank([option(cid, runs) for cid, runs in replicated.items()])[0]
        cid = winner["id"]
        weights = []
        for run in replicated[cid]:
            folder = tune.run_dir(root, model, cid, run["seed"])
            name = tune.best_file(model)
            weights.append({"seed": run["seed"], "path": str((folder / name).relative_to(root)),
                            "sha256": run["output_sha256"][name], "run_sha256": train.digest(folder / "run.json"),
                            "best_epoch": run["best_epoch"], "best_iteration": run.get("best_iteration"),
                            "git_commit": run["git_commit"]})
        models[model] = {"config_id": cid, "config": configs[cid], "weights": weights,
                         "finalists": {i: option(i, runs)["summary"] for i, runs in replicated.items()}}
        choices.append({**winner, "id": model})
        row = {"model": model, "config_id": cid, **winner["summary"]}
        for metric in tune.METRICS:
            baseline = screen["01"]["validation_metrics"][metric]
            result = screen[cid]["validation_metrics"][metric]
            row.update({f"baseline_42_{metric}": baseline, f"selected_42_{metric}": result,
                        f"delta_42_{metric}": result - baseline})
        validation.append(row)
    selected = tune.rank(choices)[0]["id"]
    selection = {"protocol": input_record["protocol"], "registration_sha256": train.digest(reg_path),
                 "trainer_sha256": source, "manifest_sha256": input_record["manifest_sha256"],
                 "selected_model": selected, "models": models, "seeds": seeds, "split_for_selection": "validation",
                 "history_end_exclusive": "2011-10-01", "test_target": ["2011-11-01", "2011-12-01"],
                 "auc_sample_seed": 5481}
    with train.run_lock(root):
        marker = root / "selection.json"
        tune.save_fixed(marker, selection)
        tune.export_ledger(root)
        train.write_atomic(root / "validation_metrics.csv", lambda p: pd.DataFrame(validation).to_csv(p, index=False))
        tune.save_fixed(root / "selection_digest.json", {"sha256": train.digest(marker)})
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=tune.ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=tune.ROOT / "data/processed/d2")
    args = parser.parse_args()
    result = freeze(args.results_dir, args.data_dir)
    print(json.dumps({"selected_model": result["selected_model"], "frozen": str(args.results_dir / "selection.json")}))


if __name__ == "__main__":
    main()
