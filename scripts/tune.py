"""Screen the registered D6 configurations in October and confirm each model's top two."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from math import isclose, isfinite
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import train
from scripts.build_queries import verify_outputs
from src.data.monthly_queries import ordered_hash
from src.evaluation.ranking import summarize_metrics

ROOT = train.ROOT
GRID = ROOT / "configs/d6_search.yaml"
METRICS = ("ndcg_at_10", "recall_at_10", "sampled_auc")
# The checked D5 trainers differ only on files irrelevant to the CPU baselines.
D5_AUTHOR_SOURCE = "79344136b969a3190f4babfd62f87706e9871e8ba58227c1d922d1fd852d5c3f"
D5_CPU_SOURCE = "d3c7ee544439f8dd39751b4d22a8fa33fe3f3e537762fdd077ba0340e5e17825"


def registration():
    grid = yaml.safe_load(GRID.read_text())
    configs = {}
    for model, rows in grid["models"].items():
        base = yaml.safe_load((ROOT / "configs" / f"{model}.yaml").read_text())
        configs[model] = {}
        for number, row in enumerate(rows, 1):
            defaults = grid["sequence_tuning_defaults"] if number > 1 and model in {"lrurec", "mamba4rec"} else {}
            configs[model][f"{number:02}"] = {**base, **defaults, **row}
    return {"configs": configs, "seeds": grid["seeds"], "finalists": grid["finalists"]}


def development_inputs(directory):
    manifest = verify_outputs(directory)
    if manifest["splits"] != ["train", "validation"] or manifest["auc_sample_seed"] != 5481:
        raise ValueError("D6 tuning requires the D2 development splits and fixed AUC sample.")
    queries = pd.read_parquet(directory / "queries.parquet")
    counts = pd.read_csv(directory / "coverage.csv").query("split == 'validation'").iloc[0]
    result = {"manifest_sha256": train.digest(directory / "manifest.json"),
              "validation_candidates": int(counts.catalog_items), "protocol": manifest["protocol"]}
    for split, name in [("train", "training"), ("validation", "validation")]:
        ids = queries.loc[queries.split.eq(split), "query_id"]
        result.update({f"{name}_queries": len(ids), f"{name}_queries_sha256": ordered_hash(ids)})
    return result


def registration_record(plan, inputs):
    files = ("scripts/tune.py", "scripts/freeze_experiment.py", "scripts/evaluate.py",
             "src/data/purchase_snapshot.py", "src/data/baskets.py", "src/data/clean.py",
             "third_party/mamba4rec/source.json", "third_party/mamba4rec/mamba4rec.py")
    return {"plan": plan, "development": inputs, "trainer_sha256": train.source_digest(),
            "protocol_files": {name: train.digest(ROOT / name) for name in files}}


def read_json(path):
    return json.loads(path.read_text())


def save_fixed(path, value):
    """Re-execution may verify an existing registration or freeze, but cannot replace it."""
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Experiment differs from its saved {path.name}. Use a separate results directory.")
    else:
        train.write_json(path, value)


def run_dir(root, model, config_id, seed):
    return root / "d6_runs" / model / config_id / str(seed)


def best_file(model):
    return "best.txt" if model == "lightgbm" else "best.pt"


def verify_run(path, model, cid, seed, config, inputs, source):
    """Verify the same D2 cohort, effective configuration, scored evidence and saved weights."""
    run = read_json(path / "run.json")
    allowed_sources = {source}
    if cid == "01" and seed == 42:
        allowed_sources.add(D5_AUTHOR_SOURCE)
        if model in {"mf", "lightgbm"}:
            allowed_sources.add(D5_CPU_SOURCE)
    expected = {**inputs, "model": model, "seed": seed, "config": {**config, "seed": seed}, "smoke": False,
                "status": "complete", "split": "validation", "reload_identical_top10": True}
    if run.get("source_sha256") not in allowed_sources or any(run.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Run does not match the registered full October trial: {path}")
    if model in {"lrurec", "mamba4rec"}:
        if model == "lrurec":
            from src.models import lrurec_source
            commit, sha = lrurec_source.UPSTREAM_COMMIT, lrurec_source.SOURCE_SHA256
        else:
            origin = read_json(ROOT / "third_party/mamba4rec/source.json")
            commit, sha = origin["commit"], origin["sha256"]["mamba4rec.py"]
        upstream = run.get("upstream_model", {})
        if upstream.get("commit") != commit or upstream.get("source_sha256") != sha:
            raise ValueError(f"Unexpected upstream recommender in {path}")
    expected_files = {best_file(model), "top10.parquet", "auc_scores.parquet", "query_metrics.csv", "metrics.csv"}
    hashes = run["output_sha256"]
    if not expected_files.issubset(hashes):
        raise ValueError(f"Missing required evidence in {path}")
    for name, sha in hashes.items():
        if Path(name).name != name or train.digest(path / name) != sha:
            raise ValueError(f"Artifact checksum mismatch: {path} / {name}")
    rows = pd.read_csv(path / "query_metrics.csv")
    primary_ids = rows.loc[rows.catalog_positive_items.gt(0), "query_id"]
    if rows.query_id.duplicated().any() or ordered_hash(primary_ids) != inputs["validation_queries_sha256"]:
        raise ValueError(f"Inconsistent validation query IDs in {path}")
    metrics = summarize_metrics(rows)
    for field, value in metrics.items():
        stored = run["validation_metrics"][field]
        if value is None or stored is None:
            matches = value is stored
        else:
            matches = isclose(value, stored, rel_tol=1e-12, abs_tol=1e-14)
        if not matches:
            raise ValueError(f"Validation evidence disagrees with {field} in {path}")
    if not all(isfinite(run["validation_metrics"][field]) for field in METRICS):
        raise ValueError("All three formal October metrics must be finite.")
    return run


def hardware_key(run):
    hw = run["hardware"]
    model = hw.get("gpu") if run["device"] == "cuda" else hw.get("cpu_model")
    return (run["device"], model, run["threads"]) if model else None


def rank(options):
    """Options arrive in registration order. Apply runtime only within a comparable score tie."""
    groups = defaultdict(list)
    for option in options:
        groups[(option["ndcg_at_10"], option["recall_at_10"])].append(option)
    result = []
    for scores in sorted(groups, reverse=True):
        tied = groups[scores]
        devices = {key for option in tied for key in option["hardware_keys"]}
        if len(devices) == 1 and None not in devices:
            tied = sorted(tied, key=lambda option: option["train_seconds"])
        result.extend(tied)
    return result


def screen_order(records):
    options = [{"id": cid, **info["validation_metrics"], "train_seconds": info["train_seconds"],
                "hardware_keys": [hardware_key(info)]} for cid, info in records.items()]
    return [option["id"] for option in rank(options)]


def export_ledger(root):
    records = [read_json(path) for path in sorted((root / "d6_runs").glob("*/*/attempts/*.json"))]
    if records:
        table = pd.DataFrame(records).sort_values(["model", "config_id", "seed", "started_utc"])
        train.write_atomic(root / "tuning.csv", lambda p: table.to_csv(p, index=False))


def existing_baseline(path, candidates_dir, model, config, inputs, source):
    if path:
        candidate = path if (path / "run.json").is_file() else path / "model"
        verify_run(candidate, model, "01", 42, config, inputs, source)
        return candidate
    if candidates_dir:
        # Earliest qualifying run, independent of the observed score.
        for candidate in sorted((candidates_dir / model).glob("*_baseline_*/model/run.json")):
            try:
                verify_run(candidate.parent, model, "01", 42, config, inputs, source)
            except (ValueError, KeyError, OSError) as exc:
                print(f"Skipping ineligible baseline {candidate.parent}: {exc}", flush=True)
            else:
                return candidate.parent
    return None


def fit_one(args, cid, seed, config, inputs, source):
    root, model = args.results_dir, args.model
    out = run_dir(root, model, cid, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists() and read_json(out / "run.json").get("status") == "complete":
        info = verify_run(out, model, cid, seed, config, inputs, source)
        # Recover if a disconnect occurred after publishing weights but before logging completion.
        attempts = sorted((out.parent / "attempts").glob(f"{seed}-*.json"))
        if attempts:
            last = read_json(attempts[-1])
            if last["status"] != "complete":
                finish_attempt(attempts[-1], last, info)
        print(f"{model}/{cid}/{seed}: verified existing result", flush=True)
        return info
    started = datetime.now(timezone.utc).isoformat()
    logdir = out.parent / "attempts"
    logdir.mkdir(exist_ok=True)
    attempt = logdir / f"{seed}-{started.replace(':', '')}-{uuid4().hex[:8]}.json"
    log = {"model": model, "config_id": cid, "seed": seed, "started_utc": started, "status": "running",
           "config_json": json.dumps({**config, "seed": seed}, sort_keys=True), "origin": "fit"}
    for old in logdir.glob(f"{seed}-*.json"):
        record = read_json(old)
        if record["status"] == "running":
            train.write_json(old, {**record, "status": "interrupted", "error": "Previous session ended before completion"})
    train.write_json(attempt, log)
    export_ledger(root)
    try:
        baseline = None
        if cid == "01" and seed == 42 and not out.exists():
            baseline = existing_baseline(args.baseline_run, args.pilots_dir, model, config, inputs, source)
        if baseline:
            log.update(origin="verified_baseline", imported_from=str(baseline))
            train.write_json(attempt, log)
            with TemporaryDirectory(dir=out.parent) as tmp:
                copy = Path(tmp) / "model"
                copy.mkdir()
                for name in ["run.json", *read_json(baseline / "run.json")["output_sha256"]]:
                    shutil.copyfile(baseline / name, copy / name)
                info = verify_run(copy, model, cid, seed, config, inputs, source)
                copy.rename(out)
        else:
            out.mkdir(exist_ok=True)
            cfg = out.parent / "config.yaml"
            train.write_atomic(cfg, lambda p: p.write_text(yaml.safe_dump(config)))
            params = argparse.Namespace(model=model, seed=seed, config=cfg, device=args.device, threads=2,
                                        smoke=False, resume=True, data_dir=args.data_dir, report_epochs=True)
            print(f"Training {model}/{cid}/{seed} on October's development protocol", flush=True)
            train.run_model(params, out)
            info = verify_run(out, model, cid, seed, config, inputs, source)
        finish_attempt(attempt, log, info)
        return info
    except Exception as exc:
        train.write_json(attempt, {**log, "status": "failed", "error": str(exc)})
        raise
    finally:
        export_ledger(root)


def finish_attempt(path, log, run):
    train.write_json(path, {**log, "status": "complete", **run["validation_metrics"], "device": run["device"],
                            "hardware": str(hardware_key(run)), "best_epoch": run["best_epoch"],
                            "best_iteration": run.get("best_iteration"), "train_seconds": run["train_seconds"],
                            "source_sha256": run["source_sha256"], "signature": run["signature"]})


def tune(args):
    plan = registration()
    inputs = development_inputs(args.data_dir)
    root = args.results_dir
    root.mkdir(parents=True, exist_ok=True)
    record = registration_record(plan, inputs)
    with train.run_lock(root):
        save_fixed(root / "d6_registration.json", record)
    if args.stage == "plan":
        return record
    if (root / "selection.json").exists():
        raise ValueError("This experiment is frozen. Use evaluate.py to score its selected checkpoints.")
    configs, seeds = plan["configs"][args.model], plan["seeds"]
    source = record["trainer_sha256"]
    # A training session owns one model. Separate models write separate fit artifacts.
    with train.run_lock(root / "d6_runs" / args.model):
        screen = {}
        for cid, config in configs.items():
            if args.stage == "confirm":
                screen[cid] = verify_run(run_dir(root, args.model, cid, seeds[0]), args.model, cid,
                                         seeds[0], config, inputs, source)
            else:
                screen[cid] = fit_one(args, cid, seeds[0], config, inputs, source)
        top = screen_order(screen)[:plan["finalists"]]
        if args.stage in {"all", "confirm"}:
            for cid in top:
                for seed in seeds[1:]:
                    fit_one(args, cid, seed, configs[cid], inputs, source)
        export_ledger(root)
    return {"model": args.model, "stage": args.stage, "screen_finalists": top}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=train.MODULES)
    parser.add_argument("--stage", choices=["plan", "screen", "confirm", "all"], default="all")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/d2")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--baseline-run", type=Path, help="Existing full seed-42 baseline session or its model directory")
    parser.add_argument("--pilots-dir", type=Path, help="Automatically check baseline sessions under this directory")
    args = parser.parse_args()
    if args.stage != "plan" and not args.model:
        parser.error("--model is required for training")
    if args.device is None:
        args.device = "cuda" if args.model in {"lrurec", "mamba4rec"} else "cpu"
    print(json.dumps(tune(args), indent=2))


if __name__ == "__main__":
    main()
