"""Train one recommendation model on D2 and score October validation on its complete catalog."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, gettempdir
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.data.monthly_queries import ordered_hash
from src.evaluation.ranking import evaluate_query, ranking_metrics, summarize_metrics
from src.models.pilot_data import PilotData

MODULES = {
    "mf": "mf", "lightgbm": "lightgbm_ranker",
    "lrurec": "lrurec_adapter", "mamba4rec": "mamba4rec_adapter",
}


def write_atomic(path, writer):
    with TemporaryDirectory(dir=path.parent) as stage:
        tmp = Path(stage) / path.name
        writer(tmp)
        tmp.replace(path)


def write_json(path, value):
    write_atomic(path, lambda p: p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@contextmanager
def run_lock(directory):
    """Lock a run on its host's local disk, including when its artifacts live on Drive."""
    lock_id = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()
    with (Path(gettempdir()) / f"retail-model-{lock_id}.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def evaluate(model, data):
    raw_pairs = {key: frame for key, frame in data.auc_pairs.groupby("query_id", sort=False)}
    catalogs = {key: frame for key, frame in data.catalogs.groupby("catalog_id", sort=False)}
    tops, samples, rows = [], [], []
    # No training fit or final-test split enters this scorer.
    for query in data.val_queries.itertuples(index=False):
        ids = data.catalog_ids[query.catalog_id]
        scores = pd.DataFrame({"item_id": ids, "score": model.score(query, ids)})
        top, sampled, metrics = evaluate_query(catalogs[query.catalog_id], scores,
                                               data.positives[query.query_id], raw_pairs[query.query_id],
                                               all_positive_count=query.full_positive_count)
        tops.append(top.assign(query_id=query.query_id, split="validation"))
        samples.append(sampled)
        rows.append(dict(query_id=query.query_id, **metrics))
    for query in data.out_of_catalog_only.itertuples(index=False):
        metrics = ranking_metrics([], [], query.full_positive_count)
        rows.append(dict(query_id=query.query_id, **metrics, sampled_auc=None, auc_requested=False,
                         auc_positive_items=0, auc_negative_items=0))
    return (summarize_metrics(rows), pd.concat(tops, ignore_index=True),
            pd.concat(samples, ignore_index=True), pd.DataFrame(rows))


def check_reloaded(reference, actual, keys):
    if not reference[keys].equals(actual[keys]):
        raise ValueError("Reload changed the ranked items or the fixed AUC sample.")
    before, after = reference.score.to_numpy(), actual.score.to_numpy()
    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)
    return float(np.abs(before - after).max(initial=0))


def fallback_counts(data, top, supported):
    """Count validation catalog items, query positives and Top-10 hits using a mean fallback."""
    candidates = {item for key in data.val_queries.catalog_id.unique() for item in data.catalog_ids[key]}
    unknown = candidates.difference(supported)
    positives = sum(len(set(data.positives[q]).intersection(unknown)) for q in data.val_queries.query_id)
    hits = sum(row.item_id in data.positives[row.query_id]
               for row in top.loc[top.item_id.isin(unknown)].itertuples(index=False))
    return {"candidate_items": len(unknown), "positive_query_items": positives, "top10_hits": hits}


def run_model(args, out):
    config_file = args.config or ROOT / "configs" / f"{args.model}.yaml"
    config = yaml.safe_load(config_file.read_text())
    seed = int(config.get("seed", 42) if args.seed is None else args.seed)
    config["seed"] = seed
    if args.smoke:
        config["epochs"] = 1
        if args.model == "lightgbm":
            config["n_estimators"] = min(12, config["n_estimators"])
    epochs = int(config.get("epochs", 1))
    patience = int(config.get("patience", 10))
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must both be positive.")
    if args.model == "lightgbm" and epochs != 1:
        raise ValueError("LightGBM trains its boosting rounds in a single fit.")
    device = args.device or ("cuda" if args.model == "mamba4rec" else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    versions = {}
    for pkg in ("torch", "lightgbm", "scikit-learn", "numpy", "pandas", "pyarrow", "mamba-ssm", "causal-conv1d"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    hardware = {"device": device, "cpu": platform.processor() or platform.machine(),
                "python": platform.python_version(), "packages": versions}
    if args.model != "lightgbm":
        import torch
        torch.set_num_threads(args.threads)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Select an NVIDIA GPU runtime to use --device cuda.")
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
            hardware.update(gpu=torch.cuda.get_device_name(), torch_cuda=torch.version.cuda)
    if args.model == "lightgbm":
        if device != "cpu":
            raise ValueError("The project's LightGBM baseline uses --device cpu.")
        config["n_jobs"] = args.threads
    data = PilotData(args.data_dir, smoke=args.smoke)
    if not data.positives:
        raise ValueError("D2 eligible labels must be present.")
    signature_fields = {
        "model": args.model, "config": config, "device": device, "threads": args.threads,
        "smoke": args.smoke, "manifest_sha256": digest(args.data_dir / "manifest.json"),
        "training_queries_sha256": ordered_hash(data.train_queries.query_id),
        "validation_queries_sha256": ordered_hash(data.val_queries.query_id),
    }
    sources = [Path(__file__).resolve(), ROOT / "src/evaluation/ranking.py",
               *sorted((ROOT / "src/models").glob("*.py")), *sorted((ROOT / "src/features").glob("*.py"))]
    signature_fields["source_sha256"] = ordered_hash([(str(p.relative_to(ROOT)), digest(p)) for p in sources])
    signature = hashlib.sha256(json.dumps(signature_fields, sort_keys=True).encode()).hexdigest()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    base = dict(**signature_fields, signature=signature, seed=seed, hardware=hardware,
                git_commit=revision.stdout.strip() or None, protocol=data.manifest["protocol"],
                training_queries=len(data.train_queries), validation_queries=len(data.val_queries),
                training_pair_rows=len(data.train_pairs), validation_candidates=len(data.catalog_ids[data.val_queries.iloc[0].catalog_id]),
                split="validation", started_utc=datetime.now(timezone.utc).isoformat())
    progress_path = out / "progress.json"
    run_path = out / "run.json"
    if run_path.exists():
        state = json.loads(run_path.read_text())
        if not args.resume or state["signature"] != signature:
            raise ValueError("Choose a new run directory, or resume the same model, code, inputs and configuration.")
        if state["status"] == "complete":
            for name, sha in state["output_sha256"].items():
                if digest(out / name) != sha:
                    raise ValueError(f"Completed run artifact failed verification: {name}")
            return state
        base["started_utc"] = state["started_utc"]
    elif any(out.iterdir()):
        raise ValueError("A new run requires an empty output directory.")
    write_json(run_path, dict(base, status="running"))
    suffix = ".txt" if args.model == "lightgbm" else ".pt"
    try:
        factory = importlib.import_module(f"src.models.{MODULES[args.model]}").make_model
        model = factory(data, config, device)
        rows, best, best_epoch = [], -math.inf, 0
        def state_file(epoch):
            return out / f"epoch-{epoch}{suffix}"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            rows, best_epoch = progress["rows"], progress["best_epoch"]
            best = rows[best_epoch-1]["ndcg_at_10"]
            model.restore(state_file(len(rows)))
        for epoch in range(len(rows)+1, epochs+1):
            if best_epoch and len(rows) - best_epoch >= patience:
                break
            start = perf_counter()
            objective = float(model.train_epoch(data))
            train_time = perf_counter() - start
            if not math.isfinite(objective):
                raise ValueError("Training produced a nonfinite value.")
            start = perf_counter()
            metrics, top, sample, _ = evaluate(model, data)
            row = {"epoch": epoch, "train_seconds": train_time, "score_seconds": perf_counter()-start,
                   "recall_at_10": metrics["recall_at_10"], "ndcg_at_10": metrics["ndcg_at_10"],
                   "sampled_auc": metrics["sampled_auc"],
                   ("boosting_validation_ndcg_at_10" if args.model == "lightgbm" else "training_loss"): objective}
            rows.append(row)
            previous_best = best_epoch
            write_atomic(state_file(epoch), model.save)
            if args.model == "lightgbm":
                write_json(out / "boosting_history.json", model.validation_history)
            if row["ndcg_at_10"] > best:
                best, best_epoch = row["ndcg_at_10"], epoch
                top.to_parquet(out / f"reference-{epoch}-top.parquet", index=False)
                sample.to_parquet(out / f"reference-{epoch}-auc.parquet", index=False)
            # Commit only after the checkpoint and its score references exist.
            write_json(progress_path, {"rows": rows, "best_epoch": best_epoch})
            write_atomic(out / "metrics.csv", lambda p: pd.DataFrame(rows).to_csv(p, index=False))
            for old in {epoch-1, previous_best} - {0, epoch, best_epoch}:
                state_file(old).unlink(missing_ok=True)
                for name in ("top", "auc"):
                    (out / f"reference-{old}-{name}.parquet").unlink(missing_ok=True)
        best_path = out / f"best{suffix}"
        write_atomic(best_path, lambda p: shutil.copyfile(state_file(best_epoch), p))
        if args.model != "lightgbm":
            write_atomic(out / "last.pt", lambda p: shutil.copyfile(state_file(len(rows)), p))
        # A fresh instance must reproduce the best epoch's original predictions.
        restored = factory(data, config, device)
        restored.restore(best_path)
        metrics, top, sampled, query_rows = evaluate(restored, data)
        delta_top = check_reloaded(pd.read_parquet(out / f"reference-{best_epoch}-top.parquet"), top,
                                   ["query_id", "rank", "item_id"])
        delta_auc = check_reloaded(pd.read_parquet(out / f"reference-{best_epoch}-auc.parquet"), sampled,
                                   ["query_id", "item_id"])
        for table, name in [(top, "top10"), (sampled, "auc_scores")]:
            write_atomic(out / f"{name}.parquet",
                         lambda p: table.assign(model=args.model, seed=seed).to_parquet(p, index=False))
        write_atomic(out / "query_metrics.csv", lambda p: query_rows.to_csv(p, index=False))
        write_atomic(out / "metrics.csv", lambda p: pd.DataFrame(rows).to_csv(p, index=False))
        files = ["top10.parquet", "auc_scores.parquet", "query_metrics.csv", "metrics.csv", best_path.name]
        if args.model != "lightgbm":
            files.append("last.pt")
        if args.model == "lightgbm":
            files.append("boosting_history.json")
        result = dict(base, status="complete", best_epoch=best_epoch, epochs_completed=len(rows),
                      validation_metrics=metrics, reload_identical_top10=True,
                      reload_max_score_delta=max(delta_top, delta_auc),
                      train_seconds=sum(r["train_seconds"] for r in rows),
                      full_catalog_score_seconds=rows[best_epoch-1]["score_seconds"],
                      output_sha256={name: digest(out / name) for name in files})
        if hasattr(model, "best_iteration"):
            result["best_iteration"] = int(model.best_iteration)
        if hasattr(model, "last_encoder_grad_norm"):
            result["last_epoch_max_gradient_norms"] = {
                "encoder": float(model.last_encoder_grad_norm), "scoring_head": float(model.last_head_grad_norm)}
        if hasattr(restored, "supported_item_ids"):
            result["embedding_mean_fallback"] = fallback_counts(data, top, restored.supported_item_ids)
        if args.model == "lrurec":
            supported_bias = set(restored.catalog_support.nonzero().flatten().tolist())
            result["item_bias_mean_fallback"] = fallback_counts(data, top, supported_bias)
        if hasattr(model, "factors"):
            result["parameter_count"] = sum(p.numel() for p in model.factors.parameters())
        if hasattr(model, "parameter_count"):
            result["parameter_count"] = int(model.parameter_count)
        if hasattr(model, "real_parameter_count"):
            result["real_scalar_parameter_count"] = int(model.real_parameter_count)
        if hasattr(model, "support_summary"):
            result["training_support"] = model.support_summary()
        if device == "cuda":
            result["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated())
        write_json(run_path, result)
        for path in [progress_path, state_file(len(rows)), state_file(best_epoch),
                     out / f"reference-{best_epoch}-top.parquet", out / f"reference-{best_epoch}-auc.parquet"]:
            path.unlink(missing_ok=True)
        return result
    except Exception as error:
        write_json(run_path, dict(base, status="failed", error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODULES)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/d2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true", help="One deterministic small training/validation pilot")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    out = args.output_dir or ROOT / "runs" / args.model / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        result = run_model(args, out)
    print(f"{args.model} / {result['device']}: {result['validation_queries']} validation queries; {out / 'run.json'}")
    print(json.dumps(result["validation_metrics"], indent=2))


if __name__ == "__main__":
    main()
