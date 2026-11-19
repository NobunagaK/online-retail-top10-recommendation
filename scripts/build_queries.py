"""Generate the shared D2 monthly training and validation artifacts."""

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import pandas as pd

# Permit both python scripts/build_queries.py and python -m scripts.build_queries.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean import RAW_COLUMNS
from src.data.monthly_queries import DEFAULT_SEED, HISTORY_END, build_monthly_tables


D1_MANIFEST = "results/data_manifest.json"
LEDGER = "data/processed/signed_ledger_b.csv.gz"
ALLOCATIONS = "results/credit_allocations.csv.gz"
QUANTITY_POLICY = "net_retained_all_matched_credits_v1"
D2_PROTOCOL = "v3.4_monthly_net_retained_v1"
TABLE_DEFINITIONS = {
    "queries": "One eligible customer/month. Catalog, chronological history and in-catalog positive hashes use compact UTF-8 JSON. train_negative_count is zero for evaluation.",
    "labels": "All target-month distinct positive customer/SKU pairs after target-end credits, including ineligible customers and out-of-catalog SKUs. Only in-catalog rows receive an item_id. Main positives require eligible and in_catalog.",
    "cohorts": "All history, previous training-query, or active customers. has_valid_training_query refers to earlier eligible training months; for validation it covers all eight months. supplemental_eligible includes all active evaluation customers with a valid training query.",
    "coverage": "Customer counts overlap unless grouped by cohorts.coverage_status. Out-of-catalog share uses all active customer/SKU pairs. Cut counts use eligible queries with L=64.",
    "catalogs": "One row per eligible StockCode/cutoff, ordered by exact StockCode. All four models score this entire catalog for their eligible evaluation customers.",
    "histories": "One row per customer/cutoff. Full chronological item IDs and basket sizes are kept. L=64 fields describe model input truncation.",
    "item_vocab": "Append-only ID at first visible training snapshot. Ties on earliest retained basket time at the same snapshot use exact StockCode.",
    "item_support": "Source diagnostics: observed MF positive/negative training pairs; presence in a sequential training query's full-softmax candidate set. Actual supported parameter rows are tracked by each fitted model.",
    "train_pairs": "Contiguous eligible MF/LightGBM groups; all positives plus at most five unique negatives per positive, sampled from the StockCode-ordered historical catalog with seed and query ID.",
    "auc_pairs": "October evaluation only. All positives plus at most ten unique negatives per positive under the independent seed 5481. All models and training seeds share these pairs.",
    "customer_features": "One row per history customer/cutoff, with remaining units, purchase amount, distinct invoice counts, fixed-cutoff recency and 7/30/90-day invoice frequencies.",
    "item_features": "One row per historical StockCode/cutoff, with item aggregates and distinct buying customers.",
    "customer_item_features": "Sparse historical customer/SKU pairs. Unobserved pair frequencies are zero and time of last purchase is undefined.",
}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_inputs(project):
    """Read and verify D1's known-customer ledger and credit-to-sale links."""
    manifest_path = project / D1_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("recommendation_quantity_policy") != QUANTITY_POLICY
            or manifest.get("training_snapshot", {}).get("end_exclusive") != HISTORY_END.date().isoformat()):
        raise ValueError("D2 requires the v3.4 D1 net-retained purchase policy and October 1 cutoff.")
    hashes = {D1_MANIFEST: file_hash(manifest_path)}
    for name in (LEDGER, ALLOCATIONS):
        hashes[name] = file_hash(project / name)
        if hashes[name] != manifest["outputs"][name]["sha256"]:
            raise ValueError(f"D1 output differs from its source manifest: {name}")
    id_types = {k: "string" for k in ("InvoiceNo", "StockCode", "CustomerID")}
    ledger = pd.read_csv(project / LEDGER, usecols=["ExcelRow", *RAW_COLUMNS, "ledger_type"],
                         dtype=id_types, keep_default_na=False, na_values=[""], parse_dates=["InvoiceDate"])
    allocations = pd.read_csv(project / ALLOCATIONS, usecols=["credit_time", "purchase_excel_row", "credited_units", "status"],
                              dtype={"purchase_excel_row": "Int64"}, parse_dates=["credit_time"])
    return ledger, allocations, hashes


def verify_outputs(output_dir):
    """Return the manifest only if the complete published set has the declared digests."""
    output_dir = Path(output_dir)
    path = output_dir / "manifest.json"
    original = path.read_bytes()
    manifest = json.loads(original)
    expected = {f"{name}.parquet" if name != "coverage" else "coverage.csv" for name in TABLE_DEFINITIONS}
    if manifest.get("protocol") != D2_PROTOCOL or set(manifest.get("outputs", {})) != expected:
        raise ValueError("Incompatible D2 manifest. Rebuild the monthly queries.")
    for name, metadata in manifest["outputs"].items():
        artifact = output_dir / name
        if not artifact.is_file() or file_hash(artifact) != metadata["sha256"]:
            raise ValueError(f"Incomplete D2 generation ({name}). Rerun the builder to repair it.")
    if path.read_bytes() != original:
        raise ValueError("D2 changed during verification. Verify again after the build finishes.")
    return manifest


@contextmanager
def _publication_lock(output_dir):
    """Serialize local writers using the host temp filesystem, including for a Drive destination."""
    key = hashlib.sha256(str(output_dir.resolve()).encode("utf-8")).hexdigest()
    lock = Path(gettempdir()) / f"retail-d2-{key}.lock"
    # The kernel releases a file lock on process exit. The empty lock file is
    # reusable; its continued existence never blocks another invocation.
    with lock.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def save_outputs(output_dir, tables, input_hashes, seed):
    """Write all files in staging, then publish data files before their digest manifest."""
    import pyarrow

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": D2_PROTOCOL, "recommendation_quantity_policy": QUANTITY_POLICY,
        "python": sys.version.split()[0], "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
        "input_sha256": input_hashes, "train_negative_seed": int(seed), "auc_sample_seed": 5481,
        "max_training_negatives_per_positive": 5, "max_auc_negatives_per_positive": 10,
        "shared_initial_sequence_length": 64, "first_train_target": "2011-02-01",
        "last_train_target": "2011-09-01", "frozen_evaluation_history_end": "2011-10-01",
        "last_target_end_exclusive": "2011-11-01", "splits": ["train", "validation"],
        "test_gate": "November labels, cohorts and AUC samples are reserved for D6 after selection.json is frozen.",
        "outputs": {},
    }
    with TemporaryDirectory(dir=output_dir) as tmp:
        stage = Path(tmp)
        for name, table in tables.items():
            if name == "coverage":
                path = stage / "coverage.csv"
                table.to_csv(path, index=False, float_format="%.15g")
            else:
                path = stage / f"{name}.parquet"
                table.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
            manifest["outputs"][path.name] = {
                "rows": len(table), "columns": list(table.columns), "sha256": file_hash(path),
                "definition": TABLE_DEFINITIONS[name],
            }
        marker = stage / "manifest.json"
        marker.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with _publication_lock(output_dir):
            for filename, metadata in manifest["outputs"].items():
                published = output_dir / filename
                if not published.is_file() or file_hash(published) != metadata["sha256"]:
                    (stage / filename).replace(published)
            marker.replace(output_dir / "manifest.json")
            # Verify Drive-mounted writes and catch a competing remote writer.
            verified = verify_outputs(output_dir)
            if verified != manifest:
                raise ValueError("A different D2 generation was published concurrently. Rerun with one writer.")
            return verified


def run(project_root, output_dir=None, *, seed=DEFAULT_SEED):
    """Verify D1 inputs, replay all development months, and return the saved manifest."""
    project = Path(project_root).resolve()
    output = Path(output_dir) if output_dir is not None else project / "data/processed/d2"
    ledger, allocations, hashes = load_inputs(project)
    tables = build_monthly_tables(ledger, allocations, seed=seed)
    return save_outputs(output, tables, hashes, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="MF/LightGBM training negative-sampling seed")
    args = parser.parse_args()
    manifest = run(args.project_root, args.output_dir, seed=args.seed)
    output = args.output_dir or (args.project_root / "data/processed/d2")
    print(f"{manifest['outputs']['queries.parquet']['rows']:,} train/validation queries; {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
