"""Build the evidence-linked cleaning audit, paid lines and basket-item events."""

import argparse
import gzip
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.baskets import build_basket_events
from src.data.clean import RAW_COLUMNS, WORKBOOK_SHA256, audit_transactions, read_workbook
from src.data.merchandise import merchandise_rules
from src.data.purchase_cleaning import clean_purchase_lines
from src.data.purchase_snapshot import purchase_line_balances, purchase_snapshot, snapshot_decisions
from src.data.reversals import compute_reversal_tables


TRAINING_CUTOFF = "2011-10-01"
BASELINE_PATH = "results/cleaning_audit.csv"


def snapshot_counts(purchases, events):
    return {
        "paid_merchandise_lines": len(purchases),
        "basket_item_events": len(events),
        "baskets": len(events[["CustomerID", "InvoiceNo"]].drop_duplicates()),
        "customers": int(purchases["CustomerID"].nunique()),
        "stock_codes": int(purchases["StockCode"].nunique()),
        "paid_units": int(purchases["Quantity"].sum()),
        "recorded_amount_gbp": float(format(purchases["line_amount_gbp"].sum(), ".15g")),
    }


def build_cleaning_outputs(raw, reversal_tables=None):
    """Build all D1 CSV tables in memory, keyed by relative output path, plus the manifest."""
    baseline = audit_transactions(raw)
    gross_steps, gross_decisions, _ = clean_purchase_lines(raw)
    reversal = compute_reversal_tables(raw) if reversal_tables is None else reversal_tables
    ledger, allocations = reversal["signed_ledger_a"], reversal["credit_allocations"]
    balances = purchase_line_balances(ledger, allocations)
    purchases = balances.loc[balances["Quantity"].gt(0)].copy()
    events = build_basket_events(purchases)
    steps, decisions = snapshot_decisions(
        raw, gross_steps, gross_decisions, ledger, allocations, balances,
    )
    train_lines, train_events = purchase_snapshot(ledger, allocations, cutoff=TRAINING_CUTOFF)
    frames = {
        BASELINE_PATH: baseline,
        "results/cleaning_steps.csv": steps,
        "results/merchandise_rules.csv": merchandise_rules(),
        "data/processed/row_decisions.csv.gz": decisions,
        "data/processed/purchase_lines.csv.gz": purchases,
        "data/processed/basket_events.csv.gz": events,
        "data/processed/training_purchase_lines.csv.gz": train_lines,
        "data/processed/training_basket_events.csv.gz": train_events,
        "data/processed/signed_ledger_a.csv.gz": ledger,
        "data/processed/signed_ledger_b.csv.gz": reversal["signed_ledger_b"],
        "results/credit_allocations.csv.gz": allocations,
    }
    for name in ["source_audit", "source_differences", "reconciliation", "region_summary",
                 "match_differences", "benchmark"]:
        frames[f"results/reversal_{name}.csv"] = reversal[name]

    manifest = {
        "source": {
            "dataset_doi": "10.24432/C5BW33", "workbook_sha256": WORKBOOK_SHA256,
            "sheet": "Online Retail", "raw_fields": RAW_COLUMNS, "excel_first_data_row": 2,
            "rows": len(raw), "first_timestamp": raw["InvoiceDate"].min().isoformat(),
            "last_timestamp": raw["InvoiceDate"].max().isoformat(),
        },
        "merchandise_evidence_period": {"start": "2010-12-01", "end_exclusive": TRAINING_CUTOFF},
        "counts": {
            "after_five_rules": int(baseline.iloc[-1]["rows_after"]),
            "gross_positive_lines_before_credits": len(balances),
            "removed_total": int(steps["rows_removed"].sum()),
            "fully_cancelled_sale_lines": int(balances["Quantity"].eq(0).sum()),
            "partially_cancelled_sale_lines": int((balances["cancelled_units"].gt(0) & balances["Quantity"].gt(0)).sum()),
            "cancelled_units": int(balances["cancelled_units"].sum()),
            "cancellation_amount_at_sale_prices_gbp": float(format((balances["cancelled_units"] * balances["UnitPrice"]).sum(), ".15g")),
            **snapshot_counts(purchases, events),
        },
        "training_snapshot": {
            "start_inclusive": "2010-12-01", "end_exclusive": TRAINING_CUTOFF,
            "credit_time_end_exclusive": TRAINING_CUTOFF, **snapshot_counts(train_lines, train_events),
        },
        "event_key": ["CustomerID", "InvoiceNo", "StockCode"],
        "basket_time": "Latest original positive eligible merchandise line timestamp on the customer invoice, fixed before credit adjustments.",
        "scope": "Full exports are retrospective. Training exports replay only baskets and credits strictly before 2011-10-01. For any earlier or target cutoff use purchase_snapshot on the signed ledger and canonical allocations.",
        "amounts": "Recommendation lines use surviving units at original sale prices. Signed A/B ledgers retain each original sale/credit quantity and actual price. Do not subtract matched cancellations again from ledger net amounts.",
    }
    return frames, manifest


def save_cleaning_results(project, baseline_path, frames, manifest):
    """Stage each artifact beside its destination, then publish the checksum manifest last."""
    project, baseline_path = Path(project), Path(baseline_path)
    results = project / "results"
    frames = [(baseline_path if name == BASELINE_PATH else project / name, frame)
              for name, frame in frames.items()]
    manifest_path = results / "data_manifest.json"
    if baseline_path.suffix.lower() != ".csv":
        raise ValueError("Choose a .csv file for the five-rule audit.")
    if len({path.resolve() for path, _ in frames}) != len(frames):
        raise ValueError("Choose distinct output paths for each table.")

    with ExitStack() as stack:
        workdirs = {}
        for directory in {path.parent for path, _ in frames}:
            directory.mkdir(parents=True, exist_ok=True)
            workdirs[directory] = Path(stack.enter_context(TemporaryDirectory(dir=directory)))
        outputs = {}
        staged = []
        for path, frame in frames:
            temp = workdirs[path.parent] / path.name
            if path.suffix == ".gz":
                with temp.open("wb") as stream:
                    with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0, compresslevel=6) as zipped:
                        frame.to_csv(zipped, index=False, float_format="%.15g")
            else:
                frame.to_csv(temp, index=False)
            with temp.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            outputs[os.path.relpath(path, project)] = {"rows": len(frame), "sha256": digest}
            staged.append((temp, path))
        manifest_temp = workdirs[results] / manifest_path.name
        manifest_temp.write_text(json.dumps({**manifest, "outputs": outputs}, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        for temp, path in staged:
            temp.replace(path)
        manifest_temp.replace(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("."), help="Root for the workbook and generated files.")
    parser.add_argument("--workbook", type=Path, help="Defaults to data/raw/Online Retail.xlsx in the project.")
    parser.add_argument("--output", type=Path, help="Override the path of the original five-rule audit CSV.")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    raw = read_workbook(args.workbook or project / "data/raw/Online Retail.xlsx")
    frames, manifest = build_cleaning_outputs(raw)
    save_cleaning_results(project, args.output or project / BASELINE_PATH, frames, manifest)
    steps = frames["results/cleaning_steps.csv"]
    print(steps[["step", "rule", "raw_flagged", "rows_removed", "rows_after"]].to_string(index=False))
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Saved the output manifest to {project / 'results/data_manifest.json'}")


if __name__ == "__main__":
    main()
