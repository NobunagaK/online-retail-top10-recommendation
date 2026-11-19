"""Replay the training and validation periods with the UK's operational cutoff at 3 and 5 days."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from src.data.purchase_snapshot import net_retained_labels, purchase_snapshot
from src.data.reversals import match_reversals


LEDGER_PATH = "data/processed/signed_ledger_a.csv.gz"


def compare_thresholds(ledger):
    """Change the regional classification rule, rerun allocation, and compare complete outputs."""
    standard = match_reversals(ledger)
    alternate = match_reversals(ledger, uk_hours=120)
    # The threshold and operational status are the only fields allowed to change.
    pd.testing.assert_frame_equal(
        standard.drop(columns=["threshold_hours", "status"]),
        alternate.drop(columns=["threshold_hours", "status"]),
    )
    changes = standard.loc[standard["status"].ne(alternate["status"]), "credit_time"]
    checks = []
    # February–September training targets, then October validation. The final
    # November evaluation stays sealed. Only counts and equality results are exported.
    for start in pd.date_range("2011-02-01", "2011-10-01", freq="MS"):
        end = start + pd.offsets.MonthBegin(1)
        _, history = purchase_snapshot(ledger, standard, cutoff=start)
        _, other_history = purchase_snapshot(ledger, alternate, cutoff=start)
        pd.testing.assert_frame_equal(history, other_history, check_exact=True)
        targets = net_retained_labels(ledger, standard, target_start=start, target_end=end)
        other_targets = net_retained_labels(ledger, alternate, target_start=start, target_end=end)
        pd.testing.assert_frame_equal(targets, other_targets, check_exact=True)
        checks.append({
            "target_month": start.strftime("%Y-%m"),
            "split": "validation" if start.month == 10 else "training",
            "standard_uk_hours": 72, "alternate_uk_hours": 120,
            "reclassified_in_full_workbook": len(changes),
            "reclassified_before_history_cutoff": int(changes.lt(start).sum()),
            "reclassified_before_target_end": int(changes.lt(end).sum()),
            "history_basket_item_events": len(history),
            "distinct_net_positive_customer_skus_in_month": len(targets),
            "identical_sale_allocations": True,
            "identical_history_keys_quantities_amounts_times": True,
            "identical_month_end_labels_before_catalog_filter": True,
        })
    return pd.DataFrame(checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    root = parser.parse_args().project_dir
    manifest = json.loads((root / "results/data_manifest.json").read_text(encoding="utf-8"))
    path = root / LEDGER_PATH
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["outputs"][LEDGER_PATH]["sha256"]:
            raise ValueError("Signed ledger differs from its D1 cleaning manifest.")
    ledger = pd.read_csv(
        path, dtype={column: "string" for column in ("CustomerID", "InvoiceNo", "StockCode")},
        keep_default_na=False, na_values=[""], parse_dates=["InvoiceDate"],
    )
    checks = compare_thresholds(ledger)
    output = root / "results/net_retention_threshold_check.csv"
    with NamedTemporaryFile(dir=output.parent, suffix=".csv", delete=False) as temp:
        staged = Path(temp.name)
    try:
        checks.to_csv(staged, index=False)
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
