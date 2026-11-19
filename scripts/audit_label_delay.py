"""Audit February–October 2011 month-end positives against the last observed D1 balances.

Each denominator counts distinct (target month, customer, SKU) positives before
catalog or customer eligibility. Observation ends with the workbook, so additional
credits outside it cannot be assessed. The output contains aggregate diagnostics
only and is separate from query construction, model selection and evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data.baskets import INVOICE_KEYS
from src.data.purchase_snapshot import net_retained_labels


START = pd.Timestamp("2011-02-01")
TEST_START = pd.Timestamp("2011-11-01")
KEYS = ["CustomerID", "StockCode"]
LEDGER = "data/processed/signed_ledger_b.csv.gz"
CREDITS = "results/credit_allocations.csv.gz"
FINAL_LINES = "data/processed/purchase_lines.csv.gz"
OUTPUT = "results/d2_audit/label_delay_pretest.csv"


def compare_month_ends(ledger, credits, final_lines, observed_through):
    """Compare frozen month-end keys with the same baskets at workbook end."""
    if observed_through < TEST_START:
        raise ValueError("Observation must include the end of the October target.")
    sales = ledger.loc[ledger["ledger_type"].eq("sale")].copy()
    completion = sales.groupby(INVOICE_KEYS)["InvoiceDate"].transform("max")
    pretest = completion.ge(START) & completion.lt(TEST_START)
    sales = sales.loc[pretest].assign(target_month=completion.loc[pretest].dt.to_period("M"))
    matched = credits.loc[credits["purchase_excel_row"].isin(sales["ExcelRow"])]
    if matched["credit_time"].gt(observed_through).any():
        raise ValueError("Matched credit timestamp exceeds the declared observation endpoint.")
    last_credit = matched.groupby("purchase_excel_row")["credit_time"].max()
    final = final_lines.loc[
        final_lines["basket_time"].ge(START) & final_lines["basket_time"].lt(TEST_START)
    ]
    counts = []
    for start in pd.date_range(START, TEST_START, freq="MS", inclusive="left"):
        end = start + pd.offsets.MonthBegin(1)
        month_sales = sales.loc[sales["target_month"].eq(start.to_period("M"))]
        frozen = net_retained_labels(month_sales, matched, target_start=start, target_end=end)
        in_month = final["basket_time"].ge(start) & final["basket_time"].lt(end)
        at_end = pd.MultiIndex.from_frame(frozen[KEYS])
        at_horizon = pd.MultiIndex.from_frame(final.loc[in_month, KEYS].drop_duplicates())
        if not at_horizon.difference(at_end).empty:
            raise ValueError("Following later credits added a positive label.")
        lost = at_end.difference(at_horizon)
        # All units on all target-month sales of a lost pair must be credited.
        # Its last such credit is the point at which the pair reaches zero.
        depletion = month_sales.assign(last_credit=month_sales["ExcelRow"].map(last_credit))
        dates = depletion.groupby(KEYS)["last_credit"].max().reindex(lost)
        if dates.isna().any() or dates.lt(end).any():
            raise ValueError("Lost labels must have a credit at or after month-end.")
        counts.append({
            "target_month": start.strftime("%Y-%m"),
            "split": "validation" if end == TEST_START else "training",
            "target_end_exclusive": end.isoformat(),
            "workbook_observed_through_inclusive": observed_through.isoformat(),
            "followup_days": round((observed_through - end) / pd.Timedelta(days=1), 4),
            "month_end_positive_customer_skus": len(at_end),
            "still_positive_at_observed_end": len(at_horizon),
            "positive_to_zero": len(lost),
            "share_of_month_end_positives": len(lost) / len(at_end) if len(at_end) else None,
            "depleted_during_following_month": int(dates.lt(end + pd.offsets.MonthBegin(1)).sum()),
            "depleted_after_following_month": int(dates.ge(end + pd.offsets.MonthBegin(1)).sum()),
        })
    return pd.DataFrame(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    root = parser.parse_args().project_dir
    manifest = json.loads((root / "results/data_manifest.json").read_text(encoding="utf-8"))
    for name in (LEDGER, CREDITS, FINAL_LINES):
        with (root / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["outputs"][name]["sha256"]:
                raise ValueError(f"Input differs from the D1 manifest: {name}")
    ids = {name: "string" for name in [*KEYS, "InvoiceNo"]}
    ledger = pd.read_csv(root / LEDGER, dtype=ids, parse_dates=["InvoiceDate"],
                         keep_default_na=False, na_values=[""])
    credits = pd.read_csv(root / CREDITS, usecols=["purchase_excel_row", "credit_time", "status", "credited_units"],
                          parse_dates=["credit_time"])
    # Select pre-test baskets before constructing any customer/SKU keys.
    final = pd.read_csv(root / FINAL_LINES, usecols=[*KEYS, "basket_time"],
                        dtype={name: "string" for name in KEYS}, parse_dates=["basket_time"],
                        keep_default_na=False, na_values=[""])
    endpoint = pd.Timestamp(manifest["source"]["last_timestamp"])
    report = compare_month_ends(ledger, credits, final, endpoint)
    path = root / OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as stage:
        staged = Path(stage) / path.name
        report.to_csv(staged, index=False)
        staged.replace(path)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
