"""Audit the first five purchase-line filters for UCI Online Retail."""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


WORKBOOK_SHA256 = "43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d"
RAW_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


def audit_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Count each flag on raw rows and assign each removed row to its first rule."""
    flags = {
        "exact_duplicate": frame.duplicated(subset=RAW_COLUMNS, keep="first"),
        "missing_customer_id": frame["CustomerID"].isna(),
        "cancelled_invoice": frame["InvoiceNo"].astype("string").str.startswith("C", na=False),
        "nonpositive_quantity": frame["Quantity"].le(0).fillna(False),
        "nonpositive_unit_price": frame["UnitPrice"].le(0).fillna(False),
    }
    remaining = pd.Series(True, index=frame.index)
    records = []
    for step, (rule, flagged) in enumerate(flags.items(), start=1):
        before = int(remaining.sum())
        removed = int((remaining & flagged).sum())
        remaining &= ~flagged
        records.append({
            "step": step,
            "rule": rule,
            "raw_flagged": int(flagged.sum()),
            "rows_before": before,
            "rows_removed": removed,
            "rows_after": int(remaining.sum()),
        })
    return pd.DataFrame.from_records(records)


def run_audit(workbook: Path, output: Path) -> pd.DataFrame:
    """Verify the source workbook and write the audit as a CSV."""
    workbook, output = Path(workbook), Path(output)
    if output.suffix.lower() != ".csv":
        raise ValueError("Choose a .csv file for the audit output.")
    with workbook.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != WORKBOOK_SHA256:
            raise ValueError(f"Workbook SHA-256 mismatch: expected {WORKBOOK_SHA256}, got {digest}")
        source.seek(0)
        frame = pd.read_excel(
            source, sheet_name="Online Retail", engine="openpyxl",
            keep_default_na=False, na_values=[""],
        )

    audit = audit_transactions(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = NamedTemporaryFile(
        dir=output.parent, mode="w", newline="", encoding="utf-8", delete=False,
    )
    temp_path = Path(temp.name)
    try:
        with temp:
            audit.to_csv(temp, index=False)
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)
    return audit
