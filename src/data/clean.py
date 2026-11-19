"""Read the identified UCI workbook and reproduce the original five-rule audit."""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


WORKBOOK_SHA256 = "43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d"
RAW_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


def read_workbook(path: Path) -> pd.DataFrame:
    with Path(path).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != WORKBOOK_SHA256:
            raise ValueError("Workbook differs from the source identified in the research plan.")
        stream.seek(0)
        raw = pd.read_excel(stream, sheet_name="Online Retail", engine="openpyxl",
                            dtype={key: "string" for key in ("InvoiceNo", "StockCode", "CustomerID")},
                            keep_default_na=False, na_values=[""])
    if raw.columns.tolist() != RAW_COLUMNS:
        raise ValueError("Unexpected source schema.")
    raw.insert(0, "ExcelRow", range(2, len(raw) + 2))
    return raw


def purchase_filter_flags(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Evaluate the five proposed purchase-sample rules in their audit order."""
    return {
        "exact_duplicate": frame.duplicated(subset=RAW_COLUMNS, keep="first"),
        "missing_customer_id": frame["CustomerID"].isna(),
        "cancelled_invoice": frame["InvoiceNo"].astype("string").str.startswith("C", na=False),
        "nonpositive_quantity": frame["Quantity"].le(0).fillna(False),
        "nonpositive_unit_price": frame["UnitPrice"].le(0).fillna(False),
    }


def audit_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Count each flag on raw rows and assign each removed row to its first rule."""
    flags = purchase_filter_flags(frame)
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
    audit = audit_transactions(read_workbook(workbook))
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
