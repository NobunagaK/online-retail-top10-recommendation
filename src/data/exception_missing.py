"""Describe missing data and assign overlapping flags to the purchase audit's first rule."""

import pandas as pd

from src.data.clean import RAW_COLUMNS, purchase_filter_flags


def missingness_tables(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize the complete source sheet, retaining each row in the audit."""
    missing_customer = raw["CustomerID"].isna()
    missing_records = []
    for field in RAW_COLUMNS:
        missing = raw[field].isna()
        missing_records.append({
            "field": field,
            "raw_rows": len(raw),
            "missing_rows": int(missing.sum()),
            "missing_pct": 100 * missing.mean(),
            "invoices_with_missing": raw.loc[missing, "InvoiceNo"].nunique(),
            "missing_with_missing_customer_id": int((missing & missing_customer).sum()),
        })

    # Count known IDs using every line of the affected invoice, including lines
    # with a known customer. A missing invoice number cannot link two records.
    affected = raw.loc[missing_customer, "InvoiceNo"].dropna().unique()
    invoice_lines = raw.loc[raw["InvoiceNo"].isin(affected), ["InvoiceNo", "CustomerID", "ExcelRow"]]
    invoices = invoice_lines.assign(
        missing_customer=invoice_lines["CustomerID"].isna(),
        missing_excel_row=invoice_lines["ExcelRow"].where(invoice_lines["CustomerID"].isna()),
    ).groupby("InvoiceNo", sort=False).agg(
        first_excel_row=("ExcelRow", "min"),
        first_missing_excel_row=("missing_excel_row", "min"),
        total_rows=("ExcelRow", "size"),
        missing_customer_rows=("missing_customer", "sum"),
        known_customer_ids=("CustomerID", "nunique"),
    ).reset_index().sort_values("first_excel_row", ignore_index=True)
    invoices["first_missing_excel_row"] = invoices["first_missing_excel_row"].astype("int64")

    # Same precedence as audit_transactions. The provenance column ExcelRow is
    # excluded from the eight-field definition of an exact duplicate.
    quantity = raw["Quantity"]
    price = raw["UnitPrice"]
    flags = purchase_filter_flags(raw)
    first_rule = pd.Series("retained", index=raw.index)
    for name, condition in flags.items():
        first_rule.loc[first_rule.eq("retained") & condition] = name

    cohorts = {
        "all_rows": pd.Series(True, index=raw.index),
        "C_invoices": flags["cancelled_invoice"],
        "Q_nonpositive": flags["nonpositive_quantity"],
        "P_nonpositive": flags["nonpositive_unit_price"],
        "zero_quantity": quantity.eq(0),
        "negative_quantity": quantity.lt(0),
        "zero_price": price.eq(0),
        "negative_price": price.lt(0),
        "missing_customer_id": missing_customer,
        "missing_description": raw["Description"].isna(),
    }
    attribution = []
    for scope, mask in cohorts.items():
        counts = first_rule.loc[mask.fillna(False)].value_counts()
        attribution.append({
            "scope": scope, "raw_rows": int(counts.sum()),
            **{rule: int(counts.get(rule, 0)) for rule in [*flags, "retained"]},
        })

    return {
        "missingness": pd.DataFrame.from_records(missing_records),
        "customer_id_invoices": invoices,
        "filter_attribution": pd.DataFrame.from_records(attribution),
    }
