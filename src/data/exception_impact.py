"""Descriptive sensitivity of invoice counts, recency and amounts to flagged lines."""

import pandas as pd


def impact_tables(raw: pd.DataFrame, exception_mask: pd.Series) -> dict[str, pd.DataFrame]:
    """Compare all worksheet rows with non-C, positive-quantity, positive-price rows.

    Supply the union of the three exception flags on the raw row index. Keep
    duplicate rows so that amounts reconcile to the input. An invoice can occur
    in both line categories, so distinct-invoice counts need not be additive.
    Recency uses calendar days relative to the day after the last worksheet
    date. An exception-only customer has no regular-purchase recency.

    Amounts are arithmetic products of the original quantity and unit price in
    GBP. They describe the worksheet and do not establish payment or a refund.
    """
    if not raw.index.equals(exception_mask.index) or exception_mask.isna().any():
        raise ValueError("The exception mask must identify every raw row on the same index.")

    lines = raw[["CustomerID", "InvoiceNo", "InvoiceDate"]].copy()
    lines["signed_gbp"] = raw["Quantity"] * raw["UnitPrice"]
    lines["positive_line_gbp"] = lines["signed_gbp"].clip(lower=0)
    lines["negative_line_gbp"] = lines["signed_gbp"].clip(upper=0)
    lines["record_type"] = pd.Categorical.from_codes(
        exception_mask.astype(int), categories=["regular", "exception"],
    )
    lines["customer_id_status"] = pd.Categorical.from_codes(
        lines["CustomerID"].isna().astype(int), categories=["identified", "missing"],
    )
    by_type = lines.groupby(["record_type", "customer_id_status"], observed=False).agg(
        rows=("InvoiceNo", "size"),
        distinct_invoices=("InvoiceNo", "nunique"),
        positive_line_gbp=("positive_line_gbp", "sum"),
        negative_line_gbp=("negative_line_gbp", "sum"),
        signed_gbp=("signed_gbp", "sum"),
    ).reset_index()

    # Only identified customers enter the paired comparison. The table above
    # accounts for the transaction counts and amounts without a customer ID.
    identified = lines.loc[lines["CustomerID"].notna()]
    all_activity = identified.groupby("CustomerID").agg(
        all_rows=("InvoiceNo", "size"),
        all_invoices=("InvoiceNo", "nunique"),
        last_activity=("InvoiceDate", "max"),
        all_signed_gbp=("signed_gbp", "sum"),
    )
    regular = identified.loc[identified["record_type"] == "regular"].groupby("CustomerID").agg(
        regular_rows=("InvoiceNo", "size"),
        regular_invoices=("InvoiceNo", "nunique"),
        last_regular=("InvoiceDate", "max"),
        regular_gross_gbp=("signed_gbp", "sum"),
    )
    customers = all_activity.join(regular)
    for column in ["regular_rows", "regular_invoices"]:
        customers[column] = customers[column].fillna(0).astype("int64")
    customers["regular_gross_gbp"] = customers["regular_gross_gbp"].fillna(0.0)
    customers["exception_rows"] = customers["all_rows"] - customers["regular_rows"]
    customers["amount_difference_gbp"] = customers["all_signed_gbp"] - customers["regular_gross_gbp"]

    reference = raw["InvoiceDate"].dt.normalize().max() + pd.Timedelta(days=1)
    customers["recency_reference_date"] = reference
    for last_date, recency in [
        ("last_regular", "regular_recency_days"), ("last_activity", "all_recency_days"),
    ]:
        customers[recency] = (reference - customers.pop(last_date).dt.normalize()).dt.days.astype("Int64")

    cohort = pd.Series("mixed", index=customers.index, dtype="string")
    cohort.loc[customers["exception_rows"] == 0] = "regular_only"
    cohort.loc[customers["regular_rows"] == 0] = "exception_only"
    customers["cohort"] = pd.Categorical(cohort, categories=["regular_only", "mixed", "exception_only"])
    summary = customers.groupby("cohort", observed=False).agg(
        customers=("all_rows", "size"),
        regular_gross_gbp=("regular_gross_gbp", "sum"),
        all_signed_gbp=("all_signed_gbp", "sum"),
        amount_difference_gbp=("amount_difference_gbp", "sum"),
        median_regular_recency_days=("regular_recency_days", "median"),
        median_all_recency_days=("all_recency_days", "median"),
    ).reset_index()

    return {
        "amount_by_record_type": by_type,
        "customer_impact_summary": summary,
        "customer_impact": customers.reset_index(),
    }
