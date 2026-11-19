"""Look up workbook evidence for flagged transactions by their exact identifiers."""

import pandas as pd


def _strictly_earlier_match(history, queries, keys, columns):
    """Both inputs are sorted by date. Return one earlier match per query position."""
    right = history.dropna(subset=keys)[[*keys, "InvoiceDate", *columns]]
    left = queries.dropna(subset=keys)[[*keys, "InvoiceDate", "_position"]]
    matched = pd.merge_asof(
        left, right, on="InvoiceDate", by=keys, allow_exact_matches=False,
        direction="backward",
    )
    return matched.set_index("_position")[columns]


def _prior_counts(history, queries, keys):
    history = history[[*keys, "InvoiceDate"]].copy()
    history["_count"] = history.groupby(keys, sort=False).cumcount() + 1
    matched = _strictly_earlier_match(history, queries, keys, ["_count"])
    # Only eligible query positions are present. A missing match there means zero.
    return matched["_count"].fillna(0).astype("Int64")


def purchase_evidence(raw: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    """Return evidence on the flagged rows' original index, in their original order.

    Both inputs use exact text identifiers, a datetime InvoiceDate and a unique
    ExcelRow (the worksheet row number, starting at 2). All counts are line counts,
    including repeated rows. Exact invoice comparisons retain any leading C.

    * invoice_rows_full_sheet counts all rows with the exact InvoiceNo, including
      the flagged row and rows with the same or later timestamp.
    * earlier_positive_invoice_rows and earlier_positive_invoice_sku_rows count
      rows with Quantity > 0 by invoice and by (invoice, StockCode), respectively.
    * earlier_positive_sku_rows counts positive-quantity rows by StockCode.
    * earlier_positive_customer_sku_rows counts them by (CustomerID, StockCode).
      Each earlier count uses InvoiceDate strictly before the flagged timestamp.
      Known keys with no matches yield zero. Missing keys or query dates yield NA.
    * latest_plausible_customer_sku_excel_row and latest_plausible_customer_sku_date
      identify the most recent strictly earlier positive-quantity, positive-price
      line on a non-C invoice for the same customer and product. At equal eligible
      times the larger ExcelRow wins. Unavailable references yield NA/NaT.
    * earlier_positive_stripped_invoice_sku_rows compares a C-prefixed invoice
      after removing its first C with earlier positive-quantity rows' unchanged
      invoice numbers and SKU. It counts identifier coincidences only.
    * earlier_matching_customer_sku_price_opposite_quantity_rows counts earlier
      plausible purchase lines with the same customer, SKU, and exact unit price,
      whose positive quantity equals the negated quantity of the flagged row.
      Both cancellation diagnostics yield NA for a non-C row or missing keys.

    These signals describe lines observed in the workbook. They establish neither
    an original purchase invoice for a cancellation nor a physical return.
    """
    queries = flagged[["InvoiceNo", "StockCode", "CustomerID", "InvoiceDate",
                       "Quantity", "UnitPrice"]].reset_index(drop=True)
    counts = raw["InvoiceNo"].value_counts()
    result = pd.DataFrame({
        "invoice_rows_full_sheet": queries["InvoiceNo"].map(counts).astype("Int64"),
    })
    queries["_position"] = queries.index
    dated_queries = queries.dropna(subset=["InvoiceDate"]).sort_values("InvoiceDate")
    positive = raw.loc[
        raw["Quantity"].gt(0) & raw["InvoiceDate"].notna(),
        ["InvoiceNo", "StockCode", "CustomerID", "InvoiceDate", "Quantity", "UnitPrice", "ExcelRow"],
    ].sort_values(["InvoiceDate", "ExcelRow"])

    groups = {
        "earlier_positive_invoice_rows": ["InvoiceNo"],
        "earlier_positive_invoice_sku_rows": ["InvoiceNo", "StockCode"],
        "earlier_positive_sku_rows": ["StockCode"],
        "earlier_positive_customer_sku_rows": ["CustomerID", "StockCode"],
    }
    for name, keys in groups.items():
        result[name] = _prior_counts(positive, dated_queries, keys)

    plausible = positive.loc[
        positive["UnitPrice"].gt(0)
        & positive["InvoiceNo"].notna()
        & ~positive["InvoiceNo"].str.startswith("C", na=False)
    ].copy()
    plausible["_purchase_date"] = plausible["InvoiceDate"]
    matched = _strictly_earlier_match(
        plausible, dated_queries, ["CustomerID", "StockCode"], ["ExcelRow", "_purchase_date"],
    )
    result["latest_plausible_customer_sku_excel_row"] = matched["ExcelRow"].astype("Int64")
    result["latest_plausible_customer_sku_date"] = matched["_purchase_date"]
    cancellations = dated_queries.loc[
        dated_queries["InvoiceNo"].str.startswith("C", na=False)
    ].copy()
    cancellations["InvoiceNo"] = cancellations["InvoiceNo"].str.slice(1).replace("", pd.NA)
    result["earlier_positive_stripped_invoice_sku_rows"] = _prior_counts(
        positive, cancellations, ["InvoiceNo", "StockCode"],
    )
    cancellations["Quantity"] = -cancellations["Quantity"]
    result["earlier_matching_customer_sku_price_opposite_quantity_rows"] = _prior_counts(
        plausible, cancellations, ["CustomerID", "StockCode", "UnitPrice", "Quantity"],
    )
    result.index = flagged.index
    return result
