"""Build invoice-item events from retained Online Retail purchase lines."""

import pandas as pd


INVOICE_KEYS = ["CustomerID", "InvoiceNo"]
ITEM_KEYS = [*INVOICE_KEYS, "StockCode"]


def build_basket_events(purchases: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each item within a customer invoice at the basket's completion time."""
    # Use the original row order for reproducible sums, including fractional pennies.
    ordered = purchases.sort_values("ExcelRow")
    # A reversed or reduced line may have carried the invoice's last timestamp.
    # Snapshot lines preserve the completion time of the original eligible basket.
    time_column = "basket_time" if "basket_time" in ordered.columns else "InvoiceDate"
    events = ordered.groupby(ITEM_KEYS, as_index=False, sort=False).agg(
        basket_time=(time_column, "max"),
        quantity=("Quantity", "sum"),
        amount_gbp=("line_amount_gbp", "sum"),
        source_lines=("ExcelRow", "size"),
    )
    events["basket_time"] = events.groupby(INVOICE_KEYS, sort=False)["basket_time"].transform("max")
    columns = ["CustomerID", "InvoiceNo", "basket_time", "StockCode", "quantity", "amount_gbp", "source_lines"]
    return events.sort_values(["CustomerID", "basket_time", "InvoiceNo", "StockCode"], ignore_index=True)[columns]


def training_basket_counts(purchases: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, int]:
    """Count multi-time invoices and multi-price items in baskets completed before cutoff."""
    if "basket_time" in purchases.columns:
        completion = purchases["basket_time"]
    else:
        completion = purchases.groupby(INVOICE_KEYS, sort=False)["InvoiceDate"].transform("max")
    history = purchases.loc[completion.lt(cutoff)]
    invoice_times = history.groupby(INVOICE_KEYS, sort=False)["InvoiceDate"].nunique()
    item_prices = history.groupby(ITEM_KEYS, sort=False)["UnitPrice"].nunique()
    return {
        "multi_time_invoices": int(invoice_times.gt(1).sum()),
        "multi_price_invoice_items": int(item_prices.gt(1).sum()),
    }
