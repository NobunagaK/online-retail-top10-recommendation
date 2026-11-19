"""Historical recency and purchase frequency for monthly recommendation queries."""

import pandas as pd

from src.data.sequences import _prior_events


def history_features(events, history_end, *, customer_ids=None):
    """Describe retained events strictly before history_end at three granularities.

    Replay D1 at history_end before calling this function. Recency is elapsed
    fractional days from the last observable basket completion to history_end.
    A 7/30/90-day count includes completions exactly at its lower bound and
    excludes history_end. The target month never enters these calculations.
    Customer–item rows are sparse: an unobserved pair has zero purchases and
    undefined recency. Optional customer_ids add explicit empty customer rows.
    """
    end = pd.Timestamp(history_end)
    prior = _prior_events(events, end).sort_values(["CustomerID", "basket_time", "InvoiceNo", "StockCode"])
    # One customer invoice is one basket, even if it contains several items.
    prior["basket_id"] = prior.groupby(["CustomerID", "InvoiceNo"], sort=False).ngroup()
    windows = (7, 30, 90)
    for days in windows:
        recent = prior["basket_time"].ge(end - pd.Timedelta(days=days))
        prior[f"basket_{days}d"] = prior["basket_id"].where(recent)

    groups = {
        "customer_features": ["CustomerID"],
        "item_features": ["StockCode"],
        "customer_item_features": ["CustomerID", "StockCode"],
    }
    tables = {}
    for name, keys in groups.items():
        measures = {
            "event_count": ("StockCode", "size"),
            "invoice_count": ("basket_id", "nunique"),
            "quantity": ("quantity", "sum"),
            "amount_gbp": ("amount_gbp", "sum"),
            "last_purchase_time": ("basket_time", "max"),
        }
        for days in windows:
            measures[f"invoices_{days}d"] = (f"basket_{days}d", "nunique")
        if name == "item_features":
            measures["distinct_customers"] = ("CustomerID", "nunique")
        if name == "customer_features":
            measures["distinct_items"] = ("StockCode", "nunique")
        result = prior.groupby(keys, as_index=False).agg(**measures)
        if name == "customer_features":
            if customer_ids is not None:
                wanted = result["CustomerID"].astype("string")
                all_customers = pd.Index(wanted, name="CustomerID").union(pd.Index(customer_ids))
                totals = [field for field in measures if field != "last_purchase_time"]
                original_types = result[totals].dtypes
                result = result.set_index("CustomerID").reindex(all_customers.rename("CustomerID"))
                result[totals] = result[totals].fillna(0).astype(original_types)
                result = result.reset_index()
            result["mean_basket_size"] = result["event_count"].div(result["invoice_count"].where(result["invoice_count"].gt(0)))
        result["days_since_last_purchase"] = (end - result["last_purchase_time"]) / pd.Timedelta(days=1)
        result["history_end"] = end
        tables[name] = result.rename(columns={"CustomerID": "customer_id", "StockCode": "stock_code"})
    return tables
