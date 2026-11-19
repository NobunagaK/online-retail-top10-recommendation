"""Popularity, repeat purchases and interaction sparsity in a training event window."""

import pandas as pd


def _share(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def training_preference(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Summarize basket-item events already restricted to the training period.

    Each event is a distinct (CustomerID, InvoiceNo, StockCode) with its completed
    basket_time. Identifiers are exact strings. Counts weight each event once,
    regardless of purchased units. Return one ranked row per SKU, and totals
    and ratios for the entire supplied window. Empty-window ratios are undefined.
    """
    item_key = ["CustomerID", "InvoiceNo", "StockCode"]
    if events[[*item_key, "basket_time"]].isna().any().any():
        raise ValueError("Each basket-item event needs a customer, invoice, SKU and completion time.")
    if events.duplicated(item_key).any():
        raise ValueError("Expected one event per customer, invoice and SKU.")

    customers = int(events["CustomerID"].nunique())
    popularity = (
        events.groupby("StockCode", as_index=False, sort=False, observed=True)
        .agg(purchasing_customers=("CustomerID", "nunique"), basket_item_events=("InvoiceNo", "size"))
        .sort_values(["purchasing_customers", "StockCode"], ascending=[False, True], ignore_index=True)
    )
    items = len(popularity)
    # Integer ceiling gives exactly the leading 10% (rounded up), even with ties.
    head_items = (items + 9) // 10
    popularity.insert(0, "rank", range(1, items + 1))
    popularity.insert(3, "customer_reach_share", popularity["purchasing_customers"] / customers)
    popularity["is_head"] = popularity["rank"].le(head_items)
    head = popularity.loc[popularity["is_head"]]
    head_events = int(head["basket_item_events"].sum())
    head_customers = int(events.loc[events["StockCode"].isin(head["StockCode"]), "CustomerID"].nunique())

    pair_key = ["CustomerID", "StockCode"]
    pairs = events.groupby(pair_key, sort=False, observed=True)
    observed_pairs = pairs.ngroups
    possible_pairs = customers * items
    # Two invoices at the same first completion time do not provide earlier
    # history for one another. The unique event key makes any later match a
    # different invoice for this customer and SKU.
    repeat = events["basket_time"].gt(pairs["basket_time"].transform("min"))
    repeat_events = int(repeat.sum())
    repeat_pairs = len(events.loc[repeat, pair_key].drop_duplicates())
    event_count = len(events)

    summary = {
        "customers": customers,
        "items": items,
        "basket_item_events": event_count,
        "observed_customer_item_pairs": observed_pairs,
        "possible_customer_item_pairs": possible_pairs,
        "interaction_sparsity": 1 - observed_pairs / possible_pairs if possible_pairs else None,
        "repeat_purchase_events": repeat_events,
        "repeat_event_share": _share(repeat_events, event_count),
        "repeated_customer_item_pairs": repeat_pairs,
        "repeated_pair_share": _share(repeat_pairs, observed_pairs),
        "head_items": head_items,
        "head_purchase_events": head_events,
        "head_event_share": _share(head_events, event_count),
        "head_purchasing_customers": head_customers,
        "head_customer_reach_share": _share(head_customers, customers),
    }
    return popularity, summary
