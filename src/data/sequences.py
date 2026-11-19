"""Cutoff-specific, chronological item histories shared by LRURec and Mamba4Rec."""

from itertools import accumulate
from numbers import Integral

import pandas as pd


def _prior_events(events, history_end):
    """Take a D1 replay for this cutoff. Filtering dates alone cannot undo later credits."""
    times = pd.to_datetime(events["basket_time"], format="mixed")
    before = times.lt(pd.Timestamp(history_end))
    frame = events.loc[before].copy()
    frame["basket_time"] = times.loc[before]
    keys = ["CustomerID", "InvoiceNo", "StockCode"]
    for key in keys:
        frame[key] = frame[key].astype("string")
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("History requires one identified event per customer, invoice and SKU.")
    return frame


def build_item_vocab(events_by_history_end, *, training_end="2011-10-01"):
    """Assign append-only positive IDs to SKUs visible in permitted historical replays.

    At each cutoff, order newly observed SKUs by first retained basket time and
    exact stock code. An item can later disappear after a full credit. Its ID
    remains reserved, and its candidate eligibility is decided separately for
    each snapshot. Observation keys after training_end cannot extend the vocab.
    """
    columns = ["stock_code", "item_id", "first_event_time", "first_history_end"]
    seen, records = set(), []
    for cutoff, events in sorted(events_by_history_end.items(), key=lambda pair: pd.Timestamp(pair[0])):
        end = pd.Timestamp(cutoff)
        if end > pd.Timestamp(training_end):
            continue
        prior = _prior_events(events, end)
        new = prior.loc[~prior["StockCode"].isin(seen)]
        first = new.groupby("StockCode", as_index=False)["basket_time"].min()
        first = first.sort_values(["basket_time", "StockCode"])
        for sku, time in first.itertuples(index=False, name=None):
            records.append((sku, len(records) + 1, time, end))
            seen.add(sku)
    vocab = pd.DataFrame.from_records(records, columns=columns)
    return vocab.astype({
        "stock_code": "string", "item_id": "int64",
        "first_event_time": "datetime64[ns]", "first_history_end": "datetime64[ns]",
    })


def _check_max_length(max_length):
    if not isinstance(max_length, Integral) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be a positive integer.")


def build_histories(events, history_end, item_vocab, max_length=64, *, customer_ids=None):
    """Return complete histories and truncation diagnostics for each customer at a cutoff.

    Additional customer_ids can represent eligible queries with no surviving history.
    Within a customer, ties on basket time use the exact text invoice identifier,
    then stock code within a basket. Repeats across invoices remain distinct events.
    """
    _check_max_length(max_length)
    end = pd.Timestamp(history_end)
    ordered = _prior_events(events, end).sort_values(["CustomerID", "basket_time", "InvoiceNo", "StockCode"])
    visible = item_vocab["first_history_end"].le(end)
    lookup = item_vocab.loc[visible].set_index("stock_code")["item_id"]
    if not lookup.index.is_unique or not lookup.is_unique or not lookup.gt(0).all():
        raise ValueError("Each stock code must map to a distinct positive item ID.")
    ordered["item_id"] = ordered["StockCode"].map(lookup)
    if ordered["item_id"].isna().any():
        raise ValueError("An eligible history SKU is missing from the item vocabulary.")
    ordered["item_id"] = ordered["item_id"].astype("int64")
    items = ordered.groupby("CustomerID", sort=False)["item_id"].agg(list)
    basket_sizes = ordered.groupby(["CustomerID", "InvoiceNo"], sort=False).size()
    sizes = basket_sizes.groupby(level="CustomerID", sort=False).agg(list)
    customers = set(items.index)
    if customer_ids is not None:
        customers.update(customer_ids)
    records = []
    for customer in sorted(customers):
        ids = items.get(customer, [])
        baskets = sizes.get(customer, [])
        length = len(ids)
        omitted = max(length - max_length, 0)
        cuts_basket = omitted > 0 and omitted not in accumulate(baskets)
        records.append((customer, end, ids, baskets, length, min(length, max_length), omitted > 0, cuts_basket))
    return pd.DataFrame.from_records(records, columns=[
        "customer_id", "history_end", "history_item_ids", "history_basket_sizes",
        "history_len", "seq_len", "is_truncated", "cuts_basket",
    ]).astype({
        "customer_id": "string", "history_end": "datetime64[ns]", "history_len": "int64",
        "seq_len": "int64", "is_truncated": bool, "cuts_basket": bool,
    })


def prepare_sequence(item_ids, max_length=64, padding="left"):
    """Return padded IDs, real length, and index of the last valid state (None if empty).

    LRURec uses left padding. Mamba4Rec uses right padding. Both keep the same
    most recent events; PAD=0 is reserved and input histories must be unpadded.
    """
    _check_max_length(max_length)
    if padding not in {"left", "right"}:
        raise ValueError("padding must be 'left' or 'right'.")
    if any(not isinstance(item, Integral) or isinstance(item, bool) or item < 1 for item in item_ids):
        raise ValueError("An unpadded history contains a nonpositive or non-integer item ID.")
    kept = list(item_ids[-max_length:])
    seq_len = len(kept)
    pads = [0] * (max_length - seq_len)
    if padding == "left":
        return pads + kept, seq_len, max_length - 1 if seq_len else None
    return kept + pads, seq_len, seq_len - 1 if seq_len else None
