"""Descriptive statistics for paid merchandise baskets completed in the training period."""

import pandas as pd

from src.data.baskets import INVOICE_KEYS


TRAIN_START = pd.Timestamp("2010-12-01")
TRAIN_END = pd.Timestamp("2011-10-01")


def select_training_data(
    events: pd.DataFrame, purchase_lines: pd.DataFrame, start=TRAIN_START, end=TRAIN_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select whole baskets in [start, end) and their retained purchase lines.

    The event table supplies one completion time per (CustomerID, InvoiceNo).
    A basket completed at the cutoff is excluded even if some lines are earlier.
    Read CSV identifiers as strings to preserve their original spelling and zeros.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start >= end:
        raise ValueError("The training start must be earlier than the end.")
    times = pd.to_datetime(events["basket_time"])
    selected = times.ge(start) & times.lt(end)
    train_events = events.loc[selected].copy()
    train_events["basket_time"] = times.loc[selected]
    train_events = train_events.sort_values(
        ["basket_time", *INVOICE_KEYS, "StockCode"], ignore_index=True,
    )
    baskets = train_events[[*INVOICE_KEYS, "basket_time"]].drop_duplicates()
    train_lines = purchase_lines.merge(baskets, on=INVOICE_KEYS, how="inner", validate="many_to_one")
    train_lines["InvoiceDate"] = pd.to_datetime(train_lines["InvoiceDate"])
    if train_lines["InvoiceDate"].gt(train_lines["basket_time"]).any():
        raise ValueError("A retained invoice line is dated after its basket's completion.")
    if len(train_lines) != train_events["source_lines"].sum():
        raise ValueError("Training basket source-line counts differ from the matched purchase lines.")
    return train_events, train_lines.sort_values("ExcelRow", ignore_index=True)


def monthly_activity(train_events: pd.DataFrame, start=TRAIN_START, end=TRAIN_END) -> pd.DataFrame:
    """Summarize each calendar month, including empty months within [start, end).

    Each basket and all its item events belong to its completion month. Customer
    counts are monthly distinct customers. Amounts preserve sub-penny GBP.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start >= end:
        raise ValueError("The training start must be earlier than the end.")
    times = pd.to_datetime(train_events["basket_time"])
    selected = times.ge(start) & times.lt(end)
    events = train_events.loc[selected].assign(month=times.loc[selected].dt.to_period("M"))
    monthly = events.groupby("month").agg(
        active_customers=("CustomerID", "nunique"),
        basket_item_events=("StockCode", "size"),
        source_lines=("source_lines", "sum"),
        units=("quantity", "sum"),
        amount_gbp=("amount_gbp", "sum"),
    )
    baskets = events[["month", *INVOICE_KEYS]].drop_duplicates()
    monthly.insert(1, "baskets", baskets.groupby("month").size())
    months = pd.period_range(start, end - pd.Timedelta(nanoseconds=1), freq="M", name="month")
    monthly = monthly.reindex(months, fill_value=0)
    monthly.index = monthly.index.astype(str)
    return monthly.reset_index()


def distribution_tables(
    train_events: pd.DataFrame, train_lines: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return exact-value frequencies and summaries of five training distributions.

    Price and quantity are observed once per source line, with no quantity
    weighting. Basket size counts distinct SKUs per completed customer invoice.
    History length counts basket-item events per customer across the full training
    window, including repeat purchases on later invoices. Basket frequency counts
    distinct completed invoices per customer in that same window. Quantiles use
    linear interpolation on the individual observations.
    """
    baskets = train_events.groupby(INVOICE_KEYS)["StockCode"].nunique()
    distributions = {
        "line_unit_price_gbp": (train_lines["UnitPrice"], "source line", "GBP per unit"),
        "line_quantity": (train_lines["Quantity"], "source line", "units"),
        "basket_distinct_skus": (baskets, "basket", "distinct SKUs"),
        "customer_history_events": (
            train_events.groupby("CustomerID").size(), "customer", "basket-item events",
        ),
        "customer_baskets": (
            baskets.groupby(level="CustomerID").size(), "customer", "baskets",
        ),
    }
    frequencies, summaries = [], []
    quantiles = {"p25": .25, "median": .5, "p75": .75, "p90": .90, "p95": .95, "p99": .99}
    for measure, (observations, observation_unit, value_unit) in distributions.items():
        counts = observations.value_counts().sort_index().rename_axis("value")
        frequencies.append(counts.rename("frequency").reset_index().assign(measure=measure))
        percentiles = observations.quantile(list(quantiles.values()))
        summaries.append({
            "measure": measure,
            "observation_unit": observation_unit,
            "value_unit": value_unit,
            "count": len(observations),
            "min": observations.min(),
            **{name: percentiles.loc[q] for name, q in quantiles.items()},
            "max": observations.max(),
            "mean": observations.mean(),
        })
    frequency_table = pd.concat(frequencies, ignore_index=True)[["measure", "value", "frequency"]]
    return frequency_table, pd.DataFrame(summaries)
