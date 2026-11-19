import pandas as pd
import pytest

from src.data.baskets import build_basket_events
from src.data.training_eda import distribution_tables, monthly_activity, select_training_data
from src.data.training_preference import training_preference


def purchase_data(rows):
    lines = pd.DataFrame(rows, columns=[
        "CustomerID", "InvoiceNo", "StockCode", "InvoiceDate", "Quantity", "UnitPrice",
    ])
    lines["InvoiceDate"] = pd.to_datetime(lines["InvoiceDate"])
    lines["ExcelRow"] = range(2, 2 + len(lines))
    lines["line_amount_gbp"] = lines["Quantity"] * lines["UnitPrice"]
    return build_basket_events(lines), lines


def test_training_window_uses_completed_customer_baskets_and_preserves_source_lines():
    start = pd.Timestamp("2011-01-01")
    end = pd.Timestamp("2011-03-01")
    # Customer B reused invoice number I1. Its first line predates the cutoff,
    # but the basket closes exactly at the excluded endpoint.
    events, lines = purchase_data([
        ("A", "old", "K", "2010-12-29", 1, 1),
        ("A", "I1", "X", "2010-12-31", 2, 1),
        ("A", "I1", "X", "2011-01-01", 3, 4),
        ("A", "I1", "Y", "2011-01-01", 1, 5),
        ("B", "I1", "X", "2011-02-28", 70, 6),
        ("B", "I1", "Z", "2011-03-01", 10, 12),
    ])
    selected_events, selected_lines = select_training_data(events, lines, start, end)

    assert set(selected_lines["ExcelRow"]) == {3, 4, 5}
    assert set(selected_events["CustomerID"]) == {"A"}
    assert set(selected_events["StockCode"]) == {"X", "Y"}
    assert set(selected_events["basket_time"]) == {start}

    future_events, future_lines = purchase_data([
        ("C", "future", "NEW", "2011-05-05", 1, 1),
    ])
    future_lines["ExcelRow"] = 99
    rerun = select_training_data(
        pd.concat([events, future_events], ignore_index=True),
        pd.concat([lines, future_lines], ignore_index=True), start, end,
    )
    for before, after in zip((selected_events, selected_lines), rerun):
        pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def activity_fixture():
    # A single invoice identifier is reused by customers A and B. Customer A
    # buys X twice at different unit prices within that invoice.
    events, lines = purchase_data([
        ("A", "I1", "X", "2011-01-02", 2, 1),
        ("A", "I1", "X", "2011-01-02", 3, 4),
        ("A", "I1", "Y", "2011-01-02", 1, 5),
        ("B", "I1", "X", "2011-01-04", 4, 3),
        ("A", "I2", "X", "2011-03-01", 2, 2),
    ])
    return events, lines


def test_monthly_counts_include_empty_month_and_keep_basket_and_event_units_distinct():
    events, _ = activity_fixture()
    monthly = monthly_activity(events, pd.Timestamp("2011-01-01"), pd.Timestamp("2011-04-01"))
    fields = ["month", "active_customers", "baskets", "basket_item_events",
              "source_lines", "units", "amount_gbp"]
    assert monthly[fields].to_records(index=False).tolist() == [
        ("2011-01", 2, 2, 3, 4, 10, 31),
        ("2011-02", 0, 0, 0, 0, 0, 0),
        ("2011-03", 1, 1, 1, 1, 2, 4),
    ]


def test_price_and_quantity_use_source_lines_while_baskets_and_histories_use_events():
    events, lines = activity_fixture()
    freq, summary = distribution_tables(events, lines)
    expected = {
        "line_unit_price_gbp": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        "line_quantity": {1: 1, 2: 2, 3: 1, 4: 1},
        "basket_distinct_skus": {1: 2, 2: 1},
        "customer_history_events": {1: 1, 3: 1},
        "customer_baskets": {1: 1, 2: 1},
    }
    summaries = summary.set_index("measure")
    for measure, counts in expected.items():
        observed = freq.loc[freq["measure"] == measure].set_index("value")["frequency"]
        assert observed.to_dict() == counts
        assert summaries.loc[measure, "count"] == sum(counts.values())
    assert summaries.loc["line_unit_price_gbp", "mean"] == 3
    assert summaries.loc["line_quantity", "mean"] == pytest.approx(2.4)
    assert summaries.loc["customer_history_events", "p25"] == 1.5


def test_popularity_cutoff_and_repeat_statistics_use_distinct_customers_and_earlier_time():
    # Three SKUs reach two customers. The eleven-item catalog has two head
    # positions. Case-sensitive code order breaks the three-way tie; 'a' has
    # more events than 'B' but still ranks behind it.
    rows = [
        (customer, invoice, sku, date, 1, 1)
        for customer, invoice, skus, date in [
            ("X", "first", ["A", "B", "a", "c", "d", "e", "f", "g", "h", "i", "j"], "2011-01-01"),
            ("X", "tied", ["A", "a"], "2011-01-01"),
            ("X", "later", ["A"], "2011-01-02"),
            ("Y", "other", ["A", "B", "a"], "2011-01-01"),
        ]
        for sku in skus
    ]
    events, _ = purchase_data(rows)
    pop, summary = training_preference(events.iloc[::-1])
    assert pop["StockCode"].tolist()[:3] == ["A", "B", "a"]
    head = pop.loc[pop["is_head"]]
    assert head["StockCode"].tolist() == ["A", "B"]
    assert head["purchasing_customers"].tolist() == [2, 2]
    assert head["basket_item_events"].tolist() == [4, 2]
    assert summary == pytest.approx({
        "customers": 2,
        "items": 11,
        "basket_item_events": 17,
        "observed_customer_item_pairs": 14,
        "possible_customer_item_pairs": 22,
        "interaction_sparsity": 1 - 14 / 22,
        "repeat_purchase_events": 1,
        "repeat_event_share": 1 / 17,
        "repeated_customer_item_pairs": 1,
        "repeated_pair_share": 1 / 14,
        "head_items": 2,
        "head_purchase_events": 6,
        "head_event_share": 6 / 17,
        "head_purchasing_customers": 2,
        "head_customer_reach_share": 1,
    })
