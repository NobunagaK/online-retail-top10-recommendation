import pandas as pd
import pytest

from src.data.baskets import build_basket_events, training_basket_counts
from src.data.merchandise import merchandise_rules
from src.data.purchase_cleaning import clean_purchase_lines


def entry(excel_row, invoice="101", sku="85123A", quantity=1, price=2.5,
          customer="001", date="2011-07-01", description="Lamp"):
    return {
        "ExcelRow": excel_row, "InvoiceNo": invoice, "StockCode": sku,
        "Description": description, "Quantity": quantity, "InvoiceDate": pd.Timestamp(date),
        "UnitPrice": price, "CustomerID": customer, "Country": "United Kingdom",
    }


def test_overlapping_flags_and_four_zero_price_contexts_preserve_paid_line():
    raw = pd.DataFrame([
        entry(2), entry(3),
        entry(4, invoice="C201", quantity=-1, price=-3, customer=None),
        entry(5, invoice="C201", quantity=-1, price=-3, customer=None),
        entry(6, invoice="C202", quantity=-1, price=-3),
        entry(7, invoice="203", quantity=-1, price=-3),
        entry(8, invoice="204", quantity=0, price=-1),
        entry(9, invoice="205", price=-1),
        entry(10, sku="M", price=0, description="Manual"),
        entry(11, price=0),
        entry(12, sku="85123a", price=0),
        entry(13, invoice="206", price=0),
        entry(14, invoice="c207", description=None),
    ])
    audit, decisions, purchases = clean_purchase_lines(raw)

    assert decisions.set_index("ExcelRow")["first_rule"].to_dict() == {
        2: "retained", 3: "exact_duplicate", 4: "missing_customer_id",
        5: "exact_duplicate", 6: "cancelled_invoice", 7: "negative_quantity",
        8: "zero_quantity", 9: "negative_unit_price", 10: "zero_price_manual",
        11: "zero_price_same_item", 12: "zero_price_other_item",
        13: "zero_price_no_priced_companion", 14: "retained",
    }
    assert set(purchases["ExcelRow"]) == {2, 14}
    assert purchases.loc[purchases["ExcelRow"].eq(2), "line_amount_gbp"].item() == 2.5
    by_rule = audit.set_index("rule")
    assert by_rule.loc["exact_duplicate", ["raw_flagged", "rows_removed"]].tolist() == [2, 2]
    assert by_rule.loc["missing_customer_id", ["raw_flagged", "rows_removed"]].tolist() == [2, 1]
    assert by_rule.loc["cancelled_invoice", ["raw_flagged", "rows_removed"]].tolist() == [3, 1]
    assert by_rule.loc["negative_quantity", ["raw_flagged", "rows_removed"]].tolist() == [4, 1]
    assert by_rule.loc["negative_unit_price", ["raw_flagged", "rows_removed"]].tolist() == [6, 1]
    assert audit["rows_before"].tolist() == [len(raw), *audit["rows_after"].tolist()[:-1]]
    assert (audit["rows_before"] - audit["rows_removed"]).tolist() == audit["rows_after"].tolist()
    assert by_rule["rows_removed"].sum() == len(raw) - len(purchases)


def test_missing_required_fields_get_individual_reasons_and_missing_description_is_retained():
    raw = pd.DataFrame([
        entry(2, invoice=None, sku=None), entry(3, sku=None),
        entry(4, date=None), entry(5, quantity=None), entry(6, price=None),
        entry(7, description=None),
    ])
    _, decisions, purchases = clean_purchase_lines(raw)
    assert decisions.set_index("ExcelRow")["first_rule"].to_dict() == {
        2: "missing_invoice_no", 3: "missing_stock_code", 4: "missing_invoice_date",
        5: "missing_quantity", 6: "missing_unit_price", 7: "retained",
    }
    assert purchases["ExcelRow"].tolist() == [7]


def test_frozen_service_codes_match_exactly_and_future_text_does_not_expand_the_list():
    exclusions = merchandise_rules().query("action == 'exclude'")
    assert exclusions["training_rows"].gt(0).all()
    assert exclusions["training_description"].notna().all()
    assert pd.to_datetime(exclusions["training_example_date"]).lt(pd.Timestamp("2011-10-01")).all()
    earlier = pd.DataFrame([
        entry(2, sku="POST", description="POSTAGE"), entry(3, sku="post"),
        entry(4, sku="POST2", description=None), entry(5, sku="00123"),
    ])
    later = pd.DataFrame([
        entry(6, invoice="301", sku="POST2", description="POSTAGE", date="2011-11-01"),
        entry(7, invoice="302", sku="POST", description=None, date="2011-11-01"),
        entry(8, invoice="303", sku="23444", description=None, date="2011-11-01"),
        entry(9, invoice="303", sku="PADS", description=None, date="2011-11-01"),
    ])
    _, earlier_decisions, _ = clean_purchase_lines(earlier)
    _, decisions, purchases = clean_purchase_lines(pd.concat([earlier, later], ignore_index=True))

    reasons = decisions.set_index("ExcelRow")["first_rule"]
    expected_earlier = earlier_decisions.set_index("ExcelRow")["first_rule"].to_dict()
    assert reasons.loc[earlier["ExcelRow"]].to_dict() == expected_earlier
    assert reasons.to_dict() == {
        2: "non_merchandise_POST", 3: "retained", 4: "retained", 5: "retained",
        6: "retained", 7: "non_merchandise_POST", 8: "non_merchandise_23444", 9: "retained",
    }
    assert set(purchases["StockCode"]) == {"post", "POST2", "00123", "PADS"}


def test_zero_price_same_sku_requires_same_customer_even_on_shared_invoice():
    raw = pd.DataFrame([
        entry(2, customer="001"),
        entry(3, price=0, customer="002"),
        entry(4, price=0, customer=None),
        entry(5, price=0, customer="001"),
        entry(6, invoice="102", customer=None),
        entry(7, invoice="102", customer=None, price=0),
    ])
    audit, decisions, _ = clean_purchase_lines(raw)
    counts = audit.set_index("rule")
    assert counts.loc["zero_price_same_item", ["raw_flagged", "rows_removed"]].tolist() == [2, 1]
    assert counts.loc["zero_price_other_item", ["raw_flagged", "rows_removed"]].tolist() == [2, 1]
    assert decisions.set_index("ExcelRow").loc[3, "first_rule"] == "zero_price_other_item"


def test_partial_invoice_uses_surviving_rows_for_basket_time_quantity_and_amount():
    # The same customer and SKU appear on another invoice. Another customer also
    # has the same invoice number. Neither contributes to the first basket.
    raw = pd.DataFrame([
        entry(2, quantity=2, price=2.5, date="2011-09-30 22:00"),
        entry(3, quantity=2, price=2.5, date="2011-09-30 22:00"),
        entry(4, quantity=3, price=3.25, date="2011-09-30 23:30"),
        entry(5, sku="A42", date="2011-10-01 00:05"),
        entry(6, quantity=40, price=0, date="2011-10-02 03:00"),
        entry(7, invoice="102", date="2011-10-04 12:00"),
        entry(8, customer="002", date="2011-10-06 08:00"),
    ])
    _, _, purchases = clean_purchase_lines(raw)
    events = build_basket_events(purchases).set_index(["CustomerID", "InvoiceNo", "StockCode"])
    assert set(events.index) == {
        ("001", "101", "85123A"), ("001", "101", "A42"),
        ("001", "102", "85123A"), ("002", "101", "85123A"),
    }
    event = events.loc[("001", "101", "85123A")]
    assert event["source_lines"] == 2
    assert event["quantity"] == 5
    assert event["amount_gbp"] == pytest.approx(2 * 2.5 + 3 * 3.25)
    assert event["basket_time"] == pd.Timestamp("2011-10-01 00:05")
    assert events.loc[("001", "101", "A42"), "basket_time"] == event["basket_time"]
    assert events.loc[("002", "101", "85123A"), "basket_time"] == pd.Timestamp("2011-10-06 08:00")
    assert training_basket_counts(purchases, pd.Timestamp("2011-10-01")) == {
        "multi_time_invoices": 0, "multi_price_invoice_items": 0,
    }
    assert training_basket_counts(purchases, pd.Timestamp("2011-10-02")) == {
        "multi_time_invoices": 1, "multi_price_invoice_items": 1,
    }
