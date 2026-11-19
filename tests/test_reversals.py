import pandas as pd
import pytest

from src.data import reversals
from src.data.reversals import (
    EUROPE_COUNTRIES, build_signed_ledger, compute_reversal_tables, match_reversals,
)


def entry(row, invoice, quantity, date="2011-03-01", sku="00100", customer="u",
          price=2., country="United Kingdom"):
    return dict(ExcelRow=row, InvoiceNo=invoice, Quantity=quantity, StockCode=sku,
                CustomerID=customer, UnitPrice=price, Country=country,
                InvoiceDate=pd.Timestamp(date), Description="product")


def test_signed_views_first_source_copy_and_frozen_vs_shape_controls():
    sale = entry(2, "0003", 2)
    raw = pd.DataFrame([
        dict(sale, ExcelRow=3), sale, entry(4, "C4", -1, price=7.),
        entry(5, "5", 1, customer=None), entry(6, "C6", -1, customer=None),
        entry(7, "7", 2, sku="PADS", price=.001), entry(8, "8", 1, sku="23444"),
        entry(9, "9", 1, sku="23574"), entry(10, "A10", 1, price=100),
        entry(11, "11", -1, price=0, customer=None), entry(12, "12", 1, price=0),
        entry(13, "13", 0), entry(14, "14", 60000), entry(15, "C15", -3, sku="POST"),
    ])
    t = compute_reversal_tables(raw)
    a, b = t["signed_ledger_a"], t["signed_ledger_b"]
    assert a.ExcelRow.tolist() == [2, 4, 5, 6, 7, 14]
    assert b.ExcelRow.tolist() == [2, 4, 7, 14]
    assert a.loc[a.customer_display.eq("Guest/Unknown"), "CustomerID"].isna().all()
    assert a.signed_amount_gbp.sum() == pytest.approx(4 - 7 + .001 * 2 + 120000)
    assert t["source_differences"].ExcelRow.tolist() == [7, 8, 9]
    assert t["source_audit"].groupby("stock_policy").rows.sum().eq(len(raw)).all()
    assert t["credit_allocations"].set_index("credit_excel_row").loc[6, "status"] == "anonymous"
    # The £7 credit cancels one £2 unit. Signed net uses its own £7 price.
    gross = t["reconciliation"].query(
        "scenario == 'frozen_balance_inclusive' and metric == 'known_gross_after_cancellations'"
    )
    assert gross.signed_amount_gbp.item() == pytest.approx(120002.002)
    assert set(t["benchmark"].metric) >= {"view_a", "view_b", "return", "anonymous"}


def test_nearest_single_sale_with_enough_remaining_quantity_and_equal_time_order():
    # All timestamps coincide. Lowest ExcelRow wins among eligible earlier
    # rows. Later worksheet rows at the same timestamp are permitted if needed.
    raw = pd.DataFrame([
        entry(2, "A-ignored", 20), entry(3, "003", 5), entry(4, "004", 6),
        entry(5, "C5", -3), entry(6, "C6", -2), entry(7, "C7", -6),
        entry(8, "C8", -8), entry(9, "009", 9), entry(10, "C10", -2),
        entry(11, "011", 1),
    ])
    ledger = build_signed_ledger(raw.sample(frac=1, random_state=10))
    m = match_reversals(ledger).set_index("credit_excel_row")
    assert m.loc[[5, 6, 7, 8], "purchase_excel_row"].tolist() == [3, 3, 4, 9]
    assert m.loc[[5, 6], "balance_after"].tolist() == [2, 0]
    assert m.loc[8, "later_excel_row_same_time"] and m.loc[8, "same_time"]
    assert m.loc[10, "status"] == "unmatched"  # cannot combine two remaining single units
    assert match_reversals(ledger, include_same_time=False).status.eq("unmatched").all()
    independent = match_reversals(ledger, use_balance=False).set_index("credit_excel_row")
    assert independent.loc[10, "purchase_excel_row"] == 3


def test_latest_sufficient_line_and_country_from_credit_with_continuous_thresholds():
    rows = []
    cases = [
        ("United Kingdom", 72, 0, "cancellation"), ("United Kingdom", 72, 1, "return"),
        ("EIRE", 120, 0, "cancellation"), ("Germany", 120, 1, "return"),
        ("Australia", 240, 0, "cancellation"), ("USA", 240, 1, "return"),
        ("Unspecified", 24, 0, "unknown_region"), (None, 24, 0, "unknown_region"),
    ]
    assert len(EUROPE_COUNTRIES) == 23
    for i, (country, hours, minutes, _) in enumerate(cases):
        start = pd.Timestamp("2011-03-01")
        credit_time = start + pd.Timedelta(hours=hours, minutes=minutes)
        rows.extend([
            entry(10*i+2, "earliest", 12, start - pd.Timedelta(days=1), customer=str(i)),
            entry(10*i+3, "chosen", 10, start, customer=str(i), country="Iceland"),
            entry(10*i+4, "too-small", 1, start + pd.Timedelta(hours=1), customer=str(i)),
            entry(10*i+5, "C-test", -3, credit_time, customer=str(i), price=7., country=country),
            entry(10*i+6, "future", 40, credit_time + pd.Timedelta(minutes=1), customer=str(i)),
            entry(10*i+7, "C-again", -10, credit_time, customer=str(i), country=country),
        ])
    m = match_reversals(build_signed_ledger(pd.DataFrame(rows))).set_index("credit_excel_row")
    for i, (_, hours, minutes, expected) in enumerate(cases):
        first = m.loc[10*i+5]
        assert first.status == expected and first.purchase_invoice_no == "chosen"
        assert first.lag_minutes == 60 * hours + minutes and first.balance_after == 7
        # Every matched credit reserves units, including returns and unknown destinations.
        assert m.loc[10*i+7, "purchase_invoice_no"] == "earliest"


def test_empty_credit_table_and_empty_input():
    raw = pd.DataFrame([entry(2, "2", 1)])
    for source in (raw, raw.iloc[:0]):
        t = compute_reversal_tables(source)
        assert t["credit_allocations"].empty
        assert t["match_differences"].empty
        assert t["benchmark"].shape[0] > 0


def test_net_retained_units_follow_links_and_are_invariant_to_operational_threshold():
    raw = pd.DataFrame([
        entry(2, "2", 10), entry(3, "3", 5, sku="00101", price=4),
        entry(4, "4", 1, customer=None, price=19),
        entry(5, "C5", -2, date="2011-03-05", price=7),  # four days after sale 2
        entry(6, "C6", -3, date="2011-03-09", price=1, country="Unspecified"),
        entry(7, "C7", -5, date="2011-03-09", sku="00101", price=3),
        entry(8, "C8", -7, date="2011-03-09", price=1),  # only 5 reversible units remain
        entry(9, "C9", -1, date="2011-03-09", customer=None, price=1),
    ])
    ledger = build_signed_ledger(raw)
    baseline = match_reversals(ledger)
    alternate = match_reversals(ledger, uk_hours=120)
    for matches in [baseline, alternate]:
        result = reversals._scenario_summary("sensitivity", ledger, matches, use_balance=True).set_index("metric")
        # All three matched credits count once. Sale 3 has no units left. Sale 2
        # keeps (10 - 2 - 3) units at £2. The unlinked and anonymous C stay in A.
        assert result.loc["known_net_retained_at_sale_prices", ["rows", "units", "signed_amount_gbp"]].tolist() == [1, 5, 10]
        assert result.loc["net_retained_at_sale_prices", ["rows", "units", "signed_amount_gbp"]].tolist() == [2, 6, 29]
        assert result.loc["view_b", "signed_amount_gbp"] == 1
        assert result.loc["view_a", "signed_amount_gbp"] == 19
    old = baseline.set_index("credit_excel_row")
    new = alternate.set_index("credit_excel_row")
    pd.testing.assert_series_equal(old.purchase_excel_row, new.purchase_excel_row)
    assert old.loc[5, "status"] == "return" and new.loc[5, "status"] == "cancellation"
    assert old.loc[6, "status"] == "unknown_region"
    assert old.loc[8, "status"] == "unmatched" and old.loc[9, "status"] == "anonymous"
