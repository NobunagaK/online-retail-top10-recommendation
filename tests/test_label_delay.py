import pandas as pd

from scripts.audit_label_delay import compare_month_ends


def test_label_delay_uses_distinct_month_keys_and_credit_visibility():
    rows = [
        (2, "1", "A", 1, "2011-09-30 10:00"),
        (3, "2", "A", 2, "2011-09-30 11:00"),
        (4, "3", "B", 1, "2011-09-27 10:00"),
        (5, "4", "B", 1, "2011-09-28 10:00"),
        (6, "5", "C", 1, "2011-09-29 10:00"),
        (7, "6", "D", 3, "2011-09-29 10:00"),
        (8, "7", "E", 1, "2011-09-30 23:59"),
        (9, "7", "F", 1, "2011-10-01 00:00"),
        (10, "8", "N", 1, "2011-11-01 00:00"),
    ]
    ledger = pd.DataFrame(rows, columns=["ExcelRow", "InvoiceNo", "StockCode", "Quantity", "InvoiceDate"])
    ledger = ledger.assign(CustomerID="001", Description="item", Country="UK", UnitPrice=1, ledger_type="sale")
    ledger["InvoiceDate"] = pd.to_datetime(ledger["InvoiceDate"])
    credits = pd.DataFrame([
        (2, "2011-10-01 00:00", 1), (3, "2011-11-03 00:00", 2),
        (4, "2011-10-02 00:00", 1), (6, "2011-09-30 23:59", 1), (7, "2011-10-03 00:00", 2),
    ], columns=["purchase_excel_row", "credit_time", "credited_units"]).assign(status="return")
    credits["credit_time"] = pd.to_datetime(credits["credit_time"])
    remaining = ledger.loc[ledger["ExcelRow"].isin([5, 7, 8, 9, 10])].copy()
    remaining["basket_time"] = remaining.groupby("InvoiceNo")["InvoiceDate"].transform("max")
    report = compare_month_ends(ledger, credits, remaining, pd.Timestamp("2011-12-09 12:50")).set_index("target_month")
    assert list(report.index) == [f"2011-{month:02}" for month in range(2, 11)]
    sep = report.loc["2011-09"]
    # B survives another invoice, D retains one unit, and C was already gone.
    assert (sep.month_end_positive_customer_skus, sep.still_positive_at_observed_end) == (3, 2)
    assert (sep.positive_to_zero, sep.depleted_during_following_month, sep.depleted_after_following_month) == (1, 0, 1)
    assert sep.share_of_month_end_positives == 1 / 3
    # E and F are both assigned to their invoice's October completion time.
    assert report.at["2011-10", "month_end_positive_customer_skus"] == 2
    assert pd.isna(report.at["2011-02", "share_of_month_end_positives"])
