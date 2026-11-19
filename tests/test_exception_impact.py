import pandas as pd
import pytest

from src.data.exception_impact import impact_tables


def test_raw_ledger_and_same_customer_comparison():
    raw = pd.DataFrame([
        ("A", "10", "2011-01-02 23:00", 2, 3.0),
        ("A", "10", "2011-01-02 23:00", 2, 3.0),  # raw duplicate
        ("A", "10", "2011-01-02 23:00", 0, 3.0),  # same invoice in both categories
        ("A", "C11", "2011-01-05 18:00", -1, 3.0),
        ("B", "C12", "2011-01-03 12:00", 1, 4.0),  # even a C line can have positive value
        ("D", "13", "2011-01-04 12:00", 1, 5.0),
        (None, "14", "2011-01-07 12:00", 1, 10.0),
        (None, "15", "2011-01-06 12:00", -2, -1.0),
    ], columns=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"])
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    mask = raw["InvoiceNo"].str.startswith("C") | raw["Quantity"].le(0) | raw["UnitPrice"].le(0)

    tables = impact_tables(raw, mask)
    amounts = tables["amount_by_record_type"].set_index(["record_type", "customer_id_status"])
    assert amounts.rows.sum() == 8
    assert amounts.loc[("exception", "identified")].to_dict() == {
        "rows": 3, "distinct_invoices": 3, "positive_line_gbp": 4.0,
        "negative_line_gbp": -3.0, "signed_gbp": 1.0,
    }
    assert amounts.loc[("exception", "missing"), "signed_gbp"] == 2.0
    assert amounts.loc[("regular", "missing"), "signed_gbp"] == 10.0
    assert amounts.signed_gbp.sum() == (raw.Quantity * raw.UnitPrice).sum()

    customers = tables["customer_impact"].set_index("CustomerID")
    assert set(customers.index) == {"A", "B", "D"}
    a = customers.loc["A"]
    assert (a.cohort, a.regular_invoices, a.all_invoices) == ("mixed", 1, 2)
    assert (a.regular_gross_gbp, a.all_signed_gbp, a.amount_difference_gbp) == (12, 9, -3)
    assert a.recency_reference_date == pd.Timestamp("2011-01-08")
    assert (a.regular_recency_days, a.all_recency_days) == (6, 3)

    b = customers.loc["B"]
    assert b.cohort == "exception_only" and b.regular_invoices == 0 and b.regular_gross_gbp == 0
    assert pd.isna(b.regular_recency_days) and b.all_recency_days == 5
    assert customers.loc["D", "cohort"] == "regular_only"

    cohorts = tables["customer_impact_summary"].set_index("cohort")
    assert cohorts.customers.to_dict() == {"regular_only": 1, "mixed": 1, "exception_only": 1}
    assert cohorts.all_signed_gbp.sum() == customers.all_signed_gbp.sum()
    assert pd.isna(cohorts.loc["exception_only", "median_regular_recency_days"])


def test_empty_ledger_and_misaligned_mask():
    raw = pd.DataFrame({
        "CustomerID": pd.Series(dtype="string"), "InvoiceNo": pd.Series(dtype="string"),
        "InvoiceDate": pd.Series(dtype="datetime64[ns]"),
        "Quantity": pd.Series(dtype="int64"), "UnitPrice": pd.Series(dtype="float64"),
    })
    tables = impact_tables(raw, pd.Series(dtype="bool"))
    assert tables["amount_by_record_type"].rows.tolist() == [0, 0, 0, 0]
    assert tables["customer_impact"].empty
    assert tables["customer_impact_summary"].customers.sum() == 0
    with pytest.raises(ValueError, match="same index"):
        impact_tables(raw, pd.Series([False]))
