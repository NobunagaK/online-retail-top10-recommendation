import pandas as pd

from src.data.purchase_evidence import purchase_evidence


def transactions():
    frame = pd.DataFrame([
        # Repeated raw lines both count; ExcelRow breaks ties for the reference.
        (2,  "101",   "A", "1",  1, 2.0, "2011-01-01"),
        (3,  "101",   "A", "1",  1, 2.0, "2011-01-01"),
        (4,  "C101",  "B", "1",  1, 2.0, "2011-01-02"),
        (5,  "C101",  "A", "1",  1, 2.0, "2011-01-02"),
        (6,  "100",   "A", "1",  1, 0.0, "2011-01-03"),
        (7,  "107",   "A", "2",  1, 2.0, "2011-01-03"),
        (8,  "103",   "A", "1",  1, 2.0, "2011-01-04"),
        (9,  "C101",  "A", "1", -1, 2.0, "2011-01-04"),
        (10, "C101",  "A", "1",  1, 2.0, "2011-01-05"),
        (11, "C101",  "A", None, -1, 2.0, "2011-01-01"),
        (12, "008",   "a", "1", -1, 2.0, "2011-01-06"),
    ], columns=["ExcelRow", "InvoiceNo", "StockCode", "CustomerID", "Quantity",
                "UnitPrice", "InvoiceDate"])
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    for key in ("InvoiceNo", "StockCode", "CustomerID"):
        frame[key] = frame[key].astype("string")
    frame["Description"] = "Product"
    frame["Country"] = "UK"
    return frame


def test_full_sheet_occurrences_and_strictly_earlier_lines_use_different_scopes():
    raw = transactions().sample(frac=1, random_state=17)
    flagged = raw.set_index("ExcelRow", drop=False).loc[[12, 9, 11]]
    flagged.index = pd.Index([5, 5, 1], name="input_index")
    evidence = purchase_evidence(raw, flagged)

    pd.testing.assert_index_equal(evidence.index, flagged.index)
    row = evidence.iloc[1]
    assert row["invoice_rows_full_sheet"] == 5
    assert row["earlier_positive_invoice_rows"] == 2
    assert row["earlier_positive_invoice_sku_rows"] == 1
    assert row["earlier_positive_sku_rows"] == 5
    assert row["earlier_positive_customer_sku_rows"] == 4
    assert row["latest_plausible_customer_sku_excel_row"] == 3
    assert row["latest_plausible_customer_sku_date"] == pd.Timestamp("2011-01-01")

    # Lowercase SKU is a distinct key, so absence of history is observable.
    assert evidence.iloc[0]["earlier_positive_sku_rows"] == 0
    assert evidence.iloc[0]["earlier_positive_customer_sku_rows"] == 0
    assert pd.isna(evidence.iloc[0]["latest_plausible_customer_sku_excel_row"])
    # Missing customer does not prevent SKU or invoice lookups.
    assert evidence.iloc[2]["earlier_positive_sku_rows"] == 0
    assert evidence.iloc[2]["invoice_rows_full_sheet"] == 5
    assert pd.isna(evidence.iloc[2]["earlier_positive_customer_sku_rows"])
    assert pd.isna(evidence.iloc[2]["latest_plausible_customer_sku_date"])


def test_no_positive_candidate_and_empty_flag_set():
    raw = transactions()
    raw = raw.loc[raw["Quantity"].lt(0)]
    evidence = purchase_evidence(raw, raw)
    assert evidence["earlier_positive_sku_rows"].eq(0).all()
    assert evidence["latest_plausible_customer_sku_excel_row"].isna().all()
    assert evidence["latest_plausible_customer_sku_date"].isna().all()
    empty = purchase_evidence(raw, raw.iloc[:0])
    pd.testing.assert_frame_equal(empty, evidence.iloc[:0])


def test_cancellation_diagnostics_use_independent_exact_comparisons():
    raw = transactions()
    # The matching invoice/SKU is a coincidence. Narrower price/quantity evidence
    # can point to a sale on an entirely different invoice.
    raw.loc[raw.ExcelRow.eq(2), "Quantity"] = 8
    raw.loc[raw.ExcelRow.eq(3), "UnitPrice"] = 4.0
    raw.loc[raw.ExcelRow.eq(7), ["Quantity", "UnitPrice", "CustomerID"]] = [8, 2.0, "1"]
    raw.loc[raw.ExcelRow.eq(9), "Quantity"] = -8
    evidence = purchase_evidence(raw, raw.set_index("ExcelRow", drop=False))
    stripped = "earlier_positive_stripped_invoice_sku_rows"
    exact = "earlier_matching_customer_sku_price_opposite_quantity_rows"
    assert evidence.loc[9, stripped] == 2  # rows 2 and 3
    assert evidence.loc[9, exact] == 2  # rows 2 and 7
    assert evidence.loc[5, stripped] == 2
    assert evidence.loc[5, exact] == 0  # a C row with positive quantity
    assert evidence.loc[11, stripped] == 0  # same-time positives are ineligible
    assert pd.isna(evidence.loc[11, exact])  # unknown customer
    assert pd.isna(evidence.loc[12, stripped]) and pd.isna(evidence.loc[12, exact])
