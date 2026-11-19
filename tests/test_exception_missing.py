import pandas as pd

from src.data.clean import RAW_COLUMNS, audit_transactions
from src.data.exception_missing import missingness_tables


def test_missing_invoice_evidence_and_first_rule_attribution():
    def row(invoice, customer, description="Lamp", quantity=1, price=2):
        return (invoice, "X", description, quantity, pd.Timestamp("2011-01-05"),
                price, customer, "UK")

    cancelled = row("C1", None, None, -1, 0)
    raw = pd.DataFrame([
        row("C1", "10", quantity=-2), cancelled, cancelled,
        row("C1", "11"), row("2", "10", quantity=0, price=-2),
        row("3", "10", price=0), row("4", None), row("5", "10"),
    ], columns=RAW_COLUMNS, index=range(10, 18))
    raw["ExcelRow"] = range(2, 10)
    original = raw.copy(deep=True)
    result = missingness_tables(raw)
    pd.testing.assert_frame_equal(raw, original)

    missing = result["missingness"].set_index("field")
    assert missing.loc["Description", "missing_with_missing_customer_id"] == 2
    assert missing.loc["CustomerID", "missing_pct"] == 37.5
    assert missing.loc["CustomerID", "invoices_with_missing"] == 2
    assert missing.loc[["Quantity", "UnitPrice"], "missing_rows"].tolist() == [0, 0]

    invoices = result["customer_id_invoices"].set_index("InvoiceNo")
    assert invoices.loc["C1"].to_dict() == {
        "first_excel_row": 2, "first_missing_excel_row": 3, "total_rows": 4,
        "missing_customer_rows": 2, "known_customer_ids": 2,
    }
    assert invoices.loc["4", "known_customer_ids"] == 0

    attribution = result["filter_attribution"].set_index("scope")
    assert (attribution.iloc[:, 1:].sum(axis=1) == attribution.raw_rows).all()
    audit = audit_transactions(raw)
    assert attribution.loc["all_rows", audit.rule.tolist()].tolist() == audit.rows_removed.tolist()
    assert attribution.loc["Q_nonpositive", ["exact_duplicate", "missing_customer_id",
                                           "cancelled_invoice", "nonpositive_quantity"]].tolist() == [1, 1, 1, 1]
    assert attribution.loc["negative_price", "nonpositive_unit_price"] == 0
    assert attribution.loc["zero_price", "nonpositive_unit_price"] == 1


def test_empty_sheet_has_no_missing_customer_invoices():
    raw = pd.DataFrame(columns=["ExcelRow", *RAW_COLUMNS])
    result = missingness_tables(raw)
    assert result["customer_id_invoices"].empty
    assert result["missingness"].missing_rows.eq(0).all()
    assert result["filter_attribution"].raw_rows.eq(0).all()
