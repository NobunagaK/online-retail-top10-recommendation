import pandas as pd

from src.data.clean import audit_transactions


COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


def transaction(invoice, quantity=1, price=2.5, customer="17850", country="UK"):
    return (invoice, "85123A", "Lamp", quantity, pd.Timestamp("2011-01-04"),
            price, customer, country)


def test_cleaning_audit_counts_overlapping_flags_before_sequential_removals():
    sale = transaction("101")
    cancelled_return = transaction("C102", quantity=-2, price=0, customer=None)
    frame = pd.DataFrame([
        sale, sale, transaction("101", country="France"),
        cancelled_return, cancelled_return, transaction("C103"),
        transaction("104", quantity=-4, price=-1), transaction("105", quantity=0),
        transaction("106", price=-3), transaction("107", price=0),
        transaction("c108"), transaction("109", customer=None),
    ], columns=COLUMNS)

    audit = audit_transactions(frame)

    assert list(audit.columns) == [
        "step", "rule", "raw_flagged", "rows_before", "rows_removed", "rows_after",
    ]
    assert audit.iloc[:, 1:].to_records(index=False).tolist() == [
        ("exact_duplicate", 2, 12, 2, 10),
        ("missing_customer_id", 3, 10, 2, 8),
        ("cancelled_invoice", 3, 8, 1, 7),
        ("nonpositive_quantity", 4, 7, 2, 5),
        ("nonpositive_unit_price", 5, 5, 2, 3),
    ]


def test_empty_transactions_report_zero_for_every_cleaning_rule():
    audit = audit_transactions(pd.DataFrame(columns=COLUMNS))

    assert len(audit) == 5
    counts = audit[["raw_flagged", "rows_before", "rows_removed", "rows_after"]]
    assert (counts == 0).all().all()
