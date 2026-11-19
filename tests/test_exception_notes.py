import pandas as pd

from src.data.exception_notes import description_evidence, mentioned_invoice_trace, sku_name_evidence


def make_transactions(entries):
    frame = pd.DataFrame(entries, columns=[
        "InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "InvoiceDate",
    ])
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    frame["CustomerID"] = pd.Series("U", index=frame.index, dtype="string")
    frame["Country"] = "United Kingdom"
    frame["ExcelRow"] = range(2, len(frame) + 2)
    return frame.astype({"InvoiceNo": "string", "StockCode": "string"})


def test_literal_signals_preserve_terms_and_do_not_label_product_names_or_missing_notes():
    raw = make_transactions([
        ("100001", "M", "Possible DAMAGES or lost? Samples", -1, 0, "2011-01-01"),
        ("A100002", "B", "Adjust bad debt", 1, -12, "2011-01-02"),
        ("100003", "85123A", "CHECK CAT DOORSTOP", -1, 0, "2011-01-02"),
        ("100004", "D", " ", -1, 0, "2011-01-02"),
        ("100005", "S", None, -1, 0, "2011-01-02"),
        ("100006", "POST", "check?", -1, 0, "2011-01-02"),
        ("100007", "PADS", "free standing item", 1, 0, "2011-01-02"),
    ])
    flagged = raw.iloc[[6, 5, 4, 3, 2, 1, 0]]
    evidence = description_evidence(raw, flagged)
    assert evidence.index.tolist() == flagged.index.tolist()
    assert evidence.loc[0, "description_note_signals"] == "damage_or_disposal|missing_or_lost|sample_or_promotion"
    assert evidence.loc[0, "description_matched_terms"] == "DAMAGES|lost|Samples"
    assert evidence.loc[1, "description_note_signals"] == "debt"
    assert evidence.loc[1, "description_matched_terms"] == "bad debt"
    assert evidence.loc[1, "special_code_signal"] == "bad_debt"
    assert evidence.loc[2, "description_note_signals"] == "unknown"
    assert evidence.loc[3, "description_note_signals"] == evidence.loc[4, "description_note_signals"] == "blank"
    assert evidence.loc[5, "description_note_signals"] == "stock_adjustment_or_count"
    assert evidence.loc[6, "special_code_signal"] == ""
    assert evidence.loc[6, "description_note_signals"] == "unknown"


def test_note_wording_for_sets_and_code_discrepancies_is_literal():
    notes = ["sold as set on dotcom", "code mix up? 84930", "wrongly coded-23343",
             "Wrongly mrked had 85123a in box", "BLUE SET OF 6", "HERB MARKER MINT"]
    raw = make_transactions([("101234", "A", note, -1, 0, "2011-01-01") for note in notes])
    evidence = description_evidence(raw, raw)
    assert evidence["description_note_signals"].tolist() == [
        "set_or_unit_sale_wording", "item_code_or_label_discrepancy",
        "item_code_or_label_discrepancy", "item_code_or_label_discrepancy", "unknown", "unknown",
    ]
    assert evidence["description_matched_terms"].tolist() == [
        "sold as set on dotcom", "code mix up", "wrongly coded", "Wrongly mrked", "", "",
    ]


def test_literal_invoice_trace_preserves_prefix_timing_and_sku_with_unresolved_mentions():
    raw = make_transactions([
        ("C550456", "P", "Item P", -2, 5, "2011-01-01"),
        ("C550456", "Q", "Item Q", -1, 5, "2011-01-01"),
        ("550456", "P", "Item P", 2, 5, "2011-01-01"),
        ("550456", "P", "Item P", 4, 5, "2011-01-05"),
        ("550456", "P", "Item P", 4, 5, "2011-01-06"),
        ("554553", "P", "credited C550456 see 47; 550456 C550456 A999999 1234567 X550456", -2, 0, "2011-01-05"),
        ("554554", None, "C550456", 1, 0, "2011-01-05"),
        ("554555", "P", "C550456", 1, 0, None),
    ])
    flagged = raw.iloc[[5, 6, 7]]
    evidence = description_evidence(raw, flagged)
    assert evidence.loc[5, "mentioned_invoice_ids"] == "C550456|550456|A999999"
    trace = mentioned_invoice_trace(raw, flagged)
    assert len(trace) == 5
    credit, sale, missing = trace.iloc[:3].to_dict("records")
    assert credit["referenced_invoice_rows"] == credit["earlier_credit_or_nonpositive_rows"] == 2
    assert credit["earlier_positive_quantity_rows"] == 0
    assert credit["earlier_credit_or_nonpositive_sku_rows"] == credit["earlier_same_sku_rows"] == 1
    assert credit["example_earlier_same_sku_excel_row"] == 2
    assert sale["referenced_invoice_rows"] == 3
    assert sale["earlier_positive_quantity_sku_rows"] == 1
    assert sale["earlier_credit_or_nonpositive_rows"] == missing["referenced_invoice_rows"] == 0
    assert pd.isna(trace.iloc[3]["earlier_same_sku_rows"])
    assert pd.isna(trace.iloc[4]["earlier_positive_quantity_rows"])


def test_trimmed_name_search_keeps_exact_labels_and_sku_rows_separate():
    raw = make_transactions([
        ("1", "82580", "BATHROOM METAL SIGN", 1, 5, "2011-01-01"),
        ("2", "21171", "BATHROOM METAL SIGN ", 1, 5, "2011-02-01"),
        ("3", "82580", "BATHROOM METAL SIGN", -1, 5, "2011-03-01"),
        ("4", "x", "Bathroom Metal Sign", 1, 5, "2011-03-01"),
    ])
    summary = sku_name_evidence(raw).set_index("StockCode")
    assert summary.loc["21171", "Description"] == "BATHROOM METAL SIGN "
    assert summary.loc["82580", "workbook_rows"] == 2
    assert summary.loc["21171", "first_excel_row"] == 3
    assert (summary["distinct_skus_for_exact_description"] == 1).all()
    assert (summary["distinct_skus_for_trimmed_name"] == 2).all()
    assert sku_name_evidence(raw, "other item").empty
    assert mentioned_invoice_trace(raw, raw).empty
    assert description_evidence(raw, raw.iloc[:0]).empty
