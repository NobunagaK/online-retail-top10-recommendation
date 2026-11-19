"""Literal workbook evidence for transaction exceptions. These signals are not labels."""

import re

import pandas as pd


# Keep the text exactly as written for the evidence column. Word boundaries prevent
# matches such as "count" in "country". A standalone "check" is a note, whereas
# "BROWN CHECK CAT DOORSTOP" describes a product.
_NOTE_RULES = {
    "damage_or_disposal": r"\b(?:damages?|damaged|breakages?|broken|faulty|smashed|crushed|cracked|destroyed|unsaleable|throw(?:n)? away|mouldy)\b",
    "missing_or_lost": r"\b(?:missing|lost|can't find|not rcvd)\b",
    "stock_adjustment_or_count": (
        r"\b(?:adjust(?:ment)?(?!\s+bad debt)|counted|stock check|incorrect stock entry|found)\b"
        r"|^\s*check(?=[\s?.!]*$)"
    ),
    "set_or_unit_sale_wording": (
        r"\bsold (?:as (?:sets?|1)|sets?)(?:/\d+)?"
        r"(?: (?:on|by) (?:dotcom|amazon)(?: and amazon)?)?\b"
    ),
    "item_code_or_label_discrepancy": (
        r"\b(?:(?:code|label) mix[- ]up|wrong(?:ly)? (?:coded|marked|mrked|code|barcode)"
        r"|came coded as|barcode problem)\b"
    ),
    "sample_or_promotion": r"\b(?:samples?|promotion(?:al)?|given away|free sample)\b",
    "fee_or_charge": r"\b(?:postage|carriage|bank charges|amazon fee|CRUK commission)\b",
    "discount": r"\bdiscount\b",
    "manual": r"^\s*manual\s*$",
    "debt": r"\bbad debt\b",
    "credit_note_wording": r"\b(?:credit|credits|credited|creditted)\b",
}
_NOTE_PATTERNS = {key: re.compile(expr, re.IGNORECASE) for key, expr in _NOTE_RULES.items()}
# These exact, case-sensitive codes have corresponding workbook descriptions.
_SPECIAL_CODES = {
    "S": "sample", "D": "discount", "B": "bad_debt", "M": "manual", "m": "manual",
    "POST": "postage", "DOT": "dotcom_postage", "C2": "carriage",
    "AMAZONFEE": "amazon_fee", "BANK CHARGES": "bank_charges", "CRUK": "commission",
}
# Workbook invoice identifiers have six digits, sometimes with a literal C or A
# prefix. Short numbers (e.g. "see 47") and longer alphanumeric IDs are not links.
_INVOICE_TOKEN = re.compile(r"\b[AC]?[0-9]{6}\b")


def _note_matches(text):
    groups, terms = [], []
    for name, pattern in _NOTE_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            groups.append(name)
            terms.extend(match.strip() for match in matches)
    status = "blank" if not text.strip() else "unknown"
    return "|".join(groups) or status, "|".join(dict.fromkeys(terms))


def _mentioned_invoices(text):
    return list(dict.fromkeys(_INVOICE_TOKEN.findall(text)))


def description_evidence(raw: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    """Return four evidence strings on flagged.index in the same order.

    Look up the original Description and StockCode by unique worksheet ExcelRow.
    Multiple signals, exact matched substrings, or distinct invoice tokens are
    joined by '|'. Literal text matches ignore case. Special codes match exactly.
    'blank' means missing/whitespace text. 'unknown' means no recognized text
    pattern; this also covers ordinary product names. An empty code signal or
    invoice list means no match. A note or code never establishes a return.
    """
    rows = raw[["ExcelRow", "Description", "StockCode"]].set_index("ExcelRow").loc[flagged["ExcelRow"]]
    notes = rows["Description"].astype("string").fillna("")
    matches = {note: _note_matches(note) for note in notes.unique()}
    references = {note: "|".join(_mentioned_invoices(note)) for note in matches}
    result = pd.DataFrame({
        "description_note_signals": notes.map({k: v[0] for k, v in matches.items()}),
        "description_matched_terms": notes.map({k: v[1] for k, v in matches.items()}),
        "special_code_signal": rows["StockCode"].map(_SPECIAL_CODES).fillna(""),
        "mentioned_invoice_ids": notes.map(references),
    })
    result.index = flagged.index
    return result


_TRACE_COLUMNS = [
    "ExcelRow", "InvoiceNo", "StockCode", "Description", "mentioned_invoice_id",
    "referenced_invoice_rows", "earlier_positive_quantity_rows",
    "earlier_positive_quantity_sku_rows", "earlier_same_sku_rows",
    "earlier_credit_or_nonpositive_rows", "earlier_credit_or_nonpositive_sku_rows",
    "example_earlier_same_sku_excel_row",
]


def mentioned_invoice_trace(raw: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    """Trace each distinct invoice-shaped token per flagged source row, including misses.

    Match the full literal document ID in the whole input (prefix and zeros
    included). referenced_invoice_rows counts all matching raw lines. All other
    counts use strictly earlier InvoiceDate. A positive-quantity line can have a
    nonpositive price. The credit-or-nonpositive signal means a C-prefixed invoice,
    Quantity <= 0, or UnitPrice <= 0. Counts can overlap. Missing source dates or
    required SKU keys yield NA. The example is the lowest ExcelRow among strictly
    earlier exact invoice+SKU matches. An invoice match alone never links a return.
    """
    descriptions = flagged["Description"].astype("string").fillna("")
    extracted = descriptions.map({note: _mentioned_invoices(note) for note in descriptions.unique()})
    mentioned = flagged.loc[extracted.str.len().gt(0), [
        "ExcelRow", "InvoiceNo", "StockCode", "Description", "InvoiceDate",
    ]].copy()
    mentioned["mentioned_invoice_id"] = extracted[extracted.str.len().gt(0)]
    mentioned = mentioned.explode("mentioned_invoice_id")
    target_rows = raw.loc[raw["InvoiceNo"].isin(mentioned["mentioned_invoice_id"])]
    grouped = {key: group for key, group in target_rows.groupby("InvoiceNo", sort=False)}
    empty = target_rows.iloc[:0]
    records = []
    for row in mentioned.itertuples(index=False):
        lines = grouped.get(row.mentioned_invoice_id, empty)
        record = {
            "ExcelRow": row.ExcelRow, "InvoiceNo": row.InvoiceNo, "StockCode": row.StockCode,
            "Description": row.Description, "mentioned_invoice_id": row.mentioned_invoice_id,
            "referenced_invoice_rows": len(lines),
        }
        if pd.notna(row.InvoiceDate):
            earlier = lines.loc[lines["InvoiceDate"].lt(row.InvoiceDate)]
            positive = earlier["Quantity"].gt(0)
            credit_or_nonpositive = (
                earlier["InvoiceNo"].str.startswith("C", na=False)
                | earlier["Quantity"].le(0) | earlier["UnitPrice"].le(0)
            )
            record["earlier_positive_quantity_rows"] = int(positive.sum())
            record["earlier_credit_or_nonpositive_rows"] = int(credit_or_nonpositive.sum())
            if pd.notna(row.StockCode):
                sku = earlier["StockCode"].eq(row.StockCode)
                record["earlier_same_sku_rows"] = int(sku.sum())
                record["earlier_positive_quantity_sku_rows"] = int((positive & sku).sum())
                record["earlier_credit_or_nonpositive_sku_rows"] = int((credit_or_nonpositive & sku).sum())
                record["example_earlier_same_sku_excel_row"] = earlier.loc[sku, "ExcelRow"].min()
        records.append(record)
    trace = pd.DataFrame.from_records(records, columns=_TRACE_COLUMNS)
    return trace.astype({col: "Int64" for col in _TRACE_COLUMNS[5:]})


def sku_name_evidence(raw: pd.DataFrame, name: str = "BATHROOM METAL SIGN") -> pd.DataFrame:
    """One row per exact raw (Description, StockCode) in a case-sensitive trimmed lookup.

    Trimming only selects examples. Grouping retains original label whitespace,
    capitalization and exact SKU. Counts include all workbook lines. first_excel_row
    lets the reader inspect each original label and source SKU on the worksheet.
    No item map is changed.
    """
    descriptions = raw["Description"].astype("string")
    rows = raw.loc[descriptions.str.strip().eq(name.strip()).fillna(False)]
    result = (
        rows.groupby(["Description", "StockCode"], sort=True, dropna=False)
        .agg(workbook_rows=("ExcelRow", "size"), first_excel_row=("ExcelRow", "min"))
        .reset_index()
    )
    result["exact_description_length"] = result["Description"].astype("string").str.len()
    result["distinct_skus_for_exact_description"] = (
        result.groupby("Description", dropna=False)["StockCode"].transform("nunique")
    )
    result["distinct_skus_for_trimmed_name"] = rows["StockCode"].nunique()
    return result
