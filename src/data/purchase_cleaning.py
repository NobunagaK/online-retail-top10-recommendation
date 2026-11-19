"""Find gross positive purchase candidates before applying matched credits."""

import pandas as pd

from src.data.clean import purchase_filter_flags
from src.data.merchandise import merchandise_rules


def _zero_price_contexts(raw: pd.DataFrame, cancelled: pd.Series) -> dict[str, pd.Series]:
    """Partition all raw zero-price lines, including lines removed by earlier rules.

    Companions have positive quantity and unit price on a non-C invoice with a
    known invoice number. A same-item match additionally requires the exact
    StockCode and CustomerID, with a missing customer key matching only another
    missing key. A priced line can itself be a fee. Search the complete invoice.
    These comparisons describe a zero price. All four cases have the same
    purchase-label decision, irrespective of where the query cutoff falls.
    """
    zero = raw["UnitPrice"].eq(0).fillna(False)
    manual = raw["StockCode"].isin(["M", "m"])
    keys = ["InvoiceNo", "CustomerID", "StockCode"]
    priced = raw.loc[
        ~cancelled & raw["Quantity"].gt(0) & raw["UnitPrice"].gt(0), keys
    ].dropna(subset=["InvoiceNo"]).drop_duplicates()
    priced_keys = priced.dropna(subset=["StockCode"])
    # Limit the three-key comparison to the small set of zero-price rows.
    same_item = pd.Series(False, index=raw.index)
    same_item.loc[zero] = pd.MultiIndex.from_frame(raw.loc[zero, keys]).isin(
        pd.MultiIndex.from_frame(priced_keys)
    )
    same_invoice = raw["InvoiceNo"].isin(priced["InvoiceNo"])
    return {
        "zero_price_manual": zero & manual,
        "zero_price_same_item": zero & ~manual & same_item,
        "zero_price_other_item": zero & ~manual & ~same_item & same_invoice,
        "zero_price_no_priced_companion": zero & ~manual & ~same_invoice,
    }


def clean_purchase_lines(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply gross eligibility rules to the eight original columns plus ExcelRow.

    Input order determines the first copy of an exact eight-field duplicate.
    raw_flagged counts each rule on the original rows. rows_removed counts only
    its first matches among survivors. The retained summary is the size of the
    final output. ExcelRow is the stable worksheet locator in both row tables.
    Credit lines are reviewed separately. Recommendation snapshots incorporate
    classified cancellations observed before their exclusive as-of times.
    """
    if raw["ExcelRow"].isna().any() or not raw["ExcelRow"].is_unique:
        raise ValueError("ExcelRow must uniquely identify each row of the source worksheet.")
    # Work by position even when a caller supplies a filtered or nonunique index.
    frame = raw.reset_index(drop=True)
    flags = purchase_filter_flags(frame)
    surviving = pd.Series(True, index=frame.index)
    first_rule = pd.Series("retained", index=frame.index, dtype="string")
    records = []

    def exclude(rule, flag, sheets, reason_en, reason_zh):
        mask = flag.fillna(False)
        removed = mask & surviving
        before = int(surviving.sum())
        count = int(removed.sum())
        first_rule.loc[removed] = rule
        surviving.loc[removed] = False
        records.append({
            "step": len(records) + 1, "rule": rule, "action": "exclude",
            "raw_flagged": int(mask.sum()), "rows_before": before,
            "rows_removed": count, "rows_after": before - count,
            "evidence_sheets": sheets, "reason_en": reason_en, "reason_zh": reason_zh,
        })

    exclude("exact_duplicate", flags["exact_duplicate"], "filter_attribution",
            "Keep the first occurrence of an identical eight-field record.",
            "按八个原始字段识别完全相同的明细，保留首次出现的一行。")
    exclude("missing_customer_id", flags["missing_customer_id"], "missingness, customer_id_invoices, filter_attribution",
            "An anonymous line cannot be assigned to a customer's history. The source audit retains these lines.",
            "匿名交易无法归入具体客户的历史，原始审计仍涵盖这些行。")
    # Separate field-level rules keep a malformed row traceable, even for fields
    # that happen to be complete throughout this version of the workbook.
    for rule, field, reason_en, reason_zh in [
        ("missing_invoice_no", "InvoiceNo", "An invoice identifier is required to construct a basket.", "缺少发票编号，无法构造购物篮。"),
        ("missing_stock_code", "StockCode", "An item identifier is required to assign a product.", "缺少商品编码，无法确定商品。"),
        ("missing_invoice_date", "InvoiceDate", "The timestamp determines the basket time and eligible history.", "缺少时间，无法确定购物篮发生时间与可用历史。"),
        ("missing_quantity", "Quantity", "The purchased quantity is unknown.", "缺少数量，无法核实购买件数。"),
        ("missing_unit_price", "UnitPrice", "The recorded unit price is unknown.", "缺少单价，无法核实计价金额。"),
    ]:
        exclude(rule, frame[field].isna(), "missingness", reason_en, reason_zh)
    exclude("cancelled_invoice", flags["cancelled_invoice"], "C_invoices, history_summary, case_examples",
            "Route C-prefixed records out of the positive-source stream. Match eligible merchandise credits before determining the purchased quantity.",
            "C 前缀记录不单独产生购买事件。符合条件的商品贷项另行匹配后，再确定最终购买数量。")
    exclude("negative_quantity", frame["Quantity"].lt(0), "Q_nonpositive, nonC_negative, note_summary",
            "Negative units do not add purchased items. Notes can describe damage or adjustments without identifying a buyer.",
            "负数量不形成新增购买。备注可能记录损坏或调整，但不足以确认买家。")
    exclude("zero_quantity", frame["Quantity"].eq(0), "Q_nonpositive, filter_attribution",
            "Zero units add no purchase event.", "零数量不产生购买事件。")
    exclude("negative_unit_price", frame["UnitPrice"].lt(0), "P_nonpositive, case_examples, note_summary",
            "A negative unit price is outside the paid-item definition. Both observed examples record bad-debt adjustments.",
            "负单价不符合计价商品定义，原表两条实例的备注均为坏账调整。")

    zero_price = _zero_price_contexts(frame, flags["cancelled_invoice"])
    zero_sheets = "P_nonpositive, case_examples, history_summary"
    exclude("zero_price_manual", zero_price["zero_price_manual"], "P_nonpositive, note_summary",
            "An exact manual-entry code (M or m) carries a recorded zero unit price.",
            "手工记录编码 M 或 m 的记录单价为零。")
    exclude("zero_price_same_item", zero_price["zero_price_same_item"], zero_sheets,
            "A priced line shares the invoice, customer key and SKU. The zero-price row adds no paid units.",
            "同票另有计价行匹配该客户键和商品编码，零价行不增加计价购买件数。")
    exclude("zero_price_other_item", zero_price["zero_price_other_item"], zero_sheets,
            "This invoice has a priced line, but none matches the customer key and SKU. A priced line can also represent a fee.",
            "同票有计价行，但没有计价行同时匹配本行的客户键和商品编码。计价行也可能是费用。")
    exclude("zero_price_no_priced_companion", zero_price["zero_price_no_priced_companion"], zero_sheets,
            "No non-C line with positive quantity and price was found under this exact invoice number.",
            "该发票编号下未查到非 C 且数量和单价均为正的计价行。")

    frozen = merchandise_rules()
    # These sheets contain examples for the exact code in the source workbook.
    # The training-period registry is the source of every exclusion, including
    # codes with no exceptions (such as lower-case m and two voucher values).
    code_sheets = {
        "23444": "C_invoices, P_nonpositive", "23574": "C_invoices, P_nonpositive",
        "AMAZONFEE": "C_invoices, note_summary", "B": "P_nonpositive, case_examples, note_summary",
        "BANK CHARGES": "C_invoices, note_summary", "C2": "C_invoices, P_nonpositive, note_summary",
        "CRUK": "C_invoices, note_summary", "D": "C_invoices, case_examples, note_summary",
        "DOT": "C_invoices, P_nonpositive, note_summary", "M": "C_invoices, P_nonpositive, note_summary",
        "POST": "C_invoices, P_nonpositive, note_summary", "S": "C_invoices, note_summary",
        "gift_0001_10": "P_nonpositive", "gift_0001_20": "P_nonpositive", "gift_0001_30": "P_nonpositive",
    }
    for item in frozen.loc[frozen["action"].eq("exclude")].itertuples(index=False):
        sources = ["merchandise_rules.csv"]
        if item.StockCode in code_sheets:
            sources.append(code_sheets[item.StockCode])
        exclude(f"non_merchandise_{item.StockCode}", frame["StockCode"].eq(item.StockCode),
                ", ".join(sources), item.reason_en, item.reason_zh)

    retained = int(surviving.sum())
    records.append({
        "step": len(records) + 1, "rule": "retained", "action": "retain",
        "raw_flagged": retained, "rows_before": retained, "rows_removed": 0,
        "rows_after": retained, "evidence_sheets": "filter_attribution, sku_names, merchandise_rules.csv",
        "reason_en": "Keep identified, dated, positive-quantity, positive-price merchandise lines. Preserve exact SKU and invoice IDs, including suffixes.",
        "reason_zh": "保留客户、时间、发票与商品明确且数量和单价均为正的商品明细，按原值保留发票及商品编码（含后缀）。",
    })
    decisions = pd.DataFrame({"ExcelRow": frame["ExcelRow"], "first_rule": first_rule})
    purchases = frame.loc[surviving].copy()
    purchases["line_amount_gbp"] = purchases["Quantity"] * purchases["UnitPrice"]
    return pd.DataFrame.from_records(records), decisions, purchases
