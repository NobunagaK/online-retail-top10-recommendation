"""Replay merchandise purchase quantities as known at a specified exclusive cutoff."""

import pandas as pd

from src.data.baskets import INVOICE_KEYS, build_basket_events
from src.data.clean import RAW_COLUMNS


def purchase_line_balances(ledger, allocations, *, cutoff=None):
    """Return all identified positive source lines in completed baskets, including fully credited ones.

    Both inputs must use canonical allocation results and Excel worksheet row
    identifiers. For historical replay, pass a cutoff. Only baskets completed
    and credits observed strictly before that timestamp enter.
    None requests a full-workbook retrospective view. The immutable invoice
    completion is computed before subtracting any matched credits. Allocation to
    the source line determines the deduction. Elapsed-time classes describe it.
    """
    sale = ledger["ledger_type"].eq("sale") & ledger["CustomerID"].notna()
    lines = ledger.loc[sale, ["ExcelRow", *RAW_COLUMNS]].copy()
    lines["InvoiceDate"] = pd.to_datetime(lines["InvoiceDate"], format="mixed")
    lines["basket_time"] = lines.groupby(INVOICE_KEYS, sort=False)["InvoiceDate"].transform("max")
    effective = allocations.loc[allocations["purchase_excel_row"].notna()]
    if cutoff is not None:
        end = pd.Timestamp(cutoff)
        lines = lines.loc[lines["basket_time"].lt(end)].copy()
        credit_dates = pd.to_datetime(effective["credit_time"], format="mixed")
        effective = effective.loc[credit_dates.lt(end)]

    lines["original_quantity"] = lines["Quantity"]
    for status, column in [("cancellation", "cancelled_units"), ("return", "returned_units")]:
        totals = (
            effective.loc[effective["status"].eq(status)]
            .groupby("purchase_excel_row")["credited_units"].sum()
        )
        lines[column] = lines["ExcelRow"].map(totals).fillna(0).astype("int64")
    matched = effective.groupby("purchase_excel_row")["credited_units"].sum()
    lines["matched_credit_units"] = lines["ExcelRow"].map(matched).fillna(0).astype("int64")
    lines["other_matched_units"] = lines["matched_credit_units"] - lines["cancelled_units"] - lines["returned_units"]
    lines["Quantity"] = lines["original_quantity"] - lines["matched_credit_units"]
    if lines["Quantity"].lt(0).any():
        raise ValueError("Matched credit units exceed the original quantity on a sale line.")
    components = {"cancelled_units": "cancellation", "returned_units": "return", "other_matched_units": "other"}
    lines["matched_credit_composition"] = "none"
    for col, label in components.items():
        lines.loc[lines[col].gt(0), "matched_credit_composition"] = label
    mixed = lines[list(components)].gt(0).sum(axis=1).gt(1)
    lines.loc[mixed, "matched_credit_composition"] = "mixed"
    lines["line_amount_gbp"] = lines["Quantity"] * lines["UnitPrice"]
    return lines.sort_values("ExcelRow", ignore_index=True)


def purchase_snapshot(ledger, allocations, *, cutoff=None):
    """Return net-positive lines and their invoice-item events at the exclusive cutoff."""
    balances = purchase_line_balances(ledger, allocations, cutoff=cutoff)
    purchases = balances.loc[balances["Quantity"].gt(0)].copy()
    return purchases, build_basket_events(purchases)


def net_retained_labels(ledger, allocations, *, target_start, target_end):
    """Distinct customer/SKU positives in [target_start, target_end), replayed as of target_end.

    Purchase time is the original basket completion. A later observed credit
    cannot rewrite an earlier target window. Cohort eligibility and catalog
    intersection remain separate query-construction decisions.
    """
    start, end = pd.Timestamp(target_start), pd.Timestamp(target_end)
    if start >= end:
        raise ValueError("The target window must end after it starts.")
    _, events = purchase_snapshot(ledger, allocations, cutoff=end)
    keys = ["CustomerID", "StockCode"]
    return (events.loc[events["basket_time"].ge(start), keys].drop_duplicates()
            .sort_values(keys, ignore_index=True))


def snapshot_decisions(raw, gross_steps, gross_decisions, ledger, allocations, balances):
    """Assign every source row one retrospective disposition for the recommendation table.

    Preserve the original pre-credit eligibility decision separately. Units on a
    positive source row distinguish full/partial credits from unaffected lines.
    Composition preserves the operational classes, including mixed cases.
    Ledger membership describes a separate signed-accounting view.
    """
    frame = raw.set_index("ExcelRow")
    decisions = gross_decisions.set_index("ExcelRow").rename(columns={"first_rule": "gross_eligibility_rule"})
    first = decisions["gross_eligibility_rule"].copy()
    duplicate = first.eq("exact_duplicate")
    is_credit = frame["InvoiceNo"].astype("string").str.startswith("C", na=False)
    writeoff = ~is_credit & frame["Quantity"].lt(0)
    debt = frame["InvoiceNo"].astype("string").str.startswith("A", na=False)
    first.loc[writeoff & ~duplicate] = "nonC_negative_inventory"
    first.loc[debt & ~duplicate] = "bad_debt_document"

    credit_status = allocations.set_index("credit_excel_row")["status"].reindex(first.index)
    signed = ledger.set_index("ExcelRow")["ledger_type"].reindex(first.index)
    # A C source line outside the merchandise ledger can represent e.g. a fee.
    outside = first.eq("cancelled_invoice") & credit_status.isna()
    first.loc[outside] = "credit_outside_ledger"
    first.loc[credit_status.notna()] = "credit_" + credit_status.dropna().astype("string")
    sale = balances.set_index("ExcelRow")
    first.loc[sale.index] = "retained"
    first.loc[sale.index[sale["matched_credit_units"].gt(0)]] = "sale_partially_credited"
    first.loc[sale.index[sale["Quantity"].eq(0)]] = "sale_fully_credited"
    kept_rules = ["retained", "sale_partially_credited"]
    retained_ids = first.index[first.isin(kept_rules)].sort_values()
    if not retained_ids.equals(sale.index[sale["Quantity"].gt(0)].sort_values()):
        raise ValueError("Source decisions and the signed-ledger purchase snapshot disagree.")

    decisions["first_rule"] = first
    decisions["in_signed_ledger_a"] = signed.notna()
    decisions["in_signed_ledger_b"] = signed.notna() & frame["CustomerID"].notna()
    decisions["credit_status"] = credit_status
    decisions["matched_purchase_excel_row"] = allocations.set_index("credit_excel_row")["purchase_excel_row"].reindex(first.index).astype("Int64")
    decisions["matched_credit_composition"] = sale["matched_credit_composition"].reindex(first.index)
    for col in ("original_quantity", "cancelled_units", "returned_units", "other_matched_units", "matched_credit_units", "Quantity"):
        name = "retained_quantity" if col == "Quantity" else col
        decisions[name] = sale[col].reindex(first.index).astype("Int64")

    counts = first.value_counts()
    rows = []
    before = len(raw)

    def add(rule, action, flagged, sources, reason_en, reason_zh):
        nonlocal before
        n = int(counts.get(rule, 0))
        removed = n if action == "exclude" else 0
        rows.append({
            "step": len(rows) + 1, "rule": rule, "action": action,
            "raw_flagged": int(flagged), "decision_rows": n, "rows_before": before,
            "rows_removed": removed, "rows_after": before - removed,
            "evidence_sheets": sources, "reason_en": reason_en, "reason_zh": reason_zh,
        })
        before -= removed

    gross = gross_steps.set_index("rule")

    def add_gross(rule):
        row = gross.loc[rule]
        add(rule, "exclude", row.raw_flagged, row.evidence_sheets, row.reason_en, row.reason_zh)

    add_gross("exact_duplicate")
    add("bad_debt_document", "exclude", debt.sum(), "P_nonpositive, case_examples",
        "A-prefixed bad-debt documents do not represent a merchandise sale or credit.",
        "A 前缀的坏账调整单不计入商品销售或商品贷项。")
    add("nonC_negative_inventory", "exclude", writeoff.sum(), "Q_nonpositive, nonC_negative, note_summary",
        "Treat non-C negative units as inventory write-offs; the observed rows are anonymous and zero-priced.",
        "非 C 负数量行按库存核销处理；原表这些行均匿名且单价为零。")
    credit_reasons = {
        "cancellation": ("Deduct linked units from the original sale at its original unit price.",
                         "在原正单上按原单价扣减匹配的取消数量。"),
        "return": ("Deduct linked units at the original sale price. Preserve the return-proxy classification for analysis.",
                   "按原正单单价扣减已匹配件数，保留按时限判定的退货标记以供分析。"),
        "unmatched": ("Keep the signed credit; there is no supported purchase link.",
                      "保留带符号贷项，缺乏可核实的原单链接。"),
        "anonymous": ("Keep the anonymous credit in view A; a customer link is unavailable.",
                      "匿名贷项进入整体视图 A，无法关联客户。"),
        "unknown_region": ("The region has no timing threshold. Deduct any units with a confirmed source-line allocation.",
                           "地区不明时保留单独的分析类别；已匹配到具体正单的件数仍予扣减。"),
    }
    for label, (en, zh) in credit_reasons.items():
        rule = "credit_" + label
        add(rule, "exclude", credit_status.eq(label).sum(), "credit_allocations, credit_summary", en, zh)
    for rule in gross.index:
        if rule not in {"exact_duplicate", "cancelled_invoice", "retained"}:
            add_gross(rule)
        elif rule == "cancelled_invoice":
            add("credit_outside_ledger", "exclude", (is_credit & signed.isna()).sum(),
                "C_invoices, ledger_summary, merchandise_rules.csv",
                "This C line falls outside the cleaned merchandise ledger.",
                "该 C 明细未进入清洗后的商品贷项账表。")
    sources = "credit_allocations, reversal_examples"
    add("sale_fully_credited", "exclude", sale["Quantity"].eq(0).sum(), sources,
        "All units on the original sale line have matching credits.", "原正单行的件数已被匹配贷项全部冲减。")
    add("sale_partially_credited", "adjust", (sale["matched_credit_units"].gt(0) & sale["Quantity"].gt(0)).sum(), sources,
        "Retain the net positive units at the original sale price.", "保留贷项冲减后剩余的正件数，按原单价计额。")
    add("retained", "retain", counts.get("retained", 0), "ledger_summary",
        "Keep the original positive merchandise purchase.", "保留原始正向商品购买。")
    audit = pd.DataFrame.from_records(rows)
    if audit["decision_rows"].sum() != len(raw):
        raise ValueError("Some source rows have no final audit rule.")
    actions = audit.set_index("rule")["action"]
    decisions["final_action"] = first.map(actions)
    return audit, decisions.reset_index()
