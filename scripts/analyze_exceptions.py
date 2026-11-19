"""Explore signed transactions on the full UCI workbook before purchase filters."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from openpyxl.utils import get_column_letter

from src.data.clean import read_workbook
from src.data.exception_impact import impact_tables
from src.data.exception_missing import missingness_tables
from src.data.exception_notes import description_evidence, mentioned_invoice_trace, sku_name_evidence
from src.data.purchase_evidence import purchase_evidence
from src.data.purchase_snapshot import purchase_line_balances
from src.data.reversals import build_signed_ledger, compute_reversal_tables, match_reversals


REVERSAL_GUIDE = [
    ("ledger_summary", "Frozen merchandise policy. Signed sales and credits retain recorded prices. At original sale prices, gross deducts cancellations only and net deducts every matched credit. Source exclusions are first-match counts.", "冻结商品规则。带符号销售与贷项取各行记录价格；按原销售单价，毛额只扣匹配取消，净额扣除所有可匹配贷项。来源筛选按最先命中的规则归因。"),
    ("credit_summary", "Canonical merchandise credits by destination region and operational status. Amounts use recorded credit prices. Time thresholds are regional assumptions.", "按贷项目的地区与操作分类汇总实际规则下的商品贷项，金额取贷项单价。时限是项目对地区的假设。"),
    ("credit_allocations", "Every eligible merchandise credit with matched sale row, time lag, region, threshold and allocation balance. An unmatched row has no allocated sale.", "全部符合条件的商品贷项，列出匹配销售行、时间差、地区、时限及分配余额；未匹配的贷项无对应销售行。"),
    ("credit_controls", "Compare SKU, timing and allocation assumptions with the discussion benchmarks, including per-credit full/partial counts and later same-SKU purchases.", "对照商品码、时间和贷项分配方式，复核讨论稿给出的逐条全退、部分退和后来同客同货的购买比例。"),
    ("credit_control_rows", "Credit rows whose inclusion, matched sale or class changes between scenarios. Use the source row numbers to reproduce a difference.", "对照情景之间在纳入状态、对应销售行或操作类别上有变化的贷项，可用原始行号复核。"),
    ("reversal_examples", "Selected credit and sale links including two large reversals, unequal prices, partial cancellations and returns. Remaining allocation capacity equals net retained units after each matched credit.", "追踪两笔大额撤销、价格不相等、部分取消及退货示例的贷项与销售。每次匹配后，可冲余额即净保留数量。"),
    ("sku_policy_differences", "Source rows with different decisions under the five-leading-digit control and frozen product-code rules, including PADS and numeric delivery or packing codes.", "五位数字开头对照规则与冻结商品清单产生不同决定的原始行，含 PADS 及数字编码的运费、包装费。"),
    ("retention_comparison", "Identified positive lines, basket–SKU events, units and original-price values under observed, cancellation-only gross and net definitions. Each window has its own credit cutoff.", "按原销售、仅扣取消的毛额和扣所有匹配贷项的净额，对比已知客户的销售行、篮内商品事件、数量和原价金额；每个窗口单独设贷项截止点。"),
    ("retention_transitions", "Full, partial and unaffected source sales and basket–SKU events, split by credit composition. Each share uses all observed units of that grain in the window.", "按来源销售行和篮内商品两种粒度，分别统计全部冲销、部分冲销和未受影响的数量及贷项构成。各占比以对应窗口该粒度全部记录为分母。"),
    ("return_repurchase", "Full-workbook follow-up after first, last or individual return credits or the last matched credit. Compare a later observed sale with a later sale still net-positive at the end.", "全期资料中，分别从首次、末次、每次操作性退货或末次匹配贷项追查，同客同货其他订单其后是否有销售，以及是否有到期末仍为净正的销售。"),
]


def reversal_examples(allocations):
    """Trace each matched credit on selected source lines in the order it became visible."""
    matches = allocations.dropna(subset=["purchase_excel_row"]).sort_values(["credit_time", "credit_excel_row"]).copy()
    cancelled = matches["status"].eq("cancellation")
    matches["cancelled_units_here"] = matches["credited_units"].where(cancelled, 0)
    matches["gross_deduction_gbp_here"] = matches["cancelled_units_here"] * matches["purchase_unit_price"]
    matches["net_deduction_gbp_here"] = matches["credited_units"] * matches["purchase_unit_price"]
    cumulative = matches.groupby("purchase_excel_row")["cancelled_units_here"].cumsum()
    matches["gross_units_after_cancellations"] = matches["purchase_quantity"] - cumulative
    matches["purchase_units_after_credit"] = matches["balance_after"]
    cutoff = pd.Timestamp("2011-10-01")
    visible = matches.loc[matches["credit_time"].lt(cutoff)].groupby("purchase_excel_row")["credited_units"].sum()
    matches["purchase_units_at_oct1"] = (
        matches["purchase_quantity"] - matches["purchase_excel_row"].map(visible).fillna(0)
    ).where(matches["purchase_time"].lt(cutoff)).astype("Int64")

    selected = matches.loc[
        matches["credit_invoice_no"].isin(["C541433", "C581484"])
        | matches["credit_excel_row"].eq(222694), "purchase_excel_row"
    ]
    partial = matches.loc[cancelled & matches["credited_units"].lt(matches["balance_before"]), "purchase_excel_row"].head(1)
    example_rows = [67, 292, 5249, 34475, 321282, 370990, 371018, 373424]
    return matches.loc[matches["purchase_excel_row"].isin([*selected, *partial, *example_rows])].reset_index(drop=True)


def retention_comparison(balances):
    """Compare the three purchase definitions on the same identified, eligible sale lines."""
    rows = []
    event_keys = ["CustomerID", "InvoiceNo", "StockCode"]
    definitions = {
        "observed_positive_sales": balances["original_quantity"],
        "gross_after_cancellations": balances["original_quantity"] - balances["cancelled_units"],
        "net_after_all_matched_credits": balances["Quantity"],
    }
    for name, qty in definitions.items():
        kept = balances.loc[qty.gt(0)]
        rows.append({
            "definition": name, "sale_lines": len(kept), "basket_sku_events": len(kept[event_keys].drop_duplicates()),
            "baskets": len(kept[["CustomerID", "InvoiceNo"]].drop_duplicates()), "customers": kept["CustomerID"].nunique(),
            "units": int(qty.sum()), "value_at_sale_prices_gbp": round((qty * balances["UnitPrice"]).sum(), 6),
        })
    return pd.DataFrame(rows)


def retention_transitions(balances):
    """Count fully, partially and unaffected source lines and basket–SKU groups."""
    qty_cols = ["original_quantity", "cancelled_units", "returned_units", "other_matched_units", "Quantity"]
    events = balances.groupby(["CustomerID", "InvoiceNo", "StockCode"], sort=False)[qty_cols].sum()
    summaries = []
    for unit, data in [("source_sale_line", balances), ("basket_sku_event", events)]:
        components = data[["cancelled_units", "returned_units", "other_matched_units"]].gt(0)
        state = pd.Series("unaffected", index=data.index)
        state.loc[data["Quantity"].lt(data["original_quantity"])] = "partial"
        state.loc[data["Quantity"].eq(0)] = "full"
        composition = pd.Series("none", index=data.index)
        for column, label in [("cancelled_units", "cancellation"), ("returned_units", "return"), ("other_matched_units", "other")]:
            composition.loc[components[column]] = label
        composition.loc[components.sum(axis=1).gt(1)] = "mixed"
        summary = data.assign(state=state, credit_composition=composition).groupby(
            ["state", "credit_composition"], as_index=False
        ).agg(count=("Quantity", "size"), observed_units=("original_quantity", "sum"), retained_units=("Quantity", "sum"))
        summary.insert(0, "unit", unit)
        summary["all_observed"] = len(data)
        summary["share_of_all_observed"] = summary["count"] / len(data)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def compare_return_credit_rows(scenarios):
    """Reproduce the discussion's two statistics with the same per-credit definition."""
    rows = []
    for scenario, matches, ledger in scenarios:
        returns = matches.loc[matches["status"].eq("return")].copy()
        returns["size_relative_to_original"] = returns["credited_units"].eq(returns["purchase_quantity"]).map(
            {True: "full", False: "partial"}
        )
        # A sale on a different invoice is a follow-up if any of its lines is strictly later.
        sales = ledger.loc[ledger["ledger_type"].eq("sale") & ledger["CustomerID"].notna()]
        orders = sales.groupby(["CustomerID", "StockCode", "InvoiceNo"], as_index=False)["InvoiceDate"].max()
        pairs = returns.merge(orders, on=["CustomerID", "StockCode"])
        later = pairs["InvoiceNo"].ne(pairs["purchase_invoice_no"]) & pairs["InvoiceDate"].gt(pairs["credit_time"])
        followed = pairs.loc[later, "credit_excel_row"]
        returns["later_observed"] = returns["credit_excel_row"].isin(followed)
        result = returns.groupby("size_relative_to_original", as_index=False).agg(
            return_credit_rows=("credit_excel_row", "size"), distinct_linked_sale_lines=("purchase_excel_row", "nunique"),
            later_observed=("later_observed", "sum")
        )
        result["observed_rate"] = result["later_observed"] / result["return_credit_rows"]
        result["scenario"] = scenario
        result["scope"] = "individual_return_credit_rows"
        result["metric"] = "return_credit_relative_to_original_sale_quantity"
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def return_repurchase(balances, allocations):
    """Compare follow-up definitions at the credit-event and distinct original-sale level."""
    returns = allocations.loc[allocations["status"].eq("return")].copy()
    balances_by_row = balances.set_index("ExcelRow")
    matched = allocations.dropna(subset=["purchase_excel_row"]).sort_values(["credit_time", "credit_excel_row"])
    anchors = []
    for label, credits, keep in [
        ("first_operational_return", returns, "first"),
        ("last_operational_return", returns, "last"),
        ("each_operational_return", returns, None),
        ("last_matched_credit", matched, "last"),
    ]:
        ordered = credits.sort_values(["credit_time", "credit_excel_row"])
        if keep is not None:
            ordered = ordered.drop_duplicates("purchase_excel_row", keep=keep)
        anchored = ordered[["credit_excel_row", "credit_time", "purchase_excel_row",
                            "purchase_invoice_no", "CustomerID", "StockCode"]].copy()
        anchored["anchor"] = label
        anchored["final_quantity"] = anchored["purchase_excel_row"].map(balances_by_row["Quantity"])
        anchors.append(anchored)
    all_anchors = pd.concat(anchors, ignore_index=True).rename_axis("anchor_id").reset_index()
    # One candidate per invoice. The latest line time determines whether any line is strictly later.
    order_keys = ["CustomerID", "StockCode", "InvoiceNo"]
    sales = balances.groupby(order_keys, as_index=False)["InvoiceDate"].max().rename(columns={"InvoiceDate": "observed_time"})
    retained = balances.loc[balances["Quantity"].gt(0)].groupby(order_keys)["InvoiceDate"].max().rename("net_time")
    orders = sales.join(retained, on=order_keys)
    pairs = all_anchors.merge(orders, on=["CustomerID", "StockCode"], how="left")
    distinct = pairs["InvoiceNo"].ne(pairs["purchase_invoice_no"])
    pairs["later_observed"] = distinct & pairs["observed_time"].gt(pairs["credit_time"])
    pairs["later_net"] = distinct & pairs["net_time"].gt(pairs["credit_time"])
    outcomes = pairs.groupby("anchor_id")[["later_observed", "later_net"]].any()
    annotated = all_anchors.join(outcomes, on="anchor_id")
    annotated["source_outcome"] = annotated["final_quantity"].gt(0).map({True: "partial", False: "full"})
    summary = annotated.groupby(["anchor", "source_outcome"], as_index=False).agg(
        denominator=("anchor_id", "size"), distinct_source_sales=("purchase_excel_row", "nunique"),
        later_observed=("later_observed", "sum"), later_net=("later_net", "sum"),
    )
    summary["unit"] = summary["anchor"].eq("each_operational_return").map({True: "return_credit_row", False: "source_sale_line"})
    summary["observed_rate"] = summary["later_observed"] / summary["denominator"]
    summary["net_rate"] = summary["later_net"] / summary["denominator"]
    return summary


def reversal_worksheets(raw):
    evidence = compute_reversal_tables(raw)
    main = "frozen_balance_inclusive"
    summary = evidence["reconciliation"].loc[lambda x: x["scenario"].eq(main)].drop(columns="scenario")
    summary = summary.assign(section="ledger_and_matching")
    source = evidence["source_audit"].loc[lambda x: x["stock_policy"].eq("frozen")].drop(columns="stock_policy")
    source = source.rename(columns={"reason": "metric"}).assign(section="raw_first_rule")
    overview = pd.concat([summary, source], ignore_index=True)
    allocations = evidence["credit_allocations"]
    historical_ledger = build_signed_ledger(raw, stock_policy="five_digit")
    historical = match_reversals(historical_ledger, use_balance=False, include_same_time=True)
    per_credit = compare_return_credit_rows([
        ("five_digit_independent_inclusive", historical, historical_ledger),
        (main, allocations, evidence["signed_ledger_b"]),
    ])
    controls = pd.concat([evidence["benchmark"], per_credit], ignore_index=True)
    comparisons, transitions = [], []
    for window, cutoff in [("training_before_october", "2011-10-01"), ("whole_workbook", None)]:
        balances = purchase_line_balances(evidence["signed_ledger_b"], allocations, cutoff=cutoff)
        comparisons.append(retention_comparison(balances).assign(window=window))
        transitions.append(retention_transitions(balances).assign(window=window))
        if cutoff is None:
            repurchases = return_repurchase(balances, allocations)
    return {
        "ledger_summary": overview[["section", "metric", "customer_id_status", "rows", "units", "signed_amount_gbp"]],
        "credit_summary": evidence["region_summary"].loc[lambda x: x["scenario"].eq(main)].drop(columns="scenario"),
        "credit_allocations": allocations,
        "credit_controls": controls,
        "credit_control_rows": evidence["match_differences"],
        "reversal_examples": reversal_examples(allocations),
        "sku_policy_differences": evidence["source_differences"],
        "retention_comparison": pd.concat(comparisons, ignore_index=True),
        "retention_transitions": pd.concat(transitions, ignore_index=True),
        "return_repurchase": repurchases,
    }


def _sign(values: pd.Series) -> pd.Series:
    signs = pd.Series("zero", index=values.index)
    signs.loc[values.gt(0)] = "positive"
    signs.loc[values.lt(0)] = "negative"
    return signs.mask(values.isna(), "missing")


def build_tables(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lines = raw.assign(
        invoice_c=raw["InvoiceNo"].str.startswith("C", na=False),
        quantity_sign=_sign(raw["Quantity"]),
        price_sign=_sign(raw["UnitPrice"]),
        customer_present=raw["CustomerID"].notna(),
        amount_gbp=raw["Quantity"] * raw["UnitPrice"],
    )
    combinations = lines.groupby(
        ["invoice_c", "quantity_sign", "price_sign", "customer_present"],
        dropna=False, sort=True,
    ).agg(rows=("ExcelRow", "size"), invoices=("InvoiceNo", "nunique"),
          stock_codes=("StockCode", "nunique"), signed_gbp=("amount_gbp", "sum")).reset_index()

    exception = lines.invoice_c | lines.Quantity.le(0) | lines.UnitPrice.le(0)
    flagged = lines.loc[exception].copy()
    priced = lines.loc[~lines.invoice_c & lines.Quantity.gt(0) & lines.UnitPrice.gt(0)]
    invoice_counts = priced.groupby("InvoiceNo").size()
    flagged["same_invoice_positive_priced_rows"] = flagged.InvoiceNo.map(invoice_counts).fillna(0).astype(int)
    flagged = flagged.join(purchase_evidence(raw, flagged)).join(description_evidence(raw, flagged))
    views = {
        "C_invoices": flagged.loc[flagged.invoice_c],
        "Q_nonpositive": flagged.loc[flagged.Quantity.le(0)],
        "P_nonpositive": flagged.loc[flagged.UnitPrice.le(0)],
        "nonC_negative": flagged.loc[~flagged.invoice_c & flagged.Quantity.lt(0)],
    }
    scopes = {
        "any_exception": flagged,
        "C_invoices": views["C_invoices"],
        "nonC_negative": views["nonC_negative"],
        "zero_qty": flagged.loc[flagged.Quantity.eq(0)],
        "positive_qty_zero_price": flagged.loc[flagged.Quantity.gt(0) & flagged.UnitPrice.eq(0)],
        "negative_price": flagged.loc[flagged.UnitPrice.lt(0)],
    }
    history_rows = []
    note_tables = []
    for scope, group in scopes.items():
        history_rows.append({
            "cohort": scope, "rows": len(group), "distinct_invoices": group.InvoiceNo.nunique(),
            "singleton_invoice_rows": group.invoice_rows_full_sheet.eq(1).sum(),
            "prior_same_invoice_rows_matched": group.earlier_positive_invoice_rows.gt(0).sum(),
            "prior_same_invoice_sku_rows_matched": group.earlier_positive_invoice_sku_rows.gt(0).sum(),
            "prior_sku_rows_matched": group.earlier_positive_sku_rows.gt(0).sum(),
            "customer_sku_comparable_rows": group.earlier_positive_customer_sku_rows.notna().sum(),
            "prior_customer_sku_rows_matched": group.earlier_positive_customer_sku_rows.gt(0).sum(),
            "prior_stripped_invoice_sku_rows_matched": group.earlier_positive_stripped_invoice_sku_rows.gt(0).sum(),
            "prior_customer_sku_price_opposite_units_rows_matched":
                group.earlier_matching_customer_sku_price_opposite_quantity_rows.gt(0).sum(),
            "same_invoice_has_positive_priced_rows": group.same_invoice_positive_priced_rows.gt(0).sum(),
        })
        counts = group.groupby(["description_note_signals", "special_code_signal"],
                               dropna=False).size().rename("rows").reset_index()
        counts.insert(0, "scope", scope)
        note_tables.append(counts.sort_values("rows", ascending=False, kind="stable"))

    # Include surrounding positive lines as well as exceptions for named exhibits.
    examples = []
    for case, invoices, sku in [
        ("same-customer opposite entries", ["541431", "C541433"], "23166"),
        ("discount credit", ["C536379"], "D"),
        ("zero recorded unit price", ["539722"], "22423"),
        ("same invoice and SKU at two prices", ["550188"], "22636"),
        ("bad-debt entries", ["A563185", "A563186", "A563187"], "B"),
        ("explicit credit note and later adjustments", ["C550456", "554550", "554551", "554553"], "47566B"),
        ("stock-code discrepancy in description", ["560039"], "20713"),
        ("damage wording in description", ["554143"], "79030G"),
        ("count/check wording in description", ["573482"], "21181"),
    ]:
        sample = lines.loc[lines.InvoiceNo.isin(invoices) & lines.StockCode.eq(sku)].copy()
        sample.insert(0, "case", case)
        examples.append(sample)
    guide = [
        ("sign_combinations", "Disjoint cells by C prefix, quantity sign, price sign and customer ID presence. Counts cover every raw line.", "按 C 前缀、数量及价格符号、客户编号有无交叉分组。各组合互斥，合计覆盖原表。"),
        ("missingness", "Missing fields, affected invoice counts, and intersection with missing customer IDs on the original sheet.", "原表各字段缺失行数、涉及的不同发票数及与缺客户编号的交集。"),
        ("customer_id_invoices", "Full invoice lookup for each invoice with a missing customer ID, including counts of known IDs elsewhere on the invoice.", "逐张回查含缺失客户编号的发票，统计同票其他行出现的不同已知客户编号。"),
        ("filter_attribution", "Attribute each row in a scope to the first applicable purchase filter or to retained. Scopes overlap. This is a count simulation.", "对每个筛选范围的行，统计最先命中的购买过滤规则或保留数。范围可重叠，仅作计数模拟。"),
        ("C_invoices", "All lines on C-prefixed invoices, including discounts and other non-merchandise lines.", "全部 C 前缀明细，含折扣和其他非商品记录。"),
        ("Q_nonpositive", "All lines with Quantity ≤ 0, including C invoices.", "数量 ≤ 0 的全部明细，包括 C 前缀记录。"),
        ("P_nonpositive", "All lines with UnitPrice ≤ 0. Filter price_sign to separate zero and negative recorded prices.", "单价 ≤ 0 的全部明细。用 price_sign 区分零价与负价。"),
        ("nonC_negative", "All non-C negative quantities. ExcelRow counts the header as row 1. Original descriptions and IDs are preserved.", "全部非 C 负数量明细。ExcelRow 对应原表行号，表头为第 1 行。保留原描述与编号。"),
        ("history_summary", "Counts of flagged rows with at least one match. Earlier means strictly before the flagged timestamp. Positive means Quantity > 0. Unknown customer keys are blank in row-level evidence.", "汇总至少有一条匹配的明细数。历史匹配严格早于当前时间。正数量为 Quantity > 0。明细中客户匹配留空表示无法比较。"),
        ("note_summary", "Counts by exact combination of note/code clues in each scope. Unknown means no recognized wording. Scopes overlap.", "按备注与特殊编码线索的组合计数。unknown 表示未识别到关键词。不同筛选范围之间有重叠。"),
        ("case_examples", "Selected original entries in chronological context. Signed amount is quantity times unit price in GBP.", "从原表选取的交易及上下文。带符号金额为数量乘单价，单位英镑。"),
        ("invoice_references", "Invoice-shaped tokens in the Description and exact matches in the workbook, with earlier-line counts.", "备注里的发票号形式文本，在原表中精确回查，并统计较早的明细。"),
        ("sku_names", "Exact stored labels and codes for a name lookup after trimming surrounding whitespace. Product identity remains unverified.", "按名称去除首尾空白查找，展示原始名称和编码。商品是否完全相同仍待商品主数据核实。"),
        ("amount_by_record_type", "Signed line arithmetic by exception status and customer ID availability. Actual payments and refunds are unobserved.", "按标记与有无客户编号汇总带符号行金额。原表没有付款到账或退款流水。"),
        ("customer_impact_summary", "Identified customers grouped by whether they have regular, exceptional or both kinds of lines.", "按已知客户拥有常规行、异常行或两类兼有进行汇总。"),
        ("customer_impact", "Per identified customer, compare distinct invoices, recency and monetary totals under both definitions. Full-window descriptive analysis.", "对每一位已知客户，比较两种口径的不同发票数、距最近记录天数及金额。仅作全期描述。"),
    ]
    guide.extend(REVERSAL_GUIDE)
    return {
        "guide": pd.DataFrame(guide, columns=["Sheet", "What it contains", "内容与口径"]),
        "sign_combinations": combinations,
        **missingness_tables(raw),
        "history_summary": pd.DataFrame(history_rows),
        "note_summary": pd.concat(note_tables, ignore_index=True),
        "case_examples": pd.concat(examples, ignore_index=True),
        **views,
        "invoice_references": mentioned_invoice_trace(raw, flagged),
        "sku_names": sku_name_evidence(raw),
        **impact_tables(raw, exception),
        **reversal_worksheets(raw),
    }


def write_workbook(tables: dict[str, pd.DataFrame], output: Path) -> None:
    """Write filterable sheets, preserving original descriptions as literal text."""
    path = Path(output)
    if path.suffix.lower() != ".xlsx":
        raise ValueError("Choose an .xlsx output file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as work:
        temp = Path(work) / path.name
        with pd.ExcelWriter(temp, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
            for name, frame in tables.items():
                frame.to_excel(writer, sheet_name=name, index=False)
                sheet = writer.sheets[name]
                sheet.freeze_panes = "B2"
                sheet.auto_filter.ref = sheet.dimensions
                for number, col in enumerate(frame.columns, start=1):
                    width = 48 if col == "Description" else max(17, len(col) + 3)
                    if name == "guide":
                        width = 32 if number == 1 else 105
                    sheet.column_dimensions[get_column_letter(number)].width = width
                # A leading '=' in an original description is data, not an Excel formula.
                for row in sheet.iter_rows(min_row=2):
                    for cell in row:
                        if cell.data_type == "f":
                            cell.data_type = "s"
        temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=Path("data/raw/Online Retail.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("results/transaction_exceptions.xlsx"))
    args = parser.parse_args()
    if args.output.resolve() == args.workbook.resolve():
        parser.error("Choose an output path separate from the source workbook.")
    tables = build_tables(read_workbook(args.workbook))
    write_workbook(tables, args.output)
    print(tables["sign_combinations"].to_string(index=False))
    print(f"Saved {len(tables)} sheets to {args.output}")


if __name__ == "__main__":
    main()
