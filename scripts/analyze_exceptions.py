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
from src.data.reversals import compute_reversal_tables


REVERSAL_GUIDE = [
    ("ledger_summary", "Frozen merchandise policy. Signed sales and credits retain recorded prices. The gross purchase view deducts only allocated cancellations at their sale prices. Source exclusions are first-match counts.", "冻结商品规则。带符号销售与贷项取各行记录价格；有效购买毛额只按原销售单价扣减已匹配的取消数量。来源筛选按最先命中的规则归因。"),
    ("credit_summary", "Canonical merchandise credits by destination region and operational status. Amounts use recorded credit prices. Time thresholds are regional assumptions.", "按贷项目的地区与操作分类汇总实际规则下的商品贷项，金额取贷项单价。时限是项目对地区的假设。"),
    ("credit_allocations", "Every eligible merchandise credit with matched sale row, time lag, region, threshold and allocation balance. An unmatched row has no allocated sale.", "全部符合条件的商品贷项，列出匹配销售行、时间差、地区、时限及分配余额；未匹配的贷项无对应销售行。"),
    ("credit_controls", "Compare code shape versus frozen SKU rules, strict versus inclusive time, and independent versus balance-constrained matches against the supplied study benchmarks.", "以讨论稿基准对照商品码外形与冻结规则、严格早于与允许同时间，以及独立匹配与数量余额约束的结果。"),
    ("credit_control_rows", "Credit rows whose inclusion, matched sale or class changes between scenarios. Use the source row numbers to reproduce a difference.", "对照情景之间在纳入状态、对应销售行或操作类别上有变化的贷项，可用原始行号复核。"),
    ("reversal_examples", "Selected credit and sale links including two large reversals, unequal prices, partial cancellations and returns. Track both reversible units and effective purchase units.", "追踪两笔大额撤销、价格不相等、部分取消及退货示例的贷项与销售。分别核对可匹配余额和有效购买数量。"),
    ("sku_policy_differences", "Source rows with different decisions under the five-leading-digit control and frozen product-code rules, including PADS and numeric delivery or packing codes.", "五位数字开头对照规则与冻结商品清单产生不同决定的原始行，含 PADS 及数字编码的运费、包装费。"),
]


def reversal_examples(allocations):
    """Trace all credits on example sales and distinguish allocation from purchase units."""
    matches = allocations.dropna(subset=["purchase_excel_row"]).sort_values(["credit_time", "credit_excel_row"]).copy()
    cancelled = matches["status"].eq("cancellation")
    matches["cancelled_units_here"] = matches["credited_units"].where(cancelled, 0)
    matches["gross_deduction_gbp_here"] = matches["cancelled_units_here"] * matches["purchase_unit_price"]
    cumulative = matches.groupby("purchase_excel_row")["cancelled_units_here"].cumsum()
    matches["purchase_units_after_credit"] = matches["purchase_quantity"] - cumulative
    cutoff = pd.Timestamp("2011-10-01")
    before_cutoff = matches["credit_time"].lt(cutoff)
    cancelled_by_cutoff = matches.loc[before_cutoff].groupby("purchase_excel_row")["cancelled_units_here"].sum()
    matches["purchase_units_at_oct1"] = (
        matches["purchase_quantity"] - matches["purchase_excel_row"].map(cancelled_by_cutoff).fillna(0)
    ).where(matches["purchase_time"].lt(cutoff)).astype("Int64")

    selected = matches.loc[
        matches["credit_invoice_no"].isin(["C541433", "C581484"])
        | matches["credit_excel_row"].eq(222694), "purchase_excel_row"
    ]
    partial = matches.loc[cancelled & matches["credited_units"].lt(matches["balance_before"]), "purchase_excel_row"].head(1)
    returned = matches.loc[matches["status"].eq("return"), "purchase_excel_row"].head(1)
    return matches.loc[matches["purchase_excel_row"].isin(pd.concat([selected, partial, returned]))].reset_index(drop=True)


def reversal_worksheets(raw):
    evidence = compute_reversal_tables(raw)
    main = "frozen_balance_inclusive"
    summary = evidence["reconciliation"].loc[lambda x: x["scenario"].eq(main)].drop(columns="scenario")
    summary = summary.assign(section="ledger_and_matching")
    source = evidence["source_audit"].loc[lambda x: x["stock_policy"].eq("frozen")].drop(columns="stock_policy")
    source = source.rename(columns={"reason": "metric"}).assign(section="raw_first_rule")
    overview = pd.concat([summary, source], ignore_index=True)
    allocations = evidence["credit_allocations"]
    return {
        "ledger_summary": overview[["section", "metric", "customer_id_status", "rows", "units", "signed_amount_gbp"]],
        "credit_summary": evidence["region_summary"].loc[lambda x: x["scenario"].eq(main)].drop(columns="scenario"),
        "credit_allocations": allocations,
        "credit_controls": evidence["benchmark"],
        "credit_control_rows": evidence["match_differences"],
        "reversal_examples": reversal_examples(allocations),
        "sku_policy_differences": evidence["source_differences"],
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
