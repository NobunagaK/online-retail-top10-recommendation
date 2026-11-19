"""Signed merchandise ledgers and time-based credit-note evidence for UCI Online Retail."""

import numpy as np
import pandas as pd

from src.data.clean import RAW_COLUMNS
from src.data.merchandise import merchandise_rules


EUROPE_COUNTRIES = frozenset({
    "EIRE", "France", "Germany", "Netherlands", "Belgium", "Spain", "Switzerland",
    "Portugal", "Channel Islands", "Italy", "Norway", "Sweden", "Denmark",
    "Finland", "Austria", "Poland", "Cyprus", "Greece", "Czech Republic",
    "Malta", "Iceland", "European Community", "Lithuania",
})

# The last scenario is the project policy. The others isolate catalogue,
# same-timestamp, and sale-balance choices for comparison with the supplied study.
SCENARIOS = {
    "five_digit_independent_strict": ("five_digit", False, False),
    "five_digit_independent_inclusive": ("five_digit", False, True),
    "frozen_independent_inclusive": ("frozen", False, True),
    "frozen_balance_strict": ("frozen", True, False),
    "frozen_balance_inclusive": ("frozen", True, True),
}
CREDIT_STATUSES = ("cancellation", "return", "unmatched", "anonymous", "unknown_region")


def _ledger_decisions(raw, stock_policy):
    """Apply mutually exclusive first-match reasons in original worksheet order."""
    if raw["ExcelRow"].isna().any() or not raw["ExcelRow"].is_unique:
        raise ValueError("ExcelRow must be present and unique in the raw worksheet.")
    frame = raw.sort_values("ExcelRow").reset_index(drop=True)
    code = frame["StockCode"].astype("string")
    if stock_policy == "frozen":
        rules = merchandise_rules()
        merchandise = ~code.isin(rules.loc[rules["action"].eq("exclude"), "StockCode"])
    elif stock_policy == "five_digit":
        merchandise = code.str.match(r"^[0-9]{5}", na=False)
    else:
        raise ValueError("stock_policy must be 'frozen' or 'five_digit'.")
    invoice = frame["InvoiceNo"].astype("string")
    credit = invoice.str.startswith("C", na=False)
    quantity = frame["Quantity"]
    price = frame["UnitPrice"]
    checks = [
        ("exact_duplicate", frame.duplicated(RAW_COLUMNS)),
        ("missing_required_field", frame[["InvoiceNo", "StockCode", "InvoiceDate", "Quantity", "UnitPrice"]].isna().any(axis=1)),
        ("bad_debt_document", invoice.str.startswith("A", na=False)),
        ("non_c_negative_quantity", ~credit & quantity.lt(0)),
        ("zero_quantity", quantity.eq(0)),
        ("negative_unit_price", price.lt(0)),
        ("zero_unit_price", price.eq(0)),
        ("excluded_stock_code", ~merchandise),
        ("unexpected_positive_c_line", credit & quantity.gt(0)),
        ("credit", credit & quantity.lt(0)),
    ]
    decisions = np.select(
        [test.to_numpy(dtype=bool, na_value=False) for _, test in checks],
        [reason for reason, _ in checks],
        default="sale",
    )
    return frame, pd.Series(decisions, index=frame.index, name="decision")


def _signed_lines(frame, decisions):
    eligible = decisions.isin(["sale", "credit"])
    ledger = frame.loc[eligible].copy()
    ledger["ledger_type"] = decisions.loc[eligible]
    # Guest is a display value only. CustomerID stays missing for all joins.
    ledger["customer_display"] = ledger["CustomerID"].astype("string").fillna("Guest/Unknown")
    ledger["signed_amount_gbp"] = ledger["Quantity"] * ledger["UnitPrice"]
    return ledger.reset_index(drop=True)


def build_signed_ledger(raw: pd.DataFrame, stock_policy="frozen") -> pd.DataFrame:
    """Build store-level signed view A. View B selects its identified customers."""
    frame, decisions = _ledger_decisions(raw, stock_policy)
    return _signed_lines(frame, decisions)


def _region(country, uk_hours=72):
    """Map the C document's destination to its continuous elapsed-time threshold."""
    if pd.isna(country) or country in ("", "Unspecified"):
        return "unknown", None
    if country == "United Kingdom":
        return "UK", uk_hours
    if country in EUROPE_COUNTRIES:
        return "Europe", 120
    return "ROW", 240


def match_reversals(
    ledger: pd.DataFrame, *, use_balance=True, include_same_time=True, uk_hours=72,
) -> pd.DataFrame:
    """Match each C once to the nearest eligible single positive line.

    Work in credit timestamp and worksheet-row order. A sale must have the
    same original customer and SKU, with enough reversible units on that
    one row. Price is evidence but is not a matching condition. At the nearest
    eligible purchase time, ascending ExcelRow prefers rows before the C and
    then the smallest row. Inclusive matching allows an equal-time row appearing
    later on the sheet. Every match reserves units on its source line. The
    regional time threshold labels the match for operational reporting. A
    net-retained snapshot deducts linked units once the credit is observable,
    including returns and matches with an unknown destination. For sensitivity
    checks, uk_hours changes the UK reporting threshold alone (whole hours).
    """
    if not isinstance(uk_hours, (int, np.integer)) or uk_hours < 0:
        raise ValueError("uk_hours must be a nonnegative whole number of hours.")
    credits = ledger.loc[ledger["ledger_type"].eq("credit")].sort_values(["InvoiceDate", "ExcelRow"])
    keys = ["CustomerID", "StockCode"]
    relevant = credits[keys].dropna().drop_duplicates()
    sales = ledger.loc[ledger["ledger_type"].eq("sale")].merge(relevant, on=keys, validate="many_to_one")
    # Newest eligible time first; the lower original row wins equal-time ties.
    sales = sales.sort_values(["InvoiceDate", "ExcelRow"], ascending=[False, True])
    groups = {}
    for sale in sales.itertuples(index=False):
        groups.setdefault((sale.CustomerID, sale.StockCode), []).append(sale)
    remaining = {}
    records = []
    for credit in credits.itertuples(index=False):
        customer, sku = credit.CustomerID, credit.StockCode
        units = -credit.Quantity
        region, hours = _region(credit.Country, uk_hours=uk_hours)
        record = {
            "credit_excel_row": credit.ExcelRow, "credit_invoice_no": credit.InvoiceNo,
            "credit_time": credit.InvoiceDate, "CustomerID": customer, "StockCode": sku,
            "Country": credit.Country, "region": region, "threshold_hours": hours,
            "credited_units": units, "credit_unit_price": credit.UnitPrice,
            "credit_amount_gbp": credit.signed_amount_gbp,
            "purchase_excel_row": None, "purchase_invoice_no": None, "purchase_time": None,
            "purchase_unit_price": None, "purchase_quantity": None, "balance_before": None,
            "balance_after": None, "lag_minutes": None, "lag_days": None,
            "same_time": False, "later_excel_row_same_time": False,
            "status": "anonymous" if pd.isna(customer) else "unmatched",
        }
        if pd.notna(customer):
            for sale in groups.get((customer, sku), ()):
                lag = credit.InvoiceDate - sale.InvoiceDate
                if lag < pd.Timedelta(0) or (lag == pd.Timedelta(0) and not include_same_time):
                    continue
                available = remaining.get(sale.ExcelRow, sale.Quantity)
                if available < units:
                    continue
                after = available - units
                if use_balance:
                    remaining[sale.ExcelRow] = after
                status = "unknown_region"
                if hours is not None:
                    status = "cancellation" if lag <= pd.Timedelta(hours=hours) else "return"
                record.update({
                    "purchase_excel_row": sale.ExcelRow, "purchase_invoice_no": sale.InvoiceNo,
                    "purchase_time": sale.InvoiceDate, "purchase_unit_price": sale.UnitPrice,
                    "purchase_quantity": sale.Quantity, "balance_before": available,
                    "balance_after": after, "lag_minutes": lag.total_seconds() / 60,
                    "lag_days": lag / pd.Timedelta(days=1), "same_time": lag == pd.Timedelta(0),
                    "later_excel_row_same_time": lag == pd.Timedelta(0) and sale.ExcelRow > credit.ExcelRow,
                    "status": status,
                })
                break
        records.append(record)
    columns = [
        "credit_excel_row", "credit_invoice_no", "credit_time", "CustomerID", "StockCode",
        "Country", "region", "threshold_hours", "credited_units", "credit_unit_price",
        "credit_amount_gbp", "purchase_excel_row", "purchase_invoice_no", "purchase_time",
        "purchase_unit_price", "purchase_quantity", "balance_before", "balance_after",
        "lag_minutes", "lag_days", "same_time", "later_excel_row_same_time", "status",
    ]
    result = pd.DataFrame.from_records(records, columns=columns)
    for name in ("credit_time", "purchase_time"):
        result[name] = pd.to_datetime(result[name])
    for name in ("credit_excel_row", "threshold_hours", "credited_units", "purchase_excel_row",
                 "purchase_quantity", "balance_before", "balance_after"):
        result[name] = result[name].astype("Int64")
    for name in ("credit_amount_gbp", "credit_unit_price", "purchase_unit_price", "lag_minutes", "lag_days"):
        result[name] = result[name].astype(float)
    for name in ("same_time", "later_excel_row_same_time"):
        result[name] = result[name].astype(bool)
    return result


def _scenario_summary(scenario, ledger, matches, use_balance):
    """Reconcile signed documents and sale-price valuations over the full period.

    Gross after cancellations measures sales before returns. Net-retained units
    deduct every allocated credit from its source sale. Signed view A/B instead
    sums each document at its own recorded price, including unlinked credits.
    """
    rows = []

    def record(metric, frame, units="Quantity", money="signed_amount_gbp"):
        rows.append({
            "scenario": scenario, "metric": metric, "rows": len(frame),
            "units": frame[units].sum(), "signed_amount_gbp": frame[money].sum(),
        })

    known = ledger.loc[ledger["CustomerID"].notna()]
    sales = ledger.loc[ledger["ledger_type"].eq("sale")]
    record("view_a", ledger)
    record("view_b", known)
    record("guest", ledger.loc[ledger["CustomerID"].isna()])
    record("sale", sales)
    record("credit", ledger.loc[ledger["ledger_type"].eq("credit")])
    record("known_observed_sales", sales.loc[sales["CustomerID"].notna()])
    statuses = matches.assign(signed_units=-matches["credited_units"])
    for status in CREDIT_STATUSES:
        selected = statuses.loc[statuses["status"].eq(status)]
        record(status, selected, units="signed_units", money="credit_amount_gbp")
    if use_balance:
        linked = matches.loc[matches["purchase_excel_row"].notna()]
        deductions = {
            "gross_after_cancellations": linked.loc[linked["status"].eq("cancellation")],
            "net_retained_at_sale_prices": linked,
        }
        for metric, credits in deductions.items():
            allocated = credits.groupby("purchase_excel_row")["credited_units"].sum()
            quantity = sales["Quantity"] - sales["ExcelRow"].map(allocated).fillna(0)
            goods = sales.loc[quantity.gt(0)].assign(effective_units=quantity)
            goods["effective_amount_gbp"] = goods["effective_units"] * goods["UnitPrice"]
            record(metric, goods, "effective_units", "effective_amount_gbp")
            record("known_" + metric, goods.loc[goods["CustomerID"].notna()],
                   "effective_units", "effective_amount_gbp")
    return pd.DataFrame(rows)


def _compare_matches(before_name, before, after_name, after):
    """Trace lines whose inclusion, selected source, classification or balance changed."""
    detail = ["status", "purchase_excel_row", "purchase_invoice_no", "purchase_time",
              "purchase_unit_price", "lag_minutes", "balance_before", "balance_after",
              "same_time", "later_excel_row_same_time"]
    ids = ["credit_excel_row", "credit_invoice_no", "credit_time", "CustomerID", "StockCode", "region",
           "credited_units", "credit_amount_gbp"]
    identity = pd.concat([before[ids], after[ids]], ignore_index=True).drop_duplicates("credit_excel_row")
    key = "credit_excel_row"
    side = before.set_index(key)[detail].add_prefix("before_").join(
        after.set_index(key)[detail].add_prefix("after_"), how="outer",
    )
    changes = {}
    for field in ["status", "purchase_excel_row", "balance_before", "balance_after"]:
        old, new = side["before_" + field], side["after_" + field]
        changes[field] = (old.isna() != new.isna()) | old.ne(new).fillna(False)
    side["status_changed"] = changes["status"]
    side["purchase_changed"] = changes["purchase_excel_row"]
    side["balance_changed"] = changes["balance_before"] | changes["balance_after"]
    relevant = side["status_changed"] | side["purchase_changed"] | side["balance_changed"]
    result = identity.merge(side.loc[relevant], on=key, how="inner")
    result.insert(0, "after_scenario", after_name)
    result.insert(0, "before_scenario", before_name)
    return result.sort_values(key, ignore_index=True)


def _reference_benchmark(summary, regional):
    """Compare supplied figures with measured outputs without fitting the policy to them."""
    targets = {
        "view_a": (531172, 9770919.71), "view_b": (399656, 8265476.93),
        "credit": (8668, None), "cancellation": (1481, -312072.49),
        "return": (5913, -134674.78), "unmatched": (1112, -25003.44),
        "anonymous": (162, -4150.45),
    }
    rows = []

    def add(scenario, scope, metric, expected_rows, expected_amount, observed_rows, observed_amount):
        rows.append({
            "scenario": scenario, "scope": scope, "metric": metric, "reported_rows": expected_rows,
            "observed_rows": observed_rows, "delta_rows": observed_rows - expected_rows,
            "reported_amount_gbp": expected_amount, "observed_amount_gbp": observed_amount,
            "delta_amount_gbp": None if expected_amount is None else observed_amount - expected_amount,
        })

    for row in summary.itertuples(index=False):
        if row.metric in targets:
            count, amount = targets[row.metric]
            add(row.scenario, "all", row.metric, count, amount, row.rows, row.signed_amount_gbp)
    # The supplied regional 'unmatched' totals include anonymous credits. The
    # main ledger retains the separate unmatched, anonymous and unknown-region states.
    region_targets = {"UK": (1111, 5084, 1134), "Europe": (194, 788, 133), "ROW": (176, 41, 7)}
    for scenario in SCENARIOS:
        for region, counts in region_targets.items():
            group = regional.loc[regional["scenario"].eq(scenario) & regional["region"].eq(region)]
            for metric, statuses, reported in zip(
                ["cancellation", "return", "unmatched_including_anonymous"],
                [["cancellation"], ["return"], ["unmatched", "anonymous"]], counts,
            ):
                selected = group.loc[group["status"].isin(statuses)]
                add(scenario, region, metric, reported, None,
                    selected["rows"].sum(), selected["signed_amount_gbp"].sum())
    return pd.DataFrame(rows)


def compute_reversal_tables(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Classify the original eight-field worksheet with its stable ExcelRow locators.

    Return canonical signed A/B, canonical credit allocations, a source audit and
    the benchmark controls. Raw worksheet counts, independently classified credits
    and balance-reserved credits have different denominators. Every comparison
    below preserves its own source policy, lag rule and allocation convention.
    """
    sources, decisions, audits = {}, {}, []
    for policy in ("five_digit", "frozen"):
        frame, decision = _ledger_decisions(raw, policy)
        decisions[policy] = decision
        sources[policy] = _signed_lines(frame, decision)
        audit = pd.DataFrame({
            "reason": decision, "customer_id_status": np.where(frame["CustomerID"].isna(), "anonymous", "known"),
            "signed_amount_gbp": frame["Quantity"] * frame["UnitPrice"],
        })
        audit = audit.groupby(["reason", "customer_id_status"], as_index=False).agg(
            rows=("reason", "size"), signed_amount_gbp=("signed_amount_gbp", "sum"),
        )
        audit.insert(0, "stock_policy", policy)
        audits.append(audit)

    difference = decisions["five_digit"].ne(decisions["frozen"])
    detail = frame.loc[difference].copy()
    detail["five_digit_decision"] = decisions["five_digit"].loc[difference]
    detail["frozen_decision"] = decisions["frozen"].loc[difference]
    detail["signed_amount_gbp"] = detail["Quantity"] * detail["UnitPrice"]

    matches, summaries, regions = {}, [], []
    for name, (policy, balance, same_time) in SCENARIOS.items():
        match = match_reversals(sources[policy], use_balance=balance, include_same_time=same_time)
        matches[name] = match
        summaries.append(_scenario_summary(name, sources[policy], match, balance))
        region = match.groupby(["region", "status"], as_index=False).agg(
            rows=("status", "size"), signed_amount_gbp=("credit_amount_gbp", "sum"),
        )
        region.insert(0, "scenario", name)
        regions.append(region)
    comparisons = [
        ("five_digit_independent_strict", "five_digit_independent_inclusive"),
        ("five_digit_independent_inclusive", "frozen_independent_inclusive"),
        ("frozen_independent_inclusive", "frozen_balance_inclusive"),
        ("frozen_balance_strict", "frozen_balance_inclusive"),
    ]
    changes = [
        _compare_matches(before, matches[before], after, matches[after]) for before, after in comparisons
    ]
    summary = pd.concat(summaries, ignore_index=True)
    region_summary = pd.concat(regions, ignore_index=True)
    signed = sources["frozen"]
    tables = {
        "signed_ledger_a": signed,
        "signed_ledger_b": signed.loc[signed["CustomerID"].notna()].reset_index(drop=True),
        "credit_allocations": matches["frozen_balance_inclusive"],
        "source_audit": pd.concat(audits, ignore_index=True),
        "source_differences": detail.reset_index(drop=True),
        "reconciliation": summary,
        "region_summary": region_summary,
        "match_differences": pd.concat(changes, ignore_index=True),
        "benchmark": _reference_benchmark(summary, region_summary),
    }
    # Keep sub-penny source prices. Remove floating-point summation noise from
    # published diagnostics without modifying source prices or the signed ledgers.
    for name in ["source_audit", "reconciliation", "region_summary", "benchmark"]:
        money = [column for column in tables[name] if column.endswith("_gbp")]
        rounded = tables[name][money].round(6)
        # A rounded zero has a single CSV representation across NumPy versions.
        tables[name][money] = rounded.mask(rounded.eq(0), 0.0)
    return tables
