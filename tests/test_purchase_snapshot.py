import json

import pandas as pd

from scripts.prepare_data import BASELINE_PATH, build_cleaning_outputs, save_cleaning_results
from src.data.baskets import build_basket_events
from src.data.purchase_snapshot import net_retained_labels, purchase_line_balances, purchase_snapshot
from src.data.reversals import build_signed_ledger, match_reversals


def source_rows():
    columns = ["ExcelRow", "InvoiceNo", "StockCode", "Quantity", "UnitPrice", "InvoiceDate", "CustomerID", "Country"]
    # 001 and 002 have the same invoice number and different completion times.
    rows = [
        (2, "201", "10001", 10, 2, "2011-09-30 08:00", "001", "United Kingdom"),
        (3, "201", "10002", 1, 8, "2011-09-30 23:45", "001", "United Kingdom"),
        (4, "C204", "10001", -3, 5, "2011-09-30 16:00", "001", "United Kingdom"),
        (5, "C205", "10002", -1, 8, "2011-10-01 00:00", "001", "United Kingdom"),
        (6, "C206", "10001", -1, 2, "2011-10-10 00:00", "001", "United Kingdom"),
        (7, "C207", "10001", -1, 2, "2011-10-10 00:01", None, "United Kingdom"),
        (8, "C208", "10001", -1, 2, "2011-10-10 00:02", "001", "Unspecified"),
        (9, "201", "10001", 2, 8, "2011-10-01 00:01", "002", "United Kingdom"),
        (10, "201", "10003", 4, 1, "2011-09-30 19:00", "002", "United Kingdom"),
        (11, "300", "POST", 1, 5, "2011-09-03 11:00", None, "United Kingdom"),
    ]
    raw = pd.DataFrame(rows, columns=columns).assign(Description="example")
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    return raw


def test_snapshot_all_matched_credit_units_cutoff_partial_units_and_original_basket_time():
    raw = source_rows()
    ledger = build_signed_ledger(raw)
    credits = match_reversals(ledger)

    early, _ = purchase_snapshot(ledger, credits, cutoff="2011-09-30 20:00")
    assert early.empty  # Neither whole invoice has closed.
    before, original_events = purchase_snapshot(ledger, credits, cutoff="2011-10-01")
    assert before.ExcelRow.tolist() == [2, 3]
    assert before.Quantity.tolist() == [7, 1]  # Three units, not £15, removed at original sale price.
    assert before.line_amount_gbp.tolist() == [14, 8]
    assert before.returned_units.tolist() == [0, 0]
    assert original_events.source_lines.sum() == 2
    without_future = raw.loc[~raw.ExcelRow.isin([5, 6, 7, 8])]
    earlier_ledger = build_signed_ledger(without_future)
    replay, replay_events = purchase_snapshot(earlier_ledger, match_reversals(earlier_ledger), cutoff="2011-10-01")
    pd.testing.assert_frame_equal(before, replay)
    pd.testing.assert_frame_equal(original_events, replay_events)

    after, final_events = purchase_snapshot(ledger, credits)
    assert after.ExcelRow.tolist() == [2, 9, 10]
    assert after.loc[after.ExcelRow.eq(2), ["original_quantity", "cancelled_units", "returned_units", "other_matched_units", "matched_credit_units", "Quantity"]].iloc[0].tolist() == [10, 3, 1, 1, 5, 5]
    assert final_events.loc[final_events.CustomerID.eq("001"), "basket_time"].item() == pd.Timestamp("2011-09-30 23:45")
    pd.testing.assert_frame_equal(final_events, build_basket_events(after))


def test_d1_frames_reconcile_individual_outcomes_and_publish_distinct_snapshot_manifests(tmp_path):
    raw = source_rows()
    frames, manifest = build_cleaning_outputs(raw)
    decisions = frames["data/processed/row_decisions.csv.gz"].set_index("ExcelRow")
    assert decisions.loc[2, "first_rule"] == "sale_partially_credited"
    assert decisions.loc[2, "returned_units"] == 1
    assert decisions.loc[2, "matched_credit_composition"] == "mixed"
    assert decisions.loc[3, "matched_credit_composition"] == "cancellation"
    assert decisions.loc[2, "retained_quantity"] == 5
    assert decisions.loc[8, "matched_purchase_excel_row"] == 2
    assert decisions.loc[3, "first_rule"] == "sale_fully_credited"
    assert decisions.loc[4, "first_rule"] == "credit_cancellation"
    assert decisions.loc[6, "first_rule"] == "credit_return"
    assert decisions.loc[7, "first_rule"] == "credit_anonymous"
    assert decisions.loc[8, "first_rule"] == "credit_unknown_region"
    assert decisions.loc[7, "in_signed_ledger_a"] and not decisions.loc[7, "in_signed_ledger_b"]
    steps = frames["results/cleaning_steps.csv"]
    assert steps.decision_rows.sum() == len(raw)
    assert steps.iloc[-1].rows_after == manifest["counts"]["paid_merchandise_lines"] == 3
    assert steps.rows_removed.sum() == len(raw) - 3
    assert frames["data/processed/training_purchase_lines.csv.gz"].Quantity.sum() == 8
    save_cleaning_results(tmp_path, tmp_path / BASELINE_PATH, frames, manifest)
    saved = json.loads((tmp_path / "results/data_manifest.json").read_text())
    assert saved["training_snapshot"]["end_exclusive"] == "2011-10-01"
    assert saved["recommendation_quantity_policy"] == "net_retained_all_matched_credits_v1"
    assert saved["counts"]["matched_credit_units"] == 6
    assert saved["training_snapshot"]["matched_credit_units"] == 3
    for name, frame in frames.items():
        assert (tmp_path / name).is_file()
        assert saved["outputs"][name]["rows"] == len(frame)


def test_relabelling_matched_credits_changes_audit_fields_without_changing_units_events_or_labels():
    ledger = build_signed_ledger(source_rows())
    credits = match_reversals(ledger)
    switched = credits.copy()
    linked = switched.purchase_excel_row.notna()
    switched.loc[linked, "status"] = switched.loc[linked, "status"].map({
        "cancellation": "return", "return": "unknown_region", "unknown_region": "cancellation",
    })
    for end in ("2011-10-01", "2011-11-01"):
        before = purchase_line_balances(ledger, credits, cutoff=end)
        after = purchase_line_balances(ledger, switched, cutoff=end)
        assert not before["matched_credit_composition"].equals(after["matched_credit_composition"])
        columns = ["ExcelRow", "matched_credit_units", "Quantity", "line_amount_gbp", "basket_time"]
        pd.testing.assert_frame_equal(before[columns], after[columns])
        _, first = purchase_snapshot(ledger, credits, cutoff=end)
        _, second = purchase_snapshot(ledger, switched, cutoff=end)
        pd.testing.assert_frame_equal(first, second)
        kwargs = dict(target_start="2011-09-01", target_end=end)
        pd.testing.assert_frame_equal(net_retained_labels(ledger, credits, **kwargs),
                                      net_retained_labels(ledger, switched, **kwargs))


def test_target_end_labels_require_a_surviving_sku_in_any_completed_basket_in_window():
    # Separate source lines on invoice 400, different invoices for the same SKU,
    # a fully reversed prior purchase outside the window, and a cutoff boundary.
    items = [
        ("399", "W", 1, "2011-09-25"),
        ("400", "X", 5, "2011-10-01"),
        ("400", "X", 3, "2011-10-02"),
        ("401", "Y", 2, "2011-10-03"),
        ("402", "Z", 1, "2011-10-04"),
        ("403", "V", 1, "2011-10-05"),
        ("C404", "V", -1, "2011-10-06"),
        ("405", "V", 1, "2011-10-08"),
        ("C406", "X", -5, "2011-10-10"),
        ("C407", "W", -1, "2011-10-11"),
        ("C408", "Y", -2, "2011-10-15"),
        ("C409", "Z", -1, "2011-11-01"),
    ]
    raw = pd.DataFrame(items, columns=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate"])
    raw = raw.assign(ExcelRow=range(2, len(raw) + 2), UnitPrice=2, CustomerID="001",
                     Description="product", Country="United Kingdom")
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    ledger = build_signed_ledger(raw)
    credits = match_reversals(ledger)
    _, events = purchase_snapshot(ledger, credits, cutoff="2011-11-01")
    x = events.loc[events.StockCode.eq("X")].iloc[0]
    assert x.InvoiceNo == "400" and x.quantity == 3 and x.source_lines == 1
    assert x.basket_time == pd.Timestamp("2011-10-02")
    assert not events["StockCode"].eq("Y").any()
    october = net_retained_labels(ledger, credits, target_start="2011-10-01", target_end="2011-11-01")
    assert list(october.itertuples(index=False, name=None)) == [("001", "V"), ("001", "X"), ("001", "Z")]
    november = net_retained_labels(ledger, credits, target_start="2011-11-01", target_end="2011-12-01")
    assert november.empty
    event_only = raw.loc[raw.InvoiceDate.lt("2011-11-01")]
    old_ledger = build_signed_ledger(event_only)
    unchanged = net_retained_labels(old_ledger, match_reversals(old_ledger),
                                    target_start="2011-10-01", target_end="2011-11-01")
    pd.testing.assert_frame_equal(october, unchanged)
