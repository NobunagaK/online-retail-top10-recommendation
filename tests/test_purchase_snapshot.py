import json

import pandas as pd

from scripts.prepare_data import BASELINE_PATH, build_cleaning_outputs, save_cleaning_results
from src.data.baskets import build_basket_events
from src.data.purchase_snapshot import purchase_snapshot
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


def test_snapshot_cancellation_cutoff_partial_units_returns_and_original_basket_time():
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
    assert after.loc[after.ExcelRow.eq(2), ["original_quantity", "cancelled_units", "returned_units", "Quantity"]].iloc[0].tolist() == [10, 3, 1, 7]
    assert final_events.loc[final_events.CustomerID.eq("001"), "basket_time"].item() == pd.Timestamp("2011-09-30 23:45")
    pd.testing.assert_frame_equal(final_events, build_basket_events(after))


def test_d1_frames_reconcile_individual_outcomes_and_publish_distinct_snapshot_manifests(tmp_path):
    raw = source_rows()
    frames, manifest = build_cleaning_outputs(raw)
    decisions = frames["data/processed/row_decisions.csv.gz"].set_index("ExcelRow")
    assert decisions.loc[2, "first_rule"] == "sale_partially_cancelled"
    assert decisions.loc[2, "returned_units"] == 1
    assert decisions.loc[3, "first_rule"] == "sale_fully_cancelled"
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
    for name, frame in frames.items():
        assert (tmp_path / name).is_file()
        assert saved["outputs"][name]["rows"] == len(frame)
