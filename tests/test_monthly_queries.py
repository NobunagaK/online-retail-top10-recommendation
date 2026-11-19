"""Point-in-time and eligibility regressions for the shared D2 data."""

import json

import pandas as pd
import pytest

from scripts.build_queries import QUANTITY_POLICY, load_inputs, save_outputs, verify_outputs
from src.data.monthly_queries import (
    HISTORY_END, build_monthly_tables, build_query_window, sample_training_pairs,
)
from src.data.purchase_snapshot import purchase_snapshot


def ledger_fixture():
    # A and B have history, but only u1 and inactive make a valid training query.
    sales = [
        ("u1", "A", "2011-01-02", 2), ("u1", "B", "2011-01-02", 1),
        ("history-only", "B", "2011-01-10", 2), ("inactive", "A", "2011-01-12", 1),
        ("oov-only", "A", "2011-01-12", 1),
        ("u1", "A", "2011-02-10", 2), ("u1", "B", "2011-02-11", 1),
        ("u1", "B", "2011-02-12", 2), ("u1", "F", "2011-02-13", 1),
        ("inactive", "A", "2011-02-14", 1), ("oov-only", "A", "2011-02-14", 1),
        ("u1", "SEP", "2011-09-12", 1),
        ("u1", "A", "2011-10-02", 2), ("u1", "A", "2011-10-03", 1),
        ("u1", "OCT", "2011-10-04", 2), ("u1", "SEP", "2011-10-05", 1),
        ("new", "A", "2011-10-05", 1), ("history-only", "A", "2011-10-05", 1),
        ("oov-only", "OCT", "2011-10-06", 1),
    ]
    ledger = pd.DataFrame([{
        "ExcelRow": n, "InvoiceNo": str(n), "StockCode": sku, "Description": sku,
        "InvoiceDate": pd.Timestamp(date), "CustomerID": customer, "Quantity": quantity,
        "UnitPrice": 1., "Country": "United Kingdom", "ledger_type": "sale",
    } for n, (customer, sku, date, quantity) in enumerate(sales, 2)])
    # Feb B on line 8 is fully credited before March; the second February sale survives.
    # Jan B on line 3 is credited in October; the frozen October 1 history retains it.
    credits = pd.DataFrame({
        "credit_time": pd.to_datetime(["2011-02-20", "2011-10-22", "2011-11-02"]),
        "purchase_excel_row": pd.array([8, 3, 15], dtype="Int64"),
        "credited_units": [1, 1, 1], "status": ["cancellation", "return", "return"],
    })
    return ledger, credits


@pytest.fixture(scope="module")
def built():
    return build_monthly_tables(*ledger_fixture(), seed=21)


def test_train_eval_shared_scope_distinct_labels_and_unseen_items(built):
    feb = built["labels"].query("query_id == '2011-02/u1'").set_index("StockCode")
    assert feb.index.tolist() == ["A", "B", "F"]
    assert feb["in_catalog"].to_dict() == {"A": True, "B": True, "F": False}
    assert pd.isna(feb.at["F", "item_id"])
    assert "F" in built["item_vocab"]["stock_code"].values
    oct_cohort = built["cohorts"].loc[lambda x: x["split"].eq("validation")].set_index("customer_id")
    assert oct_cohort["coverage_status"].to_dict() == {
        "history-only": "no_valid_training_query", "inactive": "inactive", "new": "no_history",
        "oov-only": "only_out_of_catalog", "u1": "eligible",
    }
    assert oct_cohort.index[oct_cohort["supplemental_eligible"]].tolist() == ["oov-only", "u1"]
    assert oct_cohort.loc["u1", ["positive_count", "full_positive_count"]].tolist() == [2, 3]
    oct_cov = built["coverage"].iloc[-1]
    assert (oct_cov["active_customers"], oct_cov["eligible_customers"], oct_cov["active_without_history"],
            oct_cov["inactive_valid_training_customers"], oct_cov["supplemental_full_positive_pairs"]) == (4, 1, 1, 1, 4)
    assert built["queries"]["split"].unique().tolist() == ["train", "validation"]
    assert built["queries"]["target_end"].max() == pd.Timestamp("2011-11-01")
    late = built["item_support"].set_index("stock_code").loc["SEP"]
    assert not late["mf_observed_train_pair"] and not late["sequence_in_train_catalog"]
    sampled = built["auc_pairs"].loc[lambda x: x.label.eq(1), "item_id"]
    october_positives = built["labels"].loc[lambda x: x["eligible"] & x["in_catalog"] & x["split"].eq("validation")]
    assert set(sampled) == set(october_positives["item_id"])


def test_training_query_hash_and_input_invariant_to_later_events(built):
    ledger, credits = ledger_fixture()
    future = ledger.iloc[-1:].assign(InvoiceNo="later", ExcelRow=100, CustomerID="changed",
                                      StockCode="NEW", InvoiceDate=pd.Timestamp("2011-11-19"))
    perturbed = pd.concat([ledger, future], ignore_index=True).sample(frac=1, random_state=5)
    second = build_monthly_tables(perturbed, credits.iloc[::-1], seed=21)
    for key, value in built.items():
        pd.testing.assert_frame_equal(value, second[key])
    # Adding a September purchase can extend IDs but cannot change early histories,
    # labels or sampling; F was unseen in February despite its later training ID.
    added = future.assign(InvoiceNo="other", StockCode="SEPTEMBER", InvoiceDate=pd.Timestamp("2011-09-23"))
    later = build_monthly_tables(pd.concat([ledger, added], ignore_index=True), credits, seed=21)
    for key in ["train_pairs", "labels", "queries"]:
        left = built[key].loc[lambda x: x.query_id.str.startswith("2011-02/")]
        right = later[key].loc[lambda x: x.query_id.str.startswith("2011-02/")]
        pd.testing.assert_frame_equal(left, right)


def test_november_can_only_share_frozen_history_without_accessing_real_test(built):
    ledger, credits = ledger_fixture()
    future_order = ledger.iloc[-1:].assign(ExcelRow=70, InvoiceNo="70", CustomerID="u1",
                                            StockCode="A", InvoiceDate=pd.Timestamp("2011-11-09"))
    test_ledger = pd.concat([ledger, future_order], ignore_index=True)
    _, frozen = purchase_snapshot(test_ledger, credits, cutoff=HISTORY_END)
    _, future = purchase_snapshot(test_ledger, credits, cutoff="2011-12-01")
    training_users = set(built["queries"].loc[lambda x: x.split.eq("train"), "customer_id"])
    november = build_query_window("test", HISTORY_END, "2011-11-01", "2011-12-01", frozen, future,
                                  built["item_vocab"], training_users)
    assert november["labels"]["StockCode"].tolist() == ["A"]
    val = built["queries"].query("split == 'validation'").iloc[0]
    test = november["queries"].iloc[0]
    assert (val.history_end, val.catalog_sha256) == (test.history_end, test.catalog_sha256)
    assert "OCT" not in november["catalogs"]["StockCode"].tolist()
    with pytest.raises(ValueError, match="validation target"):
        build_query_window("validation", HISTORY_END, "2011-11-01", "2011-12-01", frozen, future,
                           built["item_vocab"], training_users)
    with pytest.raises(ValueError, match="frozen"):
        build_query_window("test", "2011-11-01", "2011-11-01", "2011-12-01", frozen, future,
                           built["item_vocab"], training_users)


def test_training_sample_ratios_seed_and_known_repurchase():
    sample = sample_training_pairs("2011-02/u", range(1, 100), [1, 2], seed=42)
    assert sample["item_id"].is_unique and sample["label"].value_counts().to_dict() == {0: 10, 1: 2}
    assert set(sample.loc[sample.label.eq(1), "item_id"]) == {1, 2}
    pd.testing.assert_frame_equal(sample, sample_training_pairs("2011-02/u", range(1, 100), [2, 1], seed=42))
    assert not sample.equals(sample_training_pairs("2011-02/u", range(1, 100), [1, 2], seed=41))
    exhausted = sample_training_pairs("q", [9, 3, 5], [9, 5], seed=42)
    assert exhausted["label"].tolist() == [1, 0, 1]
    with pytest.raises(ValueError, match="catalog"):
        sample_training_pairs("q", [9, 3, 5], [7])


def test_manifest_records_every_table_and_detects_bad_source(tmp_path, built):
    marker = save_outputs(tmp_path, built, {"synthetic": "unit-test"}, 21)
    assert marker == json.loads((tmp_path / "manifest.json").read_text())
    assert marker["splits"] == ["train", "validation"]
    assert marker["outputs"]["queries.parquet"]["rows"] == len(built["queries"])
    assert pd.read_parquet(tmp_path / "histories.parquet").iloc[0].history_item_ids.tolist()
    assert verify_outputs(tmp_path) == marker
    # An interrupted copy (or stale mixed output) must be detectable and repairable.
    (tmp_path / "queries.parquet").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="repair"):
        verify_outputs(tmp_path)
    assert save_outputs(tmp_path, built, {"synthetic": "unit-test"}, 21) == marker
    (tmp_path / "results").mkdir()
    path = tmp_path / "results/data_manifest.json"
    path.write_text(json.dumps({"recommendation_quantity_policy": "pre-credit"}))
    with pytest.raises(ValueError, match="net-retained"):
        load_inputs(tmp_path)
    (tmp_path / "data/processed").mkdir(parents=True)
    relative = "data/processed/signed_ledger_b.csv.gz"
    (tmp_path / relative).write_bytes(b"changed")
    path.write_text(json.dumps({"recommendation_quantity_policy": QUANTITY_POLICY,
                                "training_snapshot": {"end_exclusive": "2011-10-01"},
                                "outputs": {relative: {"sha256": "a different digest"}}}))
    with pytest.raises(ValueError, match="manifest"):
        load_inputs(tmp_path)
