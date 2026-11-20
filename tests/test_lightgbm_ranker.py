"""As-of feature, grouped ranking, native checkpoint and tie-order regressions."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.history_features import history_features
from src.evaluation.ranking import evaluate_query
from src.features.asof import AsOfFeatures, FEATURE_NAMES
from src.models.lightgbm_ranker import _validation_metric, catalog_ndcg_at_10, make_model


def d2_fixture():
    # ID order intentionally disagrees with the StockCode tie order (A, B, C).
    vocab = pd.DataFrame({"stock_code": ["B", "C", "A"], "item_id": [1, 2, 3]})
    events = pd.DataFrame([
        ("u", "100", "2011-01-15", "A", 2, 6.),
        ("u", "101", "2011-01-29", "A", 1, 1.5),
        ("u", "101", "2011-01-29", "B", 2, 2.),
        ("v", "102", "2011-01-28", "B", 1, 1.),
        ("v", "103", "2011-01-31", "C", 3, 12.),
        ("u", "104", "2011-02-01", "A", 20, 200.),
        ("v", "105", "2011-03-01", "C", 2, 8.),
    ], columns=["CustomerID", "InvoiceNo", "basket_time", "StockCode", "quantity", "amount_gbp"])
    events["basket_time"] = pd.to_datetime(events["basket_time"])
    feb = pd.Timestamp("2011-02-01")
    oct1 = pd.Timestamp("2011-10-01")
    frames = [history_features(events, cutoff) for cutoff in (feb, oct1)]
    features = {name: pd.concat([f[name] for f in frames], ignore_index=True) for name in frames[0]}
    train_queries = pd.DataFrame([
        ("t-v", "v", feb, "feb"), ("t-u", "u", feb, "feb"),
    ], columns=["query_id", "customer_id", "history_end", "catalog_id"])
    val_queries = pd.DataFrame([("e-u", "u", oct1, "oct"), ("e-v", "v", oct1, "oct")],
                               columns=train_queries.columns)
    pairs = pd.DataFrame([
        ("t-u", 1, 0), ("t-v", 3, 0), ("t-v", 2, 1),
        ("t-u", 3, 1), ("t-v", 1, 0), ("t-u", 2, 0),
    ], columns=["query_id", "item_id", "label"])
    catalogs = pd.concat([vocab.rename(columns={"stock_code": "StockCode"}).assign(catalog_id=cut)
                          for cut in ("feb", "oct")])
    return SimpleNamespace(train_queries=train_queries, val_queries=val_queries, train_pairs=pairs, vocab=vocab,
                           catalogs=catalogs, catalog_ids={"feb": [3, 1, 2], "oct": [3, 1, 2]},
                           positives={"t-u": [3], "t-v": [2], "e-u": [3], "e-v": [2]}, features=features)


def test_purchase_frequency_cutoff_unseen_and_weighted_price():
    data = d2_fixture()
    store = AsOfFeatures(data.features, data.vocab)
    feb = next(data.train_queries.query("customer_id == 'u'").itertuples(index=False))
    october = next(data.val_queries.itertuples(index=False))
    feb_rows = pd.DataFrame(store.matrix(feb, [2, 3, 1]), columns=FEATURE_NAMES)
    bought = feb_rows.iloc[1]
    assert bought["pair_previously_purchased"] == 1
    assert bought["pair_invoice_count"] == 2 and bought["pair_quantity"] == 3
    assert bought["pair_days_since_last_purchase"] == 3
    assert bought["pair_invoices_7d"] == 1 and bought["pair_invoices_30d"] == 2
    assert bought["item_weighted_unit_price_gbp"] == pytest.approx((6.0 + 1.5) / 3)
    assert bought["customer_mean_basket_size"] == 1.5 and bought["customer_amount_gbp"] == 9.5
    unseen = feb_rows.iloc[0]
    assert unseen["pair_previously_purchased"] == unseen["pair_invoice_count"] == unseen["pair_quantity"] == 0
    assert np.isnan(unseen["pair_days_since_last_purchase"])
    oct_row = dict(zip(FEATURE_NAMES, store.matrix(october, [3])[0]))
    assert oct_row["pair_quantity"] == 23
    assert oct_row["item_weighted_unit_price_gbp"] == pytest.approx(207.5 / 23)
    data.features["customer_item_features"].loc[0, "last_purchase_time"] = feb.history_end
    with pytest.raises(ValueError, match="cutoff"):
        AsOfFeatures(data.features, data.vocab)


def test_groups_are_contiguous_share_all_positives_and_catalog():
    data = d2_fixture()
    model = make_model(data, {}, "cpu")
    train = list(model._training_groups(data))
    assert [row[0].query_id for row in train] == ["t-u", "t-v"]
    assert [ids.tolist() for _, ids, _ in train] == [[3, 1, 2], [3, 1, 2]]
    features, labels, groups = model._matrix(train)
    assert groups.tolist() == [3, 3] and sum(groups) == len(features) == len(labels)
    assert labels.tolist() == [1, 0, 0, 0, 0, 1]
    assert [row[1].tolist() for row in model._validation_groups(data)] == [[3, 1, 2], [3, 1, 2]]
    data.train_pairs = pd.concat([data.train_pairs, data.train_pairs.iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        list(model._training_groups(data))


def test_custom_macro_metric_matches_project_evaluator_on_score_ties():
    sizes, binary, values, expected = [], [], [], []
    for n, relevant in [(13, {2, 8, 11}), (5, {3})]:
        # The custom LightGBM metric is passed each group in exact StockCode order.
        catalog = pd.DataFrame({"item_id": range(1, n + 1), "StockCode": [f"SKU{i:02d}" for i in range(n)]})
        scores = pd.DataFrame({"item_id": catalog.item_id, "score": [int(i % 3 == 1) for i in range(n)]})
        _, _, metrics = evaluate_query(catalog, scores, relevant)
        expected.append(metrics["ndcg_at_10"])
        sizes.append(n)
        binary.extend(catalog.item_id.isin(relevant).astype(int))
        values.extend(scores.score)
    assert catalog_ndcg_at_10(binary, values, sizes) == pytest.approx(np.mean(expected))
    assert _validation_metric(binary, values, None, sizes) == ("project_ndcg_at_10", pytest.approx(np.mean(expected)), True)


def test_cpu_fit_best_native_tree_roundtrip_and_full_candidate_scoring(tmp_path):
    pytest.importorskip("lightgbm")
    data = d2_fixture()
    config = {"n_estimators": 8, "patience": 2, "num_leaves": 3, "min_child_samples": 1, "n_jobs": 1}
    model = make_model(data, config, "cpu")
    best = model.train_epoch(data)
    assert np.isfinite(best) and 0 <= best <= 1
    assert best == model.validation_history[model.best_iteration - 1]["ndcg_at_10"]
    query = next(data.val_queries.itertuples(index=False))
    path = tmp_path / "ranker.txt"
    model.save(path)
    loaded = make_model(data, config, "cpu").restore(path)
    assert loaded.best_iteration == model.best_iteration
    expected = model.score(query, [3, 1, 2])
    np.testing.assert_allclose(loaded.score(query, [2, 3, 1]), expected[[2, 0, 1]])
    with pytest.raises(RuntimeError, match="already fitted"):
        model.train_epoch(data)


def test_native_vs_project_metric_with_ties(record_property):
    """Record the library tie result separately from the project metric used for selection."""
    lgb = pytest.importorskip("lightgbm")
    # Two full SKU-ordered catalogs, all tied at one score (no split is possible).
    truth = np.zeros(24, dtype=np.int8)
    truth[[1, 10, 12]] = 1
    groups = [12, 12]
    x = np.zeros((24, 1), dtype=np.float32)
    logged = {}
    probe = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=1, verbosity=-1, n_jobs=1)
    probe.fit(x, truth, group=groups, eval_set=[(x.copy(), truth.copy())], eval_group=[groups],
              eval_names=["probe"], eval_at=(10,), eval_metric=_validation_metric,
              callbacks=[lgb.record_evaluation(logged)])
    raw = probe.booster_.predict(x, num_threads=1)
    assert np.unique(raw).size == 1
    actual = logged["probe"]["project_ndcg_at_10"][0]
    native = logged["probe"]["ndcg@10"][0]
    assert actual == pytest.approx(catalog_ndcg_at_10(truth, raw, groups))
    assert np.isfinite(native)
    record_property("native_ndcg_at_10", float(native))
    record_property("project_ndcg_at_10", float(actual))
    record_property("native_agrees_on_tied_fixture", bool(np.isclose(native, actual)))
