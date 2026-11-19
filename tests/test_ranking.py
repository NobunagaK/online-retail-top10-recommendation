from math import log2

import pandas as pd
import pytest

from src.evaluation.ranking import (
    evaluate_query, ranking_metrics, sample_auc_pairs, sampled_auc, summarize_metrics,
)


def catalog(*codes):
    return pd.DataFrame({"item_id": range(1, len(codes) + 1), "StockCode": list(codes)})


def scores(values):
    return pd.DataFrame({"item_id": range(1, len(values) + 1), "score": values})


def test_hand_rank_duplicate_predictions_and_distinct_targets():
    # B is second and D is eleventh. Repeating a recommended or purchased SKU
    # cannot supply extra hits or consume a second recommendation slot.
    metrics = ranking_metrics(list("AABCEFGHIJKD"), ["B", "D", "B"])
    assert metrics["hits_at_10"] == 1
    assert metrics["catalog_positive_items"] == 2
    assert metrics["recall_at_10"] == .5
    assert metrics["ndcg_at_10"] == pytest.approx(.3868528072)
    assert ranking_metrics([1, 2, 3], [4])["ndcg_at_10"] == 0
    many = ranking_metrics(range(1, 21), range(1, 21))
    assert many["recall_at_10"] == .5
    assert many["ndcg_at_10"] == 1


def test_dense_and_chunked_ranking_mask_catalog_and_save_scores_below_ten():
    # Equal raw score ties by exact StockCode, not by item_id, prior history
    # or input order. OOV 19 and PAD 0 both score above the true catalog.
    items = catalog(*list("bAcDEFGHIJKL"))
    raw = scores([10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    raw.loc[len(raw)] = [19, 999]
    raw.loc[len(raw)] = [0, 111]
    positives = [1, 12]
    pairs = sample_auc_pairs("validation", "q", items.sample(frac=1, random_state=3), positives)
    top, sample, metrics = evaluate_query(items, raw, positives, pairs, all_positive_count=4)
    chunks = [raw.iloc[5:], raw.iloc[:5], raw.iloc[[0]]]
    streamed = evaluate_query(pd.concat([items, items.iloc[[0]]]), iter(chunks),
                              positives, pairs, all_positive_count=4)
    pd.testing.assert_frame_equal(streamed[0], top)
    pd.testing.assert_frame_equal(streamed[1], sample)
    assert streamed[2] == metrics
    assert top["StockCode"].tolist() == list("AbcDEFGHIJ")
    assert top["rank"].tolist() == list(range(1, 11))
    assert metrics["recall_at_10"] == .5
    assert metrics["supplementary_recall_at_10"] == .25
    assert sample.set_index("item_id").loc[12, "score"] == 0
    assert sample.set_index("item_id").loc[1, "score"] == 10


def test_missing_candidates_and_conflicting_or_nonfinite_scores_fail():
    items = catalog("A", "B")
    with pytest.raises(ValueError, match="Missing scores"):
        evaluate_query(items, scores([9]), [1])
    with pytest.raises(ValueError, match="Conflicting scores"):
        evaluate_query(items, [scores([9, 3]), scores([6])], [1])
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluate_query(items, scores([9, float("nan")]), [1])


def test_auc_manifest_is_fixed_excludes_all_positives_and_preserves_code_order():
    items = catalog(*[f"S{n:03d}" for n in reversed(range(100))])
    positives = [4, 20, 4]
    reference = sample_auc_pairs("validation", "u1_o", items, positives)
    shuffled = sample_auc_pairs("validation", "u1_o", items.sample(frac=1, random_state=11),
                                [20, 4])
    pd.testing.assert_frame_equal(shuffled, reference)
    assert not reference.item_id.duplicated().any()
    assert set(reference.loc[reference.label == 1, "item_id"]) == {4, 20}
    assert len(reference.loc[reference.label == 0]) == 20
    assert reference.item_id.is_monotonic_decreasing
    assert set(reference.split) == {"validation"}
    assert set(reference.query_id) == {"u1_o"}
    for split, key in [("validation", "u2_o"), ("synthetic", "u1_o")]:
        alternative = sample_auc_pairs(split, key, items, positives)
        assert set(alternative.item_id) != set(reference.item_id)


def test_auc_ties_query_macro_single_class_and_supplementary_population():
    # A third eligible customer purchased only an out-of-catalog item.
    first = catalog("A", "B", "C")
    second = catalog("A", "D")
    single = catalog("B")
    per_query = []
    for q, items, values, positives, all_count in [
        ("q1", first, [.8, .8, .2], [1], 2),
        ("q2", second, [.1, .3], [1], 1),
        ("q3", first, [7, 6, 4], [], 1),
        ("q4", single, [.7], [1], 1),
    ]:
        manifest = sample_auc_pairs("validation", q, items, positives) if positives else None
        _, raw_sample, metrics = evaluate_query(items, scores(values), positives, manifest,
                                                 all_positive_count=all_count)
        if manifest is None:
            assert raw_sample.empty
        else:
            assert raw_sample.item_id.tolist() == manifest.item_id.tolist()
        per_query.append(metrics)
    assert [q["sampled_auc"] for q in per_query] == [.75, 0, None, None]
    assert per_query[2]["recall_at_10"] is None
    assert per_query[2]["supplementary_recall_at_10"] == 0
    summary = summarize_metrics(per_query)
    assert summary["sampled_auc"] == .375
    assert summary["auc_valid_queries"] == 2 and summary["auc_skipped_queries"] == 1
    assert summary["auc_positive_items"] == 2 and summary["auc_negative_items"] == 3
    assert summary["primary_queries"] == 3 and summary["supplementary_queries"] == 4
    assert summary["recall_at_10"] == 1
    assert summary["ndcg_at_10"] == pytest.approx((2 + 1 / log2(3)) / 3)
    assert summary["supplementary_recall_at_10"] == pytest.approx((.5 + 1 + 0 + 1) / 4)
    assert summarize_metrics(per_query[2:])["sampled_auc"] is None
    assert summarize_metrics([])["sampled_auc"] is None
    assert sampled_auc([], [.1]) is None


def test_manifest_requires_representative_pairs_and_fresh_positive_labels():
    items = catalog("A", "B", "C")
    manifest = sample_auc_pairs("validation", "q1", items, [1])
    with pytest.raises(ValueError, match="wrong number"):
        evaluate_query(items, scores([1, 2, 3]), [1], manifest.iloc[:2])
    with pytest.raises(ValueError, match="positives"):
        evaluate_query(items, scores([1, 2, 3]), [2], manifest)
    with pytest.raises(ValueError, match="must cover"):
        evaluate_query(items, scores([1, 2, 3]), [1, 2], all_positive_count=1)
