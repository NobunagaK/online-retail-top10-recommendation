"""Evaluate one full candidate catalog at a time for the four recommendation models."""

import hashlib
import json
from math import isfinite, log2
from numbers import Integral

import numpy as np
import pandas as pd


AUC_SAMPLE_SEED = 5481


def _ordered_catalog(catalog):
    """Return one exact StockCode per positive vocabulary ID, in score-tie order."""
    items = catalog[["item_id", "StockCode"]].drop_duplicates()
    if len(items) and (not pd.api.types.is_integer_dtype(items["item_id"])
                       or items["item_id"].isna().any() or items["item_id"].le(0).any()):
        raise ValueError("Candidate item IDs must be positive integers. ID 0 is padding.")
    if not items["StockCode"].map(lambda code: isinstance(code, str) and bool(code)).all():
        raise ValueError("Candidate StockCodes must be nonempty strings.")
    if items["item_id"].duplicated().any() or items["StockCode"].duplicated().any():
        raise ValueError("The catalog must map each item ID to one distinct StockCode.")
    return items.sort_values("StockCode", ignore_index=True)


def sample_auc_pairs(split, query_id, catalog, positive_item_ids, seed=AUC_SAMPLE_SEED):
    """Build a reusable [split, query_id, item_id, label] manifest for one eligible query.

    Keep every in-catalog positive and sample up to ten distinct negatives per
    positive. The stable draw depends on the split, query ID and sampling seed.
    The model's training seed does not enter this calculation.
    """
    items = _ordered_catalog(catalog)
    positives = set(positive_item_ids)
    if not positives.issubset(items["item_id"]):
        raise ValueError("Positive item IDs must belong to this query's candidate catalog.")
    is_positive = items["item_id"].isin(positives)
    pool = items.loc[~is_positive, "item_id"].to_numpy()
    key = json.dumps([int(seed), str(split), str(query_id)], ensure_ascii=False, separators=(",", ":"))
    query_seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64(query_seed))
    negatives = rng.choice(pool, size=min(10 * len(positives), len(pool)), replace=False)
    pairs = items.loc[is_positive | items["item_id"].isin(negatives), ["item_id"]].copy()
    pairs.insert(0, "query_id", query_id)
    pairs.insert(0, "split", split)
    pairs["label"] = pairs["item_id"].isin(positives).astype("int8")
    return pairs.reset_index(drop=True)


def sampled_auc(positive_scores, negative_scores):
    """Return P(positive > negative) + 0.5*P(tie) within a query, or None for one class."""
    positive = np.asarray(positive_scores, dtype="float64")
    negative = np.asarray(negative_scores, dtype="float64")
    if positive.ndim != 1 or negative.ndim != 1:
        raise ValueError("Pass one score per sampled item.")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("Sampled scores must be finite.")
    if not len(positive) or not len(negative):
        return None
    negatives = np.sort(negative)
    lower = np.searchsorted(negatives, positive, side="left")
    upper = np.searchsorted(negatives, positive, side="right")
    return float(((lower + upper) / (2 * len(negatives))).mean())


def ranking_metrics(ranked_item_ids, positive_item_ids, all_positive_count=None):
    """Return per-query binary ranking metrics from distinct ranked candidate IDs.

    positive_item_ids is Y (the catalog intersection). all_positive_count is the
    count of distinct, valid items bought that month before catalog filtering,
    so never-seen StockCodes need no constructed vocabulary IDs. Call only for
    customers in the desired evaluation cohort. An empty Y skips primary metrics.
    """
    positives = set(positive_item_ids)
    if all_positive_count is None:
        all_positive_count = len(positives)
    if not isinstance(all_positive_count, Integral) or all_positive_count < len(positives):
        raise ValueError("All-item positive count must cover the distinct catalog positives.")
    top = list(dict.fromkeys(ranked_item_ids))[:10]
    hit_ranks = [rank for rank, item in enumerate(top, start=1) if item in positives]
    dcg = sum(1 / log2(rank + 1) for rank in hit_ranks)
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(10, len(positives)) + 1))
    return {
        "hits_at_10": len(hit_ranks),
        "catalog_positive_items": len(positives),
        "all_positive_items": int(all_positive_count),
        "recall_at_10": len(hit_ranks) / len(positives) if positives else None,
        "ndcg_at_10": dcg / ideal if ideal else None,
        "supplementary_recall_at_10": len(hit_ranks) / all_positive_count if all_positive_count else None,
    }


def evaluate_query(catalog, score_chunks, positive_item_ids, auc_pairs=None, *, all_positive_count=None):
    """Rank one complete catalog and return (top10, sampled_raw_scores, per_query_metrics).

    Pass one DataFrame with columns item_id and score, or an iterable of such
    frames for the same query. Scores outside the query catalog (including PAD)
    are ignored. The caller supplies all catalog scores, including prior purchases.
    A repeated candidate with identical scores is counted once. Conflicting
    scores or a missing candidate make the ranking ambiguous and raise an error.

    Supply a persisted AUC manifest for this query to include its sampled
    raw scores. Memory is bounded by one query's catalog irrespective of the
    number of queries. No entire customer-by-item score matrix is required.
    """
    items = _ordered_catalog(catalog)
    codes = items.set_index("item_id")["StockCode"].to_dict()
    positives = set(positive_item_ids)
    if not positives.issubset(codes):
        raise ValueError("Positive item IDs must belong to this query's candidate catalog.")
    chunks = [score_chunks] if isinstance(score_chunks, pd.DataFrame) else score_chunks
    scores = {}
    for chunk in chunks:
        rows = chunk.loc[chunk["item_id"].isin(codes), ["item_id", "score"]]
        for item, value in rows.itertuples(index=False, name=None):
            score = float(value)
            if not isfinite(score):
                raise ValueError(f"Nonfinite score for candidate {item}.")
            if item in scores and scores[item] != score:
                raise ValueError(f"Conflicting scores for candidate {item}.")
            scores[item] = score
    if len(scores) != len(codes):
        raise ValueError(f"Missing scores for {len(codes) - len(scores)} candidates.")
    best = sorted(scores, key=lambda item: (-scores[item], codes[item]))[:10]
    top = pd.DataFrame({
        "rank": range(1, len(best) + 1), "item_id": best,
        "StockCode": [codes[item] for item in best], "score": [scores[item] for item in best],
    })
    metrics = ranking_metrics(best, positives, all_positive_count)
    sample_keys = ["split", "query_id", "item_id"]
    sampled = pd.DataFrame(columns=[*sample_keys, "score"])
    auc = None
    n_pos = n_neg = 0
    if auc_pairs is not None:
        if (auc_pairs[["split", "query_id"]].drop_duplicates().shape[0] > 1
                or auc_pairs["item_id"].duplicated().any()
                or not auc_pairs["item_id"].isin(codes).all()
                or not auc_pairs["label"].isin([0, 1]).all()):
            raise ValueError("The AUC manifest must contain unique catalog items for one query and binary labels.")
        positive = auc_pairs["label"].eq(1)
        if set(auc_pairs.loc[positive, "item_id"]) != positives:
            raise ValueError("The AUC manifest must contain all and only the query positives as label 1.")
        n_pos = len(positives)
        n_neg = len(auc_pairs) - n_pos
        if n_neg != min(10 * n_pos, len(items) - n_pos):
            raise ValueError("The AUC manifest has the wrong number of sampled negatives.")
        sampled = auc_pairs[sample_keys].copy()
        sampled["score"] = sampled["item_id"].map(scores)
        auc = sampled_auc(sampled.loc[positive, "score"], sampled.loc[~positive, "score"])
    metrics.update({
        "sampled_auc": auc, "auc_requested": auc_pairs is not None,
        "auc_positive_items": n_pos, "auc_negative_items": n_neg,
    })
    return top, sampled, metrics


def summarize_metrics(per_query):
    """Macro-average one record per customer-month, with metric-specific denominators.

    For supplementary recall, include every active customer with a valid
    training query, even if their entire target-month basket lies outside the
    catalog. The eligible query population is constructed by the caller.
    AUC sample counts cover the mixed-class queries contributing to the mean.
    """
    rows = pd.DataFrame(per_query)
    summary = {}
    for field in ("recall_at_10", "ndcg_at_10", "supplementary_recall_at_10", "sampled_auc"):
        valid = rows[field].dropna() if len(rows) else pd.Series(dtype="float64")
        summary[field] = float(valid.mean()) if len(valid) else None
    eligible = rows.loc[rows["sampled_auc"].notna()] if len(rows) else rows
    summary.update({
        "primary_queries": int(rows["recall_at_10"].count()) if len(rows) else 0,
        "supplementary_queries": int(rows["supplementary_recall_at_10"].count()) if len(rows) else 0,
        "auc_valid_queries": len(eligible),
        "auc_skipped_queries": int(rows["auc_requested"].sum()) - len(eligible) if len(rows) else 0,
        "auc_positive_items": int(eligible["auc_positive_items"].sum()) if len(eligible) else 0,
        "auc_negative_items": int(eligible["auc_negative_items"].sum()) if len(eligible) else 0,
    })
    return summary
