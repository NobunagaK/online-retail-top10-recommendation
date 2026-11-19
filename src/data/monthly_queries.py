"""Construct shared point-in-time queries and labels from D1 net purchase events."""

import hashlib
import json
import random

import pandas as pd


TRAIN_START = pd.Timestamp("2010-12-01")
HISTORY_END = pd.Timestamp("2011-10-01")
VALIDATION_END = pd.Timestamp("2011-11-01")
TRAIN_MONTHS = tuple(pd.date_range("2011-02-01", "2011-09-01", freq="MS"))
DEFAULT_SEED = 42


def ordered_hash(values):
    """SHA-256 of an ordered sequence encoded as compact UTF-8 JSON."""
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def catalog_at(history_events, history_end, vocab):
    """Use only StockCodes with a positive retained basket at this cutoff."""
    cutoff = pd.Timestamp(history_end)
    if not history_events["basket_time"].lt(cutoff).all():
        raise ValueError("A catalog received basket events beyond its history cutoff.")
    codes = sorted(history_events["StockCode"].unique())
    ids = vocab.set_index("stock_code")["item_id"].reindex(codes)
    if ids.isna().any() or ids.le(0).any() or ids.duplicated().any():
        raise ValueError("Every catalog SKU needs its own positive training-vocabulary ID.")
    return pd.DataFrame({"catalog_id": cutoff.date().isoformat(), "history_end": cutoff,
                         "StockCode": codes, "item_id": ids.to_numpy(dtype="int64")})


def build_query_window(split, history_end, target_start, target_end, history_events,
                       target_events, vocab, valid_training_customers):
    """Build a train/evaluation month using separate historical and target-end replays.

    For evaluation, pass the frozen training cutoff and training customer's union;
    target_start may be later (November). Cohorts include inactive historical
    customers, active newcomers, and formerly eligible customers with no remaining
    history at the current cutoff. No targets enter the catalog.
    """
    history_end, start, end = map(pd.Timestamp, (history_end, target_start, target_end))
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unrecognized query split.")
    if end != start + pd.offsets.MonthBegin(1) or start.day != 1 or start != start.normalize():
        raise ValueError("Each target window must contain one full calendar month.")
    if split != "train" and history_end != HISTORY_END:
        raise ValueError("Validation and test share the frozen October 1 training history.")
    if split == "train" and (history_end != start or start not in TRAIN_MONTHS):
        raise ValueError("Training features end at their February through September month starts.")
    if (split == "validation" and start != HISTORY_END) or (split == "test" and start != VALIDATION_END):
        raise ValueError("The validation target is October and the test target is November.")
    if not target_events["basket_time"].lt(end).all():
        raise ValueError("Target events must come from the exclusive target-end replay.")

    valid_training_customers = set(valid_training_customers)
    seen = set(history_events["CustomerID"].unique())
    catalog = catalog_at(history_events, history_end, vocab)
    catalog_id = history_end.date().isoformat()
    catalog_hash = ordered_hash(list(pair) for pair in catalog[["StockCode", "item_id"]].itertuples(index=False, name=None))
    labels = (target_events.loc[target_events["basket_time"].ge(start), ["CustomerID", "StockCode"]]
              .drop_duplicates().sort_values(["CustomerID", "StockCode"], ignore_index=True)
              .rename(columns={"CustomerID": "customer_id"}))
    labels["item_id"] = labels["StockCode"].map(catalog.set_index("StockCode")["item_id"]).astype("Int64")
    labels["in_catalog"] = labels["StockCode"].isin(catalog["StockCode"])
    active = set(labels["customer_id"])
    all_customers = sorted(seen | active | valid_training_customers)
    cohorts = pd.DataFrame({"customer_id": pd.Series(all_customers, dtype="string")})
    cohorts.insert(0, "query_id", start.strftime("%Y-%m/") + cohorts["customer_id"])
    for name, value in [("split", split), ("history_end", history_end), ("target_start", start),
                        ("target_end", end), ("catalog_id", catalog_id)]:
        cohorts[name] = value
    customers = cohorts["customer_id"]
    cohorts["active"] = customers.isin(active)
    cohorts["has_history"] = customers.isin(seen)
    cohorts["has_valid_training_query"] = customers.isin(valid_training_customers)
    full_counts = labels.groupby("customer_id", sort=False).size()
    inside = labels.loc[labels["in_catalog"]]
    positive_counts = inside.groupby("customer_id", sort=False).size()
    cohorts["full_positive_count"] = customers.map(full_counts).fillna(0).astype("int64")
    cohorts["positive_count"] = customers.map(positive_counts).fillna(0).astype("int64")
    allowed = cohorts["has_history"] if split == "train" else cohorts["has_valid_training_query"]
    cohorts["eligible"] = allowed & cohorts["positive_count"].gt(0)
    cohorts["supplemental_eligible"] = (split != "train") & cohorts["has_valid_training_query"] & cohorts["active"]
    # Mutually exclusive reasons; independent overlapping coverage counts appear below.
    reason = pd.Series("eligible", index=cohorts.index)
    reason.loc[cohorts["positive_count"].eq(0)] = "only_out_of_catalog"
    if split != "train":
        reason.loc[~cohorts["has_valid_training_query"]] = "no_valid_training_query"
    reason.loc[~cohorts["has_history"] & ~allowed] = "no_history"
    reason.loc[~cohorts["active"]] = "inactive"
    cohorts["coverage_status"] = reason
    labels = labels.merge(cohorts[["customer_id", "query_id", "eligible", "supplemental_eligible"]],
                          on="customer_id", validate="many_to_one")
    labels["split"] = split
    labels["target_start"] = start
    labels["catalog_id"] = catalog_id

    keys = ["query_id", "customer_id", "split", "history_end", "target_start", "target_end", "catalog_id",
            "positive_count", "full_positive_count"]
    queries = cohorts.loc[cohorts["eligible"], keys].copy()
    hashes = inside.groupby("customer_id", sort=False)["StockCode"].agg(ordered_hash)
    queries["positive_sha256"] = queries["customer_id"].map(hashes)
    queries["catalog_sha256"] = catalog_hash
    coverage = {
        "split": split, "history_end": history_end, "target_start": start, "target_end": end,
        "catalog_items": len(catalog), "catalog_sha256": catalog_hash,
        "history_customers": len(seen), "prior_valid_training_customers": len(valid_training_customers),
        "active_customers": len(active), "eligible_customers": len(queries),
        "eligible_share_of_active": len(queries) / len(active) if active else float("nan"),
        "active_without_history": len(active - seen),
        "active_with_history_without_valid_training_query": len((active & seen) - valid_training_customers),
        "active_only_out_of_catalog": int((cohorts["active"] & cohorts["positive_count"].eq(0)).sum()),
        "inactive_valid_training_customers": len(valid_training_customers - active),
        "inactive_history_customers": len(seen - active),
        "full_positive_pairs": len(labels), "in_catalog_positive_pairs": len(inside),
        "out_of_catalog_positive_pairs": len(labels) - len(inside),
        "out_of_catalog_positive_share": (len(labels) - len(inside)) / len(labels) if len(labels) else float("nan"),
        "out_of_catalog_skus": labels.loc[~labels["in_catalog"], "StockCode"].nunique(),
        "eligible_in_catalog_positive_pairs": int(queries["positive_count"].sum()),
        "supplemental_customers": int(cohorts["supplemental_eligible"].sum()),
        "supplemental_full_positive_pairs": int(cohorts.loc[cohorts["supplemental_eligible"], "full_positive_count"].sum()),
    }
    return {"catalogs": catalog, "cohorts": cohorts, "queries": queries, "labels": labels,
            "coverage": pd.DataFrame([coverage])}


def sample_training_pairs(query_id, catalog_item_ids, positive_ids, seed=DEFAULT_SEED):
    """All positives and ≤5 unique negatives per positive; supplied catalog is in SKU order."""
    all_ids = list(catalog_item_ids)
    positives = set(positive_ids)
    if not positives or not positives.issubset(all_ids):
        raise ValueError("A training query needs at least one positive, all in its own catalog.")
    negatives = [item for item in all_ids if item not in positives]
    generator = random.Random(ordered_hash([int(seed), str(query_id)]))
    sampled = set(generator.sample(negatives, min(5 * len(positives), len(negatives))))
    selected = [(query_id, item, int(item in positives)) for item in all_ids
                if item in positives or item in sampled]
    return pd.DataFrame(selected, columns=["query_id", "item_id", "label"])


def pair_tables(queries, labels, catalog, seed):
    """Build the model-shared training or evaluation pairs for eligible queries."""
    from src.evaluation.ranking import sample_auc_pairs

    inside = labels["eligible"] & labels["in_catalog"]
    positives = labels.loc[inside].groupby("query_id", sort=False)["item_id"]
    lookup = {qid: ids.astype(int).tolist() for qid, ids in positives}
    train = []
    auc = []
    item_ids = catalog["item_id"].tolist()
    for query in queries.itertuples(index=False):
        positive_ids = lookup[query.query_id]
        if query.split == "train":
            train.append(sample_training_pairs(query.query_id, item_ids, positive_ids, seed))
        else:
            auc.append(sample_auc_pairs(query.split, query.query_id, catalog, positive_ids, seed=5481))
    return (pd.concat(train, ignore_index=True) if train else pd.DataFrame(columns=["query_id", "item_id", "label"]),
            pd.concat(auc, ignore_index=True) if auc else pd.DataFrame(columns=["split", "query_id", "item_id", "label"]))


def build_monthly_tables(ledger, allocations, *, seed=DEFAULT_SEED):
    """Construct the eight training months and October validation. November is sealed."""
    from src.data.history_features import history_features
    from src.data.purchase_snapshot import purchase_snapshot
    from src.data.sequences import build_histories, build_item_vocab

    # Project the same immutable D1 events once at each boundary. The next month's
    # history can also serve as the current month's target-end replay.
    snapshots = {}
    for end in (*TRAIN_MONTHS, HISTORY_END, VALIDATION_END):
        _, events = purchase_snapshot(ledger, allocations, cutoff=end)
        snapshots[end] = events.loc[events["basket_time"].ge(TRAIN_START)]
    vocab = build_item_vocab(snapshots, training_end=HISTORY_END)
    parts = {}
    prior_training_customers = set()

    def add(name, frame):
        parts.setdefault(name, []).append(frame)

    for start in (*TRAIN_MONTHS, HISTORY_END):
        split = "train" if start < HISTORY_END else "validation"
        end = start + pd.offsets.MonthBegin(1)
        history = snapshots[start]
        window = build_query_window(split, start, start, end, history, snapshots[end],
                                    vocab, prior_training_customers)
        queries = window["queries"]
        catalogs = window["catalogs"]
        train, auc = pair_tables(queries, window["labels"], catalogs, seed)
        if split == "train":
            add("train_pairs", train)
        else:
            add("auc_pairs", auc)

        histories = build_histories(history, start, vocab, customer_ids=prior_training_customers)
        histories["history_sha256"] = histories["history_item_ids"].map(ordered_hash)
        queries["history_sha256"] = queries["customer_id"].map(histories.set_index("customer_id")["history_sha256"])
        if queries["history_sha256"].isna().any():
            raise ValueError("Every eligible query needs a point-in-time history.")
        neg_counts = train.loc[train["label"].eq(0)].groupby("query_id").size()
        queries["train_negative_count"] = queries["query_id"].map(neg_counts).fillna(0).astype(int)
        queries["train_sampling_seed"] = pd.array([seed if split == "train" else None] * len(queries), dtype="Int64")
        add("histories", histories)
        for name, frame in history_features(history, start, customer_ids=prior_training_customers).items():
            add(name, frame)
        if split == "train":
            prior_training_customers.update(queries["customer_id"])

        diagnostics = histories.loc[histories["customer_id"].isin(queries["customer_id"])]
        coverage = window["coverage"]
        coverage["truncated_queries_L64"] = int(diagnostics["is_truncated"].sum())
        coverage["within_basket_cut_queries_L64"] = int(diagnostics["cuts_basket"].sum())
        coverage["mean_history_len"] = diagnostics["history_len"].mean()
        coverage["median_history_len"] = diagnostics["history_len"].median()
        coverage["truncated_query_share_L64"] = diagnostics["is_truncated"].mean()
        coverage["within_basket_cut_query_share_L64"] = diagnostics["cuts_basket"].mean()
        coverage["train_negative_rows"] = int(train["label"].eq(0).sum())
        coverage["auc_negative_rows"] = int(auc["label"].eq(0).sum())
        for name, frame in window.items():
            add(name, frame)
    output = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}
    output["item_vocab"] = vocab
    eligible_catalogs = output["queries"].loc[output["queries"]["split"].eq("train"), "catalog_id"].unique()
    in_loss = output["catalogs"].loc[output["catalogs"]["catalog_id"].isin(eligible_catalogs), "item_id"]
    support = vocab[["stock_code", "item_id"]].copy()
    support["mf_observed_train_pair"] = support["item_id"].isin(output["train_pairs"]["item_id"])
    support["sequence_in_train_catalog"] = support["item_id"].isin(in_loss)
    output["item_support"] = support
    return output
