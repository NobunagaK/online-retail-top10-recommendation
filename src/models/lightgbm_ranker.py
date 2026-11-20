"""LambdaRank on the shared D2 training pairs with exact full-catalog validation."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from src.features.asof import AsOfFeatures, FEATURE_NAMES


def catalog_ndcg_at_10(labels, scores, groups):
    """Query-macro NDCG@10 for binary labels; ties preserve input StockCode order."""
    truth = np.asarray(labels)
    scores = np.asarray(scores)
    if len(truth) != len(scores) or sum(groups) != len(scores) or not len(groups):
        raise ValueError("Validation group sizes must cover every scored row.")
    if not np.isfinite(scores).all():
        raise ValueError("Validation contains nonfinite scores.")
    discount = 1 / np.log2(np.arange(2, 12))
    ideal = np.r_[0., np.cumsum(discount)]
    offset = 0
    ndcg = []
    for size in groups:
        group_labels = truth[offset:offset + size]
        positive = int(group_labels.sum())
        if not positive:
            raise ValueError("An eligible validation query must have at least one positive.")
        order = np.argsort(-scores[offset:offset + size], kind="stable")[:10]
        dcg = np.dot(group_labels[order], discount[:len(order)])
        ndcg.append(dcg / ideal[min(10, positive)])
        offset += size
    return float(np.mean(ndcg))


def _validation_metric(labels, scores, weight, group):
    """The sklearn LightGBM interface supplies four arguments to a grouped metric."""
    return "project_ndcg_at_10", catalog_ndcg_at_10(labels, scores, group), True


class LightGBMRanker:
    def __init__(self, data, config, device):
        if str(device) != "cpu":
            raise ValueError("Use the CPU runtime for the LightGBM configuration.")
        self.config = dict(config)
        self.features = AsOfFeatures(data.features, data.vocab)
        self.catalogs = {}
        for catalog_id, ids in data.catalog_ids.items():
            ids = np.asarray(ids, dtype=np.int64)
            codes = self.features.stock_codes.loc[ids]
            if codes.duplicated().any() or not len(codes):
                raise ValueError("Each historical catalog must contain distinct items.")
            self.catalogs[catalog_id] = ids[np.argsort(codes.to_numpy(), kind="stable")]
        self.booster = None
        self.best_iteration = None
        self.best_validation_ndcg = None
        self.validation_history = []

    def _training_groups(self, data):
        queries = data.train_queries.sort_values("query_id")
        pairs = data.train_pairs.loc[data.train_pairs["query_id"].isin(queries["query_id"])].copy()
        if pairs.duplicated(["query_id", "item_id"]).any() or not pairs["label"].isin([0, 1]).all():
            raise ValueError("Shared training pairs must be unique and binary within each query.")
        pairs["stock_code"] = pairs["item_id"].map(self.features.stock_codes)
        pairs = pairs.sort_values(["query_id", "stock_code"]).groupby("query_id", sort=False)
        for query in queries.itertuples(index=False):
            group = pairs.get_group(query.query_id)
            ids = group["item_id"].to_numpy(dtype=np.int64)
            labels = group["label"].to_numpy(dtype=np.int8)
            if (not np.isin(ids, self.catalogs[query.catalog_id]).all() or not labels.any()
                    or set(ids[labels == 1]) != set(data.positives[query.query_id])):
                raise ValueError("Shared training pairs disagree with their historical catalog or positive labels.")
            yield query, ids, labels

    def _validation_groups(self, data):
        for query in data.val_queries.sort_values("query_id").itertuples(index=False):
            ids = self.catalogs[query.catalog_id]
            positives = set(data.positives[query.query_id])
            if not positives or not positives.issubset(ids):
                raise ValueError("Eligible validation positives must be in the shared catalog.")
            yield query, ids, np.isin(ids, list(positives)).astype(np.int8)

    def _matrix(self, groups):
        groups = list(groups)
        if not groups:
            raise ValueError("LightGBM requires eligible training and validation queries.")
        sizes = np.asarray([len(ids) for _, ids, _ in groups], dtype=np.int32)
        features = np.empty((int(sizes.sum()), len(FEATURE_NAMES)), dtype=np.float32)
        labels = np.empty(len(features), dtype=np.int8)
        offset = 0
        for query, ids, truth in groups:
            rows = slice(offset, offset + len(ids))
            features[rows] = self.features.matrix(query, ids)
            labels[rows] = truth
            offset += len(ids)
        return features, labels, sizes

    def train_epoch(self, data):
        """Fit one tree ensemble; return best full-catalog validation macro NDCG@10."""
        import lightgbm as lgb

        if self.booster is not None:
            raise RuntimeError("This LightGBM ensemble is already fitted.")
        train_x, train_y, train_group = self._matrix(self._training_groups(data))
        val_x, val_y, val_group = self._matrix(self._validation_groups(data))
        cfg = self.config
        ranker = lgb.LGBMRanker(
            objective="lambdarank", metric="None", deterministic=True, force_col_wise=True,
            n_estimators=int(cfg.get("n_estimators", 1000)), num_leaves=int(cfg.get("num_leaves", 31)),
            learning_rate=float(cfg.get("learning_rate", .05)), min_child_samples=int(cfg.get("min_child_samples", 20)),
            reg_lambda=float(cfg.get("reg_lambda", 0)), random_state=int(cfg.get("seed", 42)),
            n_jobs=int(cfg.get("n_jobs", 2)), verbosity=-1,
        )
        evaluations = {}
        callbacks = [
            lgb.record_evaluation(evaluations),
            lgb.early_stopping(int(cfg.get("patience", 50)), first_metric_only=True, verbose=False),
        ]
        ranker.fit(
            train_x, train_y, group=train_group, eval_set=[(val_x, val_y)], eval_group=[val_group],
            eval_names=["validation"], eval_at=(10,), eval_metric=_validation_metric,
            feature_name=FEATURE_NAMES, callbacks=callbacks,
        )
        metric = evaluations["validation"]
        if set(metric) != {"project_ndcg_at_10"}:
            raise RuntimeError("Early stopping must monitor exactly one project validation metric.")
        self.validation_history = [
            {"iteration": i, "ndcg_at_10": float(value)}
            for i, value in enumerate(metric["project_ndcg_at_10"], start=1)
        ]
        self.booster = ranker.booster_
        self.best_iteration = int(ranker.best_iteration_ or self.booster.current_iteration())
        self.best_validation_ndcg = self.validation_history[self.best_iteration - 1]["ndcg_at_10"]
        return self.best_validation_ndcg

    def score(self, query, item_ids):
        """Return raw rank scores in the caller's original item order."""
        if self.booster is None:
            raise RuntimeError("Fit or restore a native LightGBM model before scoring.")
        features = self.features.matrix(query, item_ids)
        return self.booster.predict(features, num_iteration=self.best_iteration, raw_score=True,
                                    num_threads=int(self.config.get("n_jobs", 2)))

    def save(self, path):
        """Atomically save the native text model with only the selected trees."""
        if self.booster is None:
            raise RuntimeError("No fitted LightGBM ensemble to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            temporary = Path(temp.name)
        try:
            self.booster.save_model(str(temporary), num_iteration=self.best_iteration)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def restore(self, path):
        """Load the native selected tree prefix for evaluation on the current D2 snapshots."""
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(path))
        if booster.feature_name() != FEATURE_NAMES:
            raise ValueError("Native model feature order differs from this project's LightGBM feature dictionary.")
        self.booster = booster
        self.best_iteration = booster.current_iteration()
        self.best_validation_ndcg = None
        self.validation_history = []
        return self


def make_model(data, config, device="cpu"):
    return LightGBMRanker(data, config, device)
