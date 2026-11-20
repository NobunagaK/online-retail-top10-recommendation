"""One D2 data view for the four training and validation paths."""

import hashlib
from functools import cached_property
from pathlib import Path

import pandas as pd

from scripts.build_queries import verify_outputs


def _sample_ids(values, limit):
    """Fixed input order/seed across architectures and model initialization seeds."""
    def key(value):
        return hashlib.sha256(f"model-pilot-v1/{value}".encode()).digest()

    return sorted(set(values), key=key)[:limit]


class PilotData:
    def __init__(self, directory, *, smoke=False):
        self.directory = Path(directory)
        self.manifest = verify_outputs(self.directory)
        if self.manifest["splits"] != ["train", "validation"]:
            raise ValueError("This training entry point expects the development splits.")
        queries = self.read("queries")
        self.train_queries = queries.loc[queries.split.eq("train")].copy()
        self.val_queries = queries.loc[queries.split.eq("validation")].copy()
        if smoke:
            # Select training customers using training rows alone. Validation
            # then draws its customers from the resulting supported cohort.
            customers = _sample_ids(self.train_queries.customer_id, 64)
            self.train_queries = self.train_queries.loc[self.train_queries.customer_id.isin(customers)]
            eligible = self.val_queries.loc[self.val_queries.customer_id.isin(customers)]
            selected = _sample_ids(eligible.query_id, 16)
            self.val_queries = eligible.loc[eligible.query_id.isin(selected)]
        if self.train_queries.empty or self.val_queries.empty:
            raise ValueError("Training and validation both require eligible queries.")
        if not self.val_queries.history_end.eq(pd.Timestamp("2011-10-01")).all():
            raise ValueError("October validation must share the frozen October 1 history.")

        selected = pd.concat([self.train_queries, self.val_queries], ignore_index=True)
        self.vocab = self.read("item_vocab")
        self.catalogs = self.read("catalogs")
        self.catalog_ids = {
            key: rows.sort_values("StockCode").item_id.tolist()
            for key, rows in self.catalogs.groupby("catalog_id", sort=False)
        }
        self.train_pairs = self.read("train_pairs")
        self.train_pairs = self.train_pairs.loc[self.train_pairs.query_id.isin(self.train_queries.query_id)]
        labels = self.read("labels")
        labels = labels.loc[labels.in_catalog & labels.eligible & labels.query_id.isin(selected.query_id)]
        self.positives = labels.groupby("query_id").item_id.agg(list).to_dict()
        histories = self.read("histories")
        joined = selected.merge(histories, on=["customer_id", "history_end"], validate="many_to_one",
                                suffixes=("_query", "_history"))
        if len(joined) != len(selected) or not joined.history_sha256_query.eq(joined.history_sha256_history).all():
            raise ValueError("Query histories disagree with the D2 manifest.")
        self.histories_by_query = dict(zip(joined.query_id, joined.history_item_ids))
        self.auc_pairs = self.read("auc_pairs")
        self.auc_pairs = self.auc_pairs.loc[self.auc_pairs.query_id.isin(self.val_queries.query_id)]

        # Extra eligible active customers with only out-of-catalog purchases
        # receive zero supplementary recall. No catalog scoring is necessary.
        cohorts = self.read("cohorts")
        extra = cohorts.loc[cohorts.split.eq("validation") & cohorts.supplemental_eligible & ~cohorts.eligible]
        if smoke:
            # Pilot performance is defined only for the selected primary queries.
            extra = extra.iloc[:0]
        self.out_of_catalog_only = extra

    def read(self, table):
        return pd.read_parquet(self.directory / f"{table}.parquet")

    @cached_property
    def features(self):
        return {name: self.read(name) for name in
                ("customer_features", "item_features", "customer_item_features")}
