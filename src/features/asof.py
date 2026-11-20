"""LightGBM numeric features from D2 net-purchase snapshots at the query cutoff."""

import numpy as np
import pandas as pd


WINDOW_COUNTS = ("invoices_7d", "invoices_30d", "invoices_90d")
CUSTOMER = ("invoice_count", "mean_basket_size", "amount_gbp", "days_since_last_purchase", *WINDOW_COUNTS)
ITEM = ("distinct_customers", "invoice_count", "weighted_unit_price_gbp", "days_since_last_purchase", *WINDOW_COUNTS)
PAIR = ("previously_purchased", "invoice_count", "quantity", "days_since_last_purchase", *WINDOW_COUNTS)
FEATURE_NAMES = [f"{prefix}_{col}" for prefix, cols in (("customer", CUSTOMER), ("item", ITEM), ("pair", PAIR))
                 for col in cols]


class AsOfFeatures:
    """Look up frozen historical snapshots for an arbitrary ordered candidate list."""

    def __init__(self, tables, vocab):
        ids = vocab.set_index("stock_code")["item_id"]
        if not ids.index.is_unique or not ids.is_unique or not ids.gt(0).all():
            raise ValueError("The vocabulary must map distinct StockCodes to distinct positive IDs.")
        self.stock_codes = pd.Series(ids.index, index=ids.to_numpy())
        self.snapshots = {}
        for table_name, keys, cols in (
            ("customer_features", ["customer_id"], CUSTOMER),
            ("item_features", ["item_id"], ITEM),
            ("customer_item_features", ["customer_id", "item_id"], PAIR),
        ):
            for cutoff, frame in tables[table_name].groupby("history_end", sort=False):
                cutoff = pd.Timestamp(cutoff)
                observed = pd.to_datetime(frame["last_purchase_time"]).dropna()
                if not observed.lt(cutoff).all():
                    raise ValueError(f"{table_name} contains a purchase at or after its cutoff.")
                frame = frame.copy()
                if "item_id" in keys:
                    frame["item_id"] = frame["stock_code"].map(ids)
                    if frame["item_id"].isna().any():
                        raise ValueError("A D2 feature SKU is absent from the vocabulary.")
                if table_name == "item_features":
                    # Each remaining unit is valued at its original sale-line price.
                    frame["weighted_unit_price_gbp"] = frame["amount_gbp"] / frame["quantity"].where(frame["quantity"].gt(0))
                if table_name == "customer_item_features":
                    frame["previously_purchased"] = frame["invoice_count"].gt(0)
                indexed = frame.set_index(keys, verify_integrity=True)[list(cols)].astype("float32").sort_index()
                self.snapshots.setdefault(cutoff, {})[table_name] = indexed

    def matrix(self, query, item_ids):
        """One row per supplied ID, in supplied order; unobserved pair recency is missing."""
        ids = np.asarray(item_ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("Pass a one-dimensional candidate list.")
        cutoff = pd.Timestamp(query.history_end)
        tables = self.snapshots[cutoff]
        customer = tables["customer_features"].loc[query.customer_id]
        items = tables["item_features"].reindex(ids)
        if items["invoice_count"].isna().any():
            raise ValueError("Candidate has no retained history at this query's cutoff.")
        pairs = tables.get("customer_item_features")
        if pairs is None or query.customer_id not in pairs.index.levels[0]:
            pair_values = np.zeros((len(ids), len(PAIR)), dtype=np.float32)
            pair_values[:, PAIR.index("days_since_last_purchase")] = np.nan
        else:
            found = pairs.loc[query.customer_id].reindex(ids)
            pair_values = found.to_numpy(dtype=np.float32, copy=True)
            unseen = found["previously_purchased"].isna().to_numpy()
            pair_values[unseen] = 0
            pair_values[unseen, PAIR.index("days_since_last_purchase")] = np.nan
        output = np.empty((len(ids), len(FEATURE_NAMES)), dtype=np.float32)
        output[:, :len(CUSTOMER)] = customer.to_numpy()
        output[:, len(CUSTOMER):len(CUSTOMER) + len(ITEM)] = items.to_numpy()
        output[:, len(CUSTOMER) + len(ITEM):] = pair_values
        return output
