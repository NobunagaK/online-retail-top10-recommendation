"""Implicit-feedback matrix factorization on the shared D2 customer–month queries."""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _TrainingPairs:
    """Store contiguous D2 groups so shuffling and batching never split a query."""

    def __init__(self, data, customer_index, num_items):
        queries = data.train_queries.set_index("query_id", verify_integrity=True)
        pairs = data.train_pairs.sort_values(["query_id", "item_id"], ignore_index=True)
        sizes = pairs.groupby("query_id", sort=False).size()
        if sizes.empty or set(sizes.index) != set(queries.index):
            raise ValueError("Every selected training query must have a nonempty pair group")
        if pairs.duplicated(["query_id", "item_id"]).any():
            raise ValueError("A training query must contain each sampled item at most once")
        if not pairs["label"].isin([0, 1]).all():
            raise ValueError("Training labels must be binary")
        if not pairs["item_id"].isin(range(1, num_items + 1)).all():
            raise ValueError("Training pairs must contain positive D2 vocabulary IDs")
        lengths = sizes.to_numpy(dtype=np.int64)
        users = queries.loc[sizes.index, "customer_id"].astype(str).map(customer_index)
        self.offsets = np.r_[0, lengths.cumsum()]
        self.users = torch.from_numpy(np.repeat(users.to_numpy(dtype=np.int64), lengths))
        self.items = torch.from_numpy(pairs["item_id"].to_numpy(dtype=np.int64, copy=True))
        self.labels = torch.from_numpy(pairs["label"].to_numpy(dtype=np.float32, copy=True))
        self.weights = torch.from_numpy(np.repeat(1.0 / lengths, lengths).astype(np.float32))
        self.item_ids = self.items.unique()

        # Reject a resume with different groups, sampled pairs, labels or user assignments.
        digest = hashlib.sha256()
        for query_id in sizes.index:
            digest.update(str(query_id).encode("utf-8") + b"\0")
        for array in (self.offsets, self.users.numpy(), self.items.numpy(), self.labels.numpy()):
            digest.update(array.tobytes())
        self.signature = digest.hexdigest()

    def batches(self, batch_size, generator):
        order = torch.randperm(len(self.offsets) - 1, generator=generator).tolist()
        indices, count = [], 0
        for group in order:
            start, end = self.offsets[group:group + 2]
            if indices and count + end - start > batch_size:
                yield np.concatenate(indices), len(indices)
                indices, count = [], 0
            indices.append(np.arange(start, end))
            count += end - start
        if indices:
            yield np.concatenate(indices), len(indices)


class _Factors(nn.Module):
    def __init__(self, num_users, num_items, dim, generator):
        super().__init__()
        # Initialize on CPU with a private generator; the selected runtime can then move the tables.
        def embedding(rows):
            weights = torch.empty(rows, dim).normal_(mean=0, std=0.05, generator=generator)
            return nn.Embedding.from_pretrained(weights, freeze=False)

        self.user = embedding(num_users)
        self.item = embedding(num_items + 1)
        self.user_bias = nn.Embedding.from_pretrained(torch.zeros(num_users, 1), freeze=False)
        self.item_bias = nn.Embedding.from_pretrained(torch.zeros(num_items + 1, 1), freeze=False)
        self.register_buffer("supported", torch.zeros(num_items + 1, dtype=torch.bool))

    def objective(self, users, items, labels, weights, num_queries, l2):
        """Average each query's BCE and sampled L2 penalty, then average the queries."""
        user, item = self.user(users), self.item(items)
        user_bias, item_bias = self.user_bias(users).flatten(), self.item_bias(items).flatten()
        logits = (user * item).sum(dim=1) + user_bias + item_bias
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        squared_norm = user.square().sum(dim=1) + item.square().sum(dim=1)
        squared_norm = squared_norm + user_bias.square() + item_bias.square()
        return ((losses + 0.5 * l2 * squared_norm) * weights).sum() / num_queries


class MatrixFactorization:
    """Train on sampled D2 pairs and score an ordered list of full-catalog candidates."""

    def __init__(self, data, config, device):
        seed = int(config.get("seed", 42))
        dim = int(config.get("embedding_dim", 32))
        learning_rate = float(config.get("learning_rate", 0.001))
        self.l2 = float(config.get("l2", 1e-5))
        self.batch_size = int(config.get("batch_size", 1024))
        if dim <= 0 or self.batch_size <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("embedding_dim, batch_size and learning_rate must be positive")
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be finite and nonnegative")

        vocab = data.vocab.sort_values("item_id")
        if not np.array_equal(vocab["item_id"], np.arange(1, len(vocab) + 1)):
            raise ValueError("D2 item IDs must be consecutive from 1; ID 0 is reserved for PAD")
        self.stock_codes = vocab["stock_code"].astype(str).tolist()
        self.customers = sorted(set(data.train_queries["customer_id"].astype(str)))
        self.customer_index = {customer: row for row, customer in enumerate(self.customers)}
        self._data = data
        self._pairs = _TrainingPairs(data, self.customer_index, len(vocab))
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.factors = _Factors(len(self.customers), len(vocab), dim, self.generator).to(self.device)
        self.optimizer = torch.optim.Adam(self.factors.parameters(), lr=learning_rate)
        self.completed_epochs = 0
        self._item_means = None

    @property
    def supported_item_ids(self):
        """IDs receiving embedding and bias gradients through a completed training epoch."""
        return set(self.factors.supported.nonzero().flatten().tolist())

    def train_epoch(self, data):
        if data is not self._data:
            raise ValueError("Pass the D2 data used to construct this model")
        self.factors.train()
        self._item_means = None
        pairs = self._pairs
        total_loss, total_queries = 0.0, 0
        for indices, num_queries in pairs.batches(self.batch_size, self.generator):
            users, items, labels, weights = (
                field[indices].to(self.device) for field in
                (pairs.users, pairs.items, pairs.labels, pairs.weights)
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.factors.objective(users, items, labels, weights, num_queries, self.l2)
            loss_value = loss.detach().item()
            if not np.isfinite(loss_value):
                raise FloatingPointError("Nonfinite MF training loss")
            loss.backward()
            self.optimizer.step()
            total_loss += loss_value * num_queries
            total_queries += num_queries
        self.factors.supported[pairs.item_ids.to(self.device)] = True
        self.completed_epochs += 1
        return total_loss / total_queries

    @torch.no_grad()
    def score(self, query, item_ids):
        """Return raw logits in requested order, including candidates previously purchased."""
        candidates = np.asarray(item_ids)
        if candidates.size == 0 and candidates.ndim == 1:
            return np.empty(0, dtype=np.float32)
        if (candidates.ndim != 1 or candidates.dtype.kind not in "iu"
                or candidates.min() < 1 or candidates.max() > len(self.stock_codes)):
            raise ValueError("Scoring requires a list of positive D2 item IDs")
        if self.completed_epochs == 0:
            return np.zeros(len(candidates), dtype=np.float32)
        self.factors.eval()
        factors = self.factors
        if self._item_means is None:
            mask = factors.supported
            self._item_means = (
                factors.item.weight[mask].mean(dim=0),
                factors.item_bias.weight[mask].mean(),
            )
        row = self.customer_index.get(str(query.customer_id))
        if row is None:
            # Possible in a pilot using only a subset of the shared training groups.
            user, user_bias = factors.user.weight.mean(dim=0), factors.user_bias.weight.mean()
        else:
            user, user_bias = factors.user.weight[row], factors.user_bias.weight[row, 0]
        ids = torch.as_tensor(candidates.astype(np.int64, copy=False), device=self.device)
        supported_logits = factors.item(ids).matmul(user) + factors.item_bias(ids).flatten() + user_bias
        mean_embedding, mean_bias = self._item_means
        fallback = mean_embedding.dot(user) + mean_bias + user_bias
        return torch.where(factors.supported[ids], supported_logits, fallback).cpu().numpy()

    def save(self, path):
        """Save model, Adam moments and private shuffle RNG at the completed epoch boundary."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "factors": self.factors.state_dict(), "optimizer": self.optimizer.state_dict(),
            "completed_epochs": self.completed_epochs, "rng_state": self.generator.get_state(),
            "customers": self.customers, "stock_codes": self.stock_codes,
            "pair_signature": self._pairs.signature, "l2": self.l2, "batch_size": self.batch_size,
        }
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def restore(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["customers"] != self.customers or checkpoint["stock_codes"] != self.stock_codes
                or checkpoint["pair_signature"] != self._pairs.signature):
            raise ValueError("Checkpoint and selected D2 training pairs or ID mappings differ")
        self.factors.load_state_dict(checkpoint["factors"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.generator.set_state(checkpoint["rng_state"])
        self.completed_epochs = checkpoint["completed_epochs"]
        self.batch_size, self.l2 = checkpoint["batch_size"], checkpoint["l2"]
        self._item_means = None


def make_model(data, config, device="cpu"):
    return MatrixFactorization(data, config, device)
