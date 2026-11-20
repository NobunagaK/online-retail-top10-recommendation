"""Monthly full-catalog training and checkpointing shared by both sequence models."""

import os
from numbers import Integral
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch import nn
from torch.nn import functional as F

from src.data.sequences import prepare_sequence


def _real_ids(values):
    if torch.is_tensor(values):
        if values.ndim != 1:
            raise ValueError("Expected a one-dimensional list of item IDs.")
        values = values.tolist()
    ids = list(values)
    if any(isinstance(i, bool) or not isinstance(i, Integral) or i <= 0 for i in ids):
        raise ValueError("Item IDs must be positive integers. Zero is reserved for padding.")
    return [int(i) for i in ids]


def multi_positive_loss(logits, catalog_ids, positives):
    """Mean over queries of mean negative log probability of distinct positive items.

    Columns of logits must be exactly the supplied query catalog, in its given
    order. Labels outside that catalog cannot enter the loss. Repeated labels
    within one query count once. A supervised query needs an in-catalog target.
    """
    catalog = _real_ids(catalog_ids)
    if not catalog or len(set(catalog)) != len(catalog):
        raise ValueError("A training catalog must have distinct, real items.")
    if logits.shape != (len(positives), len(catalog)) or not len(positives):
        raise ValueError("Expected one logit per query and candidate in the shared catalog.")
    columns = {item: j for j, item in enumerate(catalog)}
    rows, positions, counts = [], [], []
    for row, labels in enumerate(positives):
        selected = sorted({columns[item] for item in _real_ids(labels) if item in columns})
        if not selected:
            raise ValueError("A training query has no in-catalog positive.")
        rows.extend([row] * len(selected))
        positions.extend(selected)
        counts.append(len(selected))
    # Accumulate half-precision logits in fp32 and keep double precision for audits.
    dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    logp = F.log_softmax(logits, dim=1, dtype=dtype)
    row_idx = torch.tensor(rows, dtype=torch.long, device=logits.device)
    col_idx = torch.tensor(positions, dtype=torch.long, device=logits.device)
    per_query_counts = logp.new_tensor(counts)
    return -(logp[row_idx, col_idx] / per_query_counts[row_idx]).sum() / len(positives)


def last_valid_state(states, lengths, padding):
    """Gather one state per customer. An empty history has an explicit zero state."""
    if padding not in {"left", "right"} or states.ndim != 3 or lengths.shape != states.shape[:1]:
        raise ValueError("Expected [batch, time, hidden] states and one length per row.")
    if states.shape[1] == 0 or (lengths < 0).any() or (lengths > states.shape[1]).any():
        raise ValueError("A history length lies outside the padded sequence.")
    last = (lengths - 1).clamp_min(0) if padding == "right" else torch.full_like(lengths, states.shape[1] - 1)
    selected = states[torch.arange(len(lengths), device=states.device), last]
    return torch.where((lengths > 0).unsqueeze(1), selected, torch.zeros_like(selected))


class _SequenceScores(nn.Module):
    """Expose encoding and scoring in a single stateless inference call."""

    def __init__(self, network, padding):
        super().__init__()
        self.network = network
        self.padding = padding

    def forward(self, sequence, lengths, items, state=None):
        if state is None:
            states = self.network.encode(sequence, lengths)
            # Mamba4Rec's own forward already selects the last real item.
            state = states if states.ndim == 2 else last_valid_state(states, lengths, self.padding)
            if state.shape != (len(lengths), self.network.item_embedding.embedding_dim):
                raise ValueError("The encoder must return one final hidden vector per sequence.")
        return self.network.score_items(state, items), state


def _mean_substitution(parameter, support):
    support = support.to(parameter.device)
    if not support.any():
        raise RuntimeError("No supported item rows. Train or restore the model before scoring.")
    effective = parameter.detach().clone()
    unsupported = ~support
    unsupported[0] = False
    effective[unsupported] = parameter.detach()[support].mean(dim=0)
    return effective


class SequenceRunner:
    """Optimize a tied-embedding sequence network on the shared monthly D2 tables."""

    def __init__(self, network, data, *, padding, max_length=64, learning_rate=1e-3,
                 weight_decay=0.0, batch_size=32, seed=42, device="cpu",
                 optimizer="adam", lr_schedule="constant", epochs=100):
        if padding not in {"left", "right"} or min(max_length, batch_size) < 1:
            raise ValueError("Specify left/right padding and positive batch and sequence lengths.")
        self.data = data
        self.padding = padding
        self.max_length = max_length
        self.batch_size = batch_size
        self.seed = int(seed)
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self._scores = _SequenceScores(self.network, padding)
        # The adapters expose the authors' original parameters through properties.
        # functional_call needs their registered paths inside the scoring module.
        parameter_names = {id(param): name for name, param in self._scores.named_parameters()}
        self._embedding_path = parameter_names[id(self.network.item_embedding.weight)]
        bias = getattr(self.network, "output_bias", None)
        self._bias_path = parameter_names[id(bias)] if bias is not None else None
        self.training_policy = {"optimizer": optimizer, "learning_rate": learning_rate,
                                "weight_decay": weight_decay, "lr_schedule": lr_schedule}
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adamw":
            decay, exempt, exempt_names = [], [], []
            for prefix, module in self.network.named_modules():
                for name, param in module.named_parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    # A_log and D are exempt in the authors' Mamba implementation.
                    if (isinstance(module, nn.LayerNorm) or name in {"bias", "params_log", "A_log", "D"}
                            or name.endswith("_bias") or getattr(param, "_no_weight_decay", False)):
                        exempt.append(param)
                        exempt_names.append(f"{prefix}.{name}".lstrip("."))
                    else:
                        decay.append(param)
            groups = [{"params": decay, "weight_decay": weight_decay}, {"params": exempt, "weight_decay": 0.0}]
            self.optimizer = torch.optim.AdamW(groups, lr=learning_rate)
            self.training_policy["decay_exempt_parameters"] = exempt_names
        else:
            raise ValueError(f"Unknown sequence optimizer: {optimizer}")
        self.scheduler = None
        if lr_schedule == "cosine" and epochs > 0:
            self.training_policy["schedule_epochs"] = epochs
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif lr_schedule != "constant":
            raise ValueError("Use constant or cosine with a positive epoch budget.")
        self._order = torch.Generator(device="cpu").manual_seed(self.seed)
        self.epoch = 0
        self.last_grad_norm = self.last_encoder_grad_norm = self.last_head_grad_norm = 0.0
        rows = self.network.item_embedding.num_embeddings
        if rows != int(data.vocab["item_id"].max()) + 1:
            raise ValueError("The shared item embedding must have one row per vocabulary ID plus PAD.")
        self.input_support = torch.zeros(rows, dtype=torch.bool)
        self.catalog_support = torch.zeros_like(self.input_support)
        self._clear_inference()

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    @property
    def real_parameter_count(self):
        """Count real scalar components, including both parts of complex LRU weights."""
        return sum(p.numel() * (2 if p.is_complex() else 1)
                   for p in self.network.parameters() if p.requires_grad)

    @property
    def supported_items(self):
        return set((self.input_support | self.catalog_support).nonzero().flatten().tolist())

    @property
    def supported_item_ids(self):
        return self.supported_items

    def support_summary(self):
        summary = {"history_input_rows": int(self.input_support.sum()),
                   "catalog_loss_rows": int(self.catalog_support.sum()),
                   "tied_item_embedding_rows": len(self.supported_items)}
        if getattr(self.network, "output_bias", None) is not None:
            summary["item_bias_rows"] = int(self.catalog_support.sum())
        return summary

    def _clear_inference(self):
        self._inference_weights = None
        self._cached_query_id = self._cached_state = None

    def _history_batch(self, data, query_ids):
        sequences, lengths = [], []
        for qid in query_ids:
            ids, n, _ = prepare_sequence(data.histories_by_query[qid], self.max_length, self.padding)
            sequences.append(ids)
            lengths.append(n)
        return (torch.tensor(sequences, dtype=torch.long, device=self.device),
                torch.tensor(lengths, dtype=torch.long, device=self.device))

    def _batches(self, data):
        queries = data.train_queries
        if queries.empty or queries["query_id"].duplicated().any():
            raise ValueError("Expected at least one distinct training query.")
        if "split" in queries and not queries["split"].eq("train").all():
            raise ValueError("Only training queries may enter an optimization batch.")
        batches = []
        for catalog_id, group in queries.groupby("catalog_id", sort=True):
            ordered = group["query_id"].iloc[torch.randperm(len(group), generator=self._order).tolist()].tolist()
            for start in range(0, len(ordered), self.batch_size):
                batches.append((catalog_id, ordered[start:start + self.batch_size]))
        return [batches[i] for i in torch.randperm(len(batches), generator=self._order).tolist()]

    def _gradient_norms(self):
        head = {id(self.network.item_embedding.weight), id(getattr(self.network, "output_bias", None))}
        encoder_norms, head_norms = [], []
        for parameter in self.network.parameters():
            if parameter.grad is not None:
                group = head_norms if id(parameter) in head else encoder_norms
                group.append(torch.linalg.vector_norm(parameter.grad.detach()))
        zero = torch.zeros((), device=self.device)
        encoder = torch.linalg.vector_norm(torch.stack(encoder_norms)) if encoder_norms else zero
        scoring = torch.linalg.vector_norm(torch.stack(head_norms)) if head_norms else zero
        total = torch.linalg.vector_norm(torch.stack([encoder, scoring]))
        norms = torch.stack([total, encoder, scoring])
        if not norms.isfinite().all():
            raise FloatingPointError("Nonfinite gradient in the sequence encoder or scoring head.")
        return norms

    def train_epoch(self, data=None):
        """Fit every supplied training query once; return the query-weighted mean loss."""
        data = self.data if data is None else data
        self._clear_inference()
        self.network.train()
        batches = self._batches(data)
        epoch_seed = int(torch.randint(2**31, (1,), generator=self._order).item())
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        total_loss = 0.0
        max_norms = torch.zeros(3, device=self.device)
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(epoch_seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(epoch_seed)
            for catalog_id, query_ids in batches:
                catalog = _real_ids(data.catalog_ids[catalog_id])
                items = torch.tensor(catalog, dtype=torch.long, device=self.device)
                sequence, lengths = self._history_batch(data, query_ids)
                self.optimizer.zero_grad(set_to_none=True)
                logits, _ = self._scores(sequence, lengths, items)
                loss = multi_positive_loss(logits, catalog, [data.positives[qid] for qid in query_ids])
                if not loss.isfinite():
                    raise FloatingPointError("Nonfinite monthly sequence loss.")
                loss.backward()
                max_norms = torch.maximum(max_norms, self._gradient_norms())
                self.optimizer.step()
                # Only count the supervised rows and history positions actually used.
                seen = sequence.detach().unique().cpu()
                self.input_support[seen[seen > 0]] = True
                self.catalog_support[catalog] = True
                total_loss += float(loss.detach()) * len(query_ids)
        self.epoch += 1
        if self.scheduler is not None:
            self.scheduler.step()
        self.last_grad_norm, self.last_encoder_grad_norm, self.last_head_grad_norm = max_norms.tolist()
        return total_loss / len(data.train_queries)

    @torch.no_grad()
    def score(self, query, item_ids):
        """Return a 1-D array of raw candidate scores in the caller's order."""
        self.network.eval()
        ids = _real_ids(item_ids)
        if not set(ids).issubset(self.data.catalog_ids[query.catalog_id]):
            raise ValueError("Requested scores must belong to the query's cutoff-specific catalog.")
        if self._inference_weights is None:
            self._inference_weights = {self._embedding_path: _mean_substitution(
                self.network.item_embedding.weight, self.input_support | self.catalog_support)}
            if self._bias_path is not None:
                self._inference_weights[self._bias_path] = _mean_substitution(
                    self.network.output_bias, self.catalog_support)
        state = self._cached_state if query.query_id == self._cached_query_id else None
        sequence, lengths = (None, None) if state is not None else self._history_batch(self.data, [query.query_id])
        items = torch.tensor(ids, dtype=torch.long, device=self.device)
        logits, state = torch.func.functional_call(self._scores, self._inference_weights,
                                                  (sequence, lengths, items, state))
        if logits.shape != (1, len(ids)) or not logits.isfinite().all():
            raise ValueError("The sequence network must return one finite score per requested item.")
        self._cached_query_id, self._cached_state = query.query_id, state
        return logits[0].to(device="cpu", dtype=torch.float64).numpy()

    def _settings(self):
        return {"network": type(self.network).__name__, "embedding_path": self._embedding_path,
                "bias_path": self._bias_path, "padding": self.padding,
                "max_length": self.max_length, "batch_size": self.batch_size,
                "vocab": self.data.vocab.sort_values("item_id")[["item_id", "stock_code"]].values.tolist()}

    def save(self, path):
        """Atomically replace a checkpoint containing optimizer, support and epoch-order state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"format": 1, "settings": self._settings(), "network": self.network.state_dict(),
                      "training_policy": self.training_policy,
                      "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                      "optimizer": self.optimizer.state_dict(), "input_support": self.input_support,
                      "catalog_support": self.catalog_support, "order": self._order.get_state(),
                      "epoch": self.epoch, "seed": self.seed, "grad_norms": (self.last_grad_norm,
                      self.last_encoder_grad_norm, self.last_head_grad_norm)}
        with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def restore(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["format"] != 1 or checkpoint["settings"] != self._settings():
            raise ValueError("The checkpoint has incompatible vocabulary, network or batching settings.")
        policy = checkpoint.get("training_policy")
        if policy is None:
            # Pre-D6 checkpoints used one coupled-Adam group without a scheduler.
            group, = checkpoint["optimizer"]["param_groups"]
            policy = {"optimizer": "adam", "learning_rate": group["lr"],
                      "weight_decay": group["weight_decay"], "lr_schedule": "constant"}
        if policy != self.training_policy:
            raise ValueError("The checkpoint has an incompatible optimizer or learning-rate schedule.")
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.input_support = checkpoint["input_support"]
        self.catalog_support = checkpoint["catalog_support"]
        self._order.set_state(checkpoint["order"])
        self.epoch, self.seed = checkpoint["epoch"], checkpoint["seed"]
        self.last_grad_norm, self.last_encoder_grad_norm, self.last_head_grad_norm = checkpoint["grad_norms"]
        self._clear_inference()
        return self
