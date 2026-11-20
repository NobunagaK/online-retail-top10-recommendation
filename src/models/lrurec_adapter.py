"""Use the authors' LRURec network to rank a frozen monthly Online Retail catalog."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from src.models.lrurec_source import SOURCE_SHA256 as UPSTREAM_SHA256
from src.models.lrurec_source import UPSTREAM_COMMIT, load_lru_class


class LRURec(nn.Module):
    """Expose the upstream encoder and tied scoring head to the monthly runner."""

    def __init__(self, num_items, *, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        if min(num_items, hidden_dim, num_layers) < 1:
            raise ValueError("LRURec needs at least one item, one hidden dimension and one layer.")
        args = SimpleNamespace(num_items=num_items, bert_hidden_units=hidden_dim, bert_num_blocks=num_layers,
                               bert_dropout=dropout, bert_attn_dropout=dropout, dataset_code="retail")
        self.upstream = load_lru_class()(args)

    @property
    def item_embedding(self):
        return self.upstream.embedding.token

    @property
    def output_bias(self):
        return self.upstream.model.bias

    def encode(self, item_ids, lengths):
        """Apply the author's embedding and recurrent blocks to left-padded D2 histories."""
        if item_ids.ndim != 2 or not item_ids.shape[1] or lengths.shape != item_ids.shape[:1]:
            raise ValueError("Expected a nonempty sequence width and one length per history.")
        width = item_ids.shape[1]
        positions = torch.arange(width, device=item_ids.device)
        valid = positions >= (width - lengths[:, None])
        if (lengths.lt(0).any() or lengths.gt(width).any() or not torch.equal(valid, item_ids.ne(0))):
            raise ValueError("Each history needs zeros on the left followed by exactly its real items.")

        # The upstream blocks perform a power-of-two scan. Split the author's forward
        # at the scoring head so candidate chunks can reuse one encoded history.
        hidden, mask = self.upstream.embedding(item_ids)
        extra = (1 << (width - 1).bit_length()) - width
        hidden = F.pad(hidden, (0, 0, extra, 0))
        mask = F.pad(mask, (extra, 0))
        for block in self.upstream.model.lru_blocks:
            hidden = block(hidden, mask)
        return hidden[:, extra:].masked_fill(~valid.unsqueeze(-1), 0)

    def score_items(self, last_hidden, item_ids):
        """Apply the authors' tied dot-product head and bias on the requested rows."""
        return last_hidden @ self.item_embedding(item_ids).T + self.output_bias[item_ids]


def build_model(num_items, config):
    return LRURec(num_items, hidden_dim=config.get("hidden_dim", 64),
                  num_layers=config.get("num_layers", 2), dropout=config.get("dropout", 0.2))


def make_model(data, config, device):
    from src.models.sequence_objective import SequenceRunner

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    network = build_model(int(data.vocab["item_id"].max()), config)
    return SequenceRunner(network, data, padding="left", max_length=config.get("max_length", 64),
                          learning_rate=config.get("learning_rate", 0.001),
                          weight_decay=config.get("weight_decay", 0.01),
                          batch_size=config.get("batch_size", 64), seed=seed, device=device,
                          optimizer=config.get("optimizer", "adam"), lr_schedule=config.get("lr_schedule", "constant"),
                          epochs=config.get("epochs", 100))
