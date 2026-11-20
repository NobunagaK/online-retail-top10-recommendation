"""LRURec for a frozen monthly catalog, from the recurrence of Yue et al. (WSDM 2024)."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LinearRecurrentBlock(nn.Module):
    """Diagonal complex recurrence, followed by two position-wise residual sublayers."""

    def __init__(self, dim, dropout):
        super().__init__()
        state_dim = 2 * dim
        radius_squared = torch.empty(state_dim).uniform_(0.8 ** 2, 0.99 ** 2)
        phase = (2 * math.pi * torch.rand(state_dim)).clamp_min(torch.finfo(torch.float32).tiny)
        # Exponentiating the first two rows yields a positive decay and oscillation frequency.
        self.log_dynamics = nn.Parameter(torch.stack((
            (-0.5 * radius_squared.log()).log(), phase.log(), 0.5 * torch.log1p(-radius_squared),
        )))
        self.input_projection = nn.Linear(dim, state_dim, dtype=torch.complex64)
        self.output_projection = nn.Linear(state_dim, dim, dtype=torch.complex64)
        self.recurrent_norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 4 * dim)
        self.contract = nn.Linear(4 * dim, dim)
        self.feedforward_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, valid):
        decay, phase, scale = self.log_dynamics.exp().unbind()
        transition = torch.complex(-decay, phase).exp()
        drive = self.input_projection(tokens.to(self.input_projection.weight.dtype)) * scale

        # Inclusive prefix scan. After each doubling, state[t] contains the last
        # 2*stride inputs, each weighted by the corresponding transition power.
        state = drive.masked_fill(~valid.unsqueeze(-1), 0)
        stride = 1
        while stride < tokens.size(1):
            combined = state[:, stride:] + transition * state[:, :-stride]
            state = torch.cat((state[:, :stride], combined), dim=1)
            transition = transition.square()
            stride *= 2

        residual = self.recurrent_norm(tokens + self.dropout(self.output_projection(state).real))
        updates = self.contract(self.dropout(F.gelu(self.expand(residual))))
        return self.feedforward_norm(residual + self.dropout(updates)).masked_fill(~valid.unsqueeze(-1), 0)


class LRURec(nn.Module):
    """Encode left-padded histories and score every requested item with the tied head."""

    def __init__(self, num_items, *, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        if min(num_items, hidden_dim, num_layers) < 1:
            raise ValueError("LRURec needs at least one item, one hidden dimension and one layer.")
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([LinearRecurrentBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.output_bias = nn.Parameter(torch.empty(num_items + 1))
        self._initialize_weights()

    @torch.no_grad()
    def _initialize_weights(self):
        # The source model initializes both complex components with the same
        # truncated normal used for real projections, item embeddings and bias.
        for component in self.modules():
            if isinstance(component, (nn.Embedding, nn.Linear)):
                for parameter in component.parameters(recurse=False):
                    parts = (parameter.real, parameter.imag) if parameter.is_complex() else (parameter,)
                    for part in parts:
                        nn.init.trunc_normal_(part, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.output_bias, std=0.02, a=-0.04, b=0.04)
        self.item_embedding.weight[0].zero_()
        self.output_bias[0].zero_()

    def encode(self, item_ids, lengths):
        """Return [batch, sequence length, hidden dimension]; padding positions are zero."""
        if item_ids.ndim != 2 or lengths.shape != item_ids.shape[:1]:
            raise ValueError("Expected a matrix of left-padded IDs and one length per history.")
        valid = item_ids.ne(0)
        tokens = self.input_norm(self.input_dropout(self.item_embedding(item_ids)))
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0)
        for block in self.blocks:
            tokens = block(tokens, valid)
        return tokens

    def score_items(self, last_hidden, item_ids):
        """Raw scores for [batch, catalog] before Top-10 selection or AUC sampling."""
        return F.linear(last_hidden, self.item_embedding(item_ids), self.output_bias[item_ids])


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
                          batch_size=config.get("batch_size", 64), seed=seed, device=device)
