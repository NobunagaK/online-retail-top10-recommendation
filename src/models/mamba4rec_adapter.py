"""Mamba4Rec (Liu et al., 2024) adapted to the project's monthly catalog and shared loss."""

import argparse
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess


UPSTREAM_COMMIT = "44a0c8432b34e3b89d46035b10e8525b1bd368dc"


def _preflight_cli():
    parser = argparse.ArgumentParser(description="Check one native Mamba4Rec CUDA update on synthetic histories.")
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.add_argument("--output", type=Path, help="Save the measured environment and check results as JSON")
    args = parser.parse_args()
    report = {"upstream_commit": UPSTREAM_COMMIT, "python": platform.python_version(),
              "platform": platform.platform(), "packages": {}}
    for package in ["torch", "mamba-ssm", "causal-conv1d", "triton", "transformers"]:
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["packages"][package] = None
    for label, command in [("nvidia_driver", ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
                           ("nvcc_version", ["nvcc", "--version"])]:
        report[label] = None
        if shutil.which(command[0]):
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    report[label] = result.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
    try:
        # Defer GPU imports so even a missing Torch or broken native extension is recorded.
        import torch
        from src.models.mamba4rec_adapter import run_gpu_preflight

        report["torch_cuda"] = torch.version.cuda
        report["cxx11_abi"] = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name()
            report["compute_capability"] = list(torch.cuda.get_device_capability())
            report["device_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report.update(run_gpu_preflight())
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    document = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document, encoding="utf-8")
    print(document, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(_preflight_cli())


import torch
from torch import nn


def _native_mamba():
    """Require the authors' Mamba-1 path and both native operations used by its fused block."""
    try:
        from mamba_ssm.modules import mamba_simple
        from mamba_ssm.ops import selective_scan_interface
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Cannot load native Mamba. Check the CUDA, PyTorch, Python and C++ ABI wheel tags. "
            "See docs/mamba4rec_adaptation.md."
        ) from error
    if (mamba_simple.causal_conv1d_fn is None
            or getattr(selective_scan_interface, "causal_conv1d_fwd_function", None) is None
            or getattr(selective_scan_interface, "selective_scan_cuda", None) is None):
        raise RuntimeError("Mamba selective scan and native causal-conv1d must both be available.")
    return mamba_simple.Mamba


class MambaBlock(nn.Module):
    """Selective state-space block and the position-wise feed-forward block in Mamba4Rec."""

    def __init__(self, operator, dim, layers, dropout, d_state, d_conv, expand):
        super().__init__()
        self.mamba = operator(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand,
                              use_fast_path=True)
        self.stacked = layers > 1
        self.dropout = nn.Dropout(dropout)
        self.mixer_norm = nn.LayerNorm(dim, eps=1e-12)
        self.expand = nn.Linear(dim, 4 * dim)
        self.contract = nn.Linear(4 * dim, dim)
        self.activation = nn.GELU()
        self.output_norm = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, inputs):
        mixed = self.dropout(self.mamba(inputs))
        if self.stacked:
            mixed = mixed + inputs
        hidden = self.mixer_norm(mixed)
        feedforward = self.dropout(self.activation(self.expand(hidden)))
        return self.output_norm(hidden + self.dropout(self.contract(feedforward)))


class Mamba4Rec(nn.Module):
    """Causal network with a tied item embedding and the original dot-product scoring head."""

    def __init__(self, num_items, *, hidden_dim=64, num_layers=2, dropout=0.2,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        if min(num_items, hidden_dim, num_layers, d_state, expand) < 1 or d_conv not in (2, 3, 4):
            raise ValueError("Use positive model dimensions and a native causal kernel width of 2, 3 or 4.")
        operator = _native_mamba()
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.blocks = nn.ModuleList(
            MambaBlock(operator, hidden_dim, num_layers, dropout, d_state, d_conv, expand)
            for _ in range(num_layers)
        )
        self.apply(self._init_weights)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    @staticmethod
    def _init_weights(module):
        # Match the recommender's initialization after constructing each official Mamba block.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def encode(self, item_ids, lengths):
        """Return every causal hidden state. The shared runner selects length - 1 for nonempty rows."""
        if item_ids.ndim != 2 or lengths.shape != item_ids.shape[:1]:
            raise ValueError("Supply a right-padded ID matrix with one valid length per history.")
        if not item_ids.is_cuda:
            raise RuntimeError("Mamba4Rec requires a CUDA device and the native Mamba kernels.")
        positions = torch.arange(item_ids.size(1), device=item_ids.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        if lengths.lt(0).any() or lengths.gt(item_ids.size(1)).any() or not torch.equal(valid, item_ids.ne(0)):
            raise ValueError("Each Mamba4Rec history requires its real items first, then zero padding.")
        hidden = self.input_norm(self.input_dropout(self.item_embedding(item_ids)))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden.masked_fill(~valid.unsqueeze(-1), 0)

    def score_items(self, last_hidden, item_ids):
        return last_hidden @ self.item_embedding(item_ids).T


def build_model(num_items, config):
    return Mamba4Rec(num_items, hidden_dim=config.get("hidden_dim", 64), num_layers=config.get("num_layers", 2),
                     dropout=config.get("dropout", 0.2), d_state=config.get("d_state", 16),
                     d_conv=config.get("d_conv", 4), expand=config.get("expand", 2))


def make_model(data, config, device):
    from src.models.sequence_objective import SequenceRunner

    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Select an NVIDIA GPU runtime for Mamba4Rec and run its native preflight.")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    network = build_model(int(data.vocab["item_id"].max()), config)
    return SequenceRunner(network, data, padding="right", max_length=config.get("max_length", 64),
                          learning_rate=config.get("learning_rate", 0.001),
                          weight_decay=config.get("weight_decay", 0.01),
                          batch_size=config.get("batch_size", 64), seed=seed, device=device)


def run_gpu_preflight():
    """Prove native forward/backward, encoder and output-row updates, right padding and reload."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run the preflight in an NVIDIA GPU runtime.")
    from src.models.sequence_objective import last_valid_state, multi_positive_loss

    device = torch.device("cuda")
    torch.manual_seed(42)
    model = Mamba4Rec(5, hidden_dim=32, d_state=16, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    ids = torch.tensor([[1, 2, 1, 0, 0], [2, 3, 0, 0, 0]], device=device)
    lengths = torch.tensor([3, 2], device=device)
    catalog = torch.arange(1, 6, device=device)
    positive_ids = [[1, 4], [4]]

    def scores(tokens, sizes):
        states = model.encode(tokens, sizes)
        return model.score_items(last_valid_state(states, sizes, "right"), catalog)

    initial = scores(ids, lengths)
    loss = multi_positive_loss(initial, catalog, positive_ids)
    loss.backward()
    mixer = model.blocks[0].mamba.in_proj.weight
    head = model.item_embedding.weight
    grad_norms = {
        "mamba_input_projection": mixer.grad.norm().item(),
        "mamba_transition_log": model.blocks[0].mamba.A_log.grad.norm().item(),
        "output_only_item_4": head.grad[4].norm().item(),
    }
    if not torch.isfinite(loss) or not all(0 < x < float("inf") for x in grad_norms.values()):
        raise RuntimeError(f"Expected a finite loss and nonzero finite gradients; measured {grad_norms}.")
    if any(p.grad is None or not p.grad.isfinite().all() for p in model.parameters() if p.requires_grad):
        raise RuntimeError("At least one trainable parameter has a missing or non-finite gradient.")
    mixer_before = mixer.detach().clone()
    head_before = head[4].detach().clone()
    optimizer.step()
    if torch.equal(mixer, mixer_before) or torch.equal(head[4], head_before):
        raise RuntimeError("The optimizer must update both the encoder and an output-only item.")

    model.eval()
    with torch.no_grad():
        reference = scores(ids, lengths)
        short = scores(ids[:1, :3], lengths[:1])
        torch.testing.assert_close(short[0], reference[0], atol=1e-4, rtol=1e-4)
        # Serialize through Torch's checkpoint format, then score with newly constructed weights.
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model = Mamba4Rec(5, hidden_dim=32, d_state=16, dropout=0.0).to(device).eval()
        model.load_state_dict(torch.load(buffer, map_location=device, weights_only=True))
        reloaded = scores(ids, lengths)
        torch.testing.assert_close(reloaded, reference, atol=1e-5, rtol=1e-5)
    torch.cuda.synchronize()
    return {"backend": "mamba_ssm.modules.mamba_simple.Mamba", "loss": loss.item(),
            "grad_norms": grad_norms, "encoder_and_output_updated": True,
            "padding_max_abs_error": (short[0] - reference[0]).abs().max().item(),
            "reload_max_abs_error": (reloaded - reference).abs().max().item(),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
