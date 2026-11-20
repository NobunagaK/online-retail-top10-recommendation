# Mamba4Rec for monthly product recommendations

[English](mamba4rec_adaptation.md) · [简体中文](mamba4rec_adaptation.zh-CN.md)

Mamba4Rec ranks the same monthly catalog as MF, LightGBM and LRURec. The adapter implements the architecture of Chengkai Liu et al. at commit `44a0c8432b34e3b89d46035b10e8525b1bd368dc` and calls the official Mamba-1 selective state-space module. The authors' PyTorch source file at that commit has SHA-256 `5cfb15241ad3608ef21e7abd9739ad03b363c10759dcdf136aa02305392c91f2`. [Paper](https://arxiv.org/abs/2403.03900) · [Pinned implementation](https://github.com/chengkai-liu/Mamba4Rec/blob/44a0c8432b34e3b89d46035b10e8525b1bd368dc/mamba4rec.py) · [Project adapter](../src/models/mamba4rec_adapter.py)

## From a shopping history to scores

1. The D2 history supplies the real item IDs, preserving chronological order and repeat purchases. Keep the most recent `max_length` IDs and pad on the right with 0. For a row `[A, B, A, 0, 0]` the valid length is three and the state at index two represents the customer. An empty history has a zero representation. [Shared histories](../src/data/sequences.py)
2. The tied embedding table maps each real ID to a vector. The input passes through dropout and layer normalization. Each block calls the authors' Mamba-1 operator with the fused selective scan and causal convolution. With one block, the mixer output passes through dropout and normalization. With stacked blocks, each mixer also adds its input before normalization. Each feed-forward sublayer expands to four times the hidden dimension, applies GELU, projects back, and adds a residual. The adapter follows the recommender's normal initialization for linear/embedding weights and zeros the padding row. [Official Mamba-1 operator](https://github.com/state-spaces/mamba/blob/v2.2.5/mamba_ssm/modules/mamba_simple.py)
3. The common runner selects the state at `length − 1` and takes its dot product with each eligible candidate's embedding. The scoring head uses the same embedding table as the input and has no item bias. A previously purchased SKU stays eligible. PAD and items outside the query's catalog are excluded. For inference, unsupported embedding rows use the mean of the supported real item rows for both inputs and scoring. [Shared runner and objective](../src/models/sequence_objective.py) · [Monthly protocol](data_protocol.md)
4. The upstream training sample has one purchased target item. Here each customer–month supplies all distinct net-retained positives. The loss averages their negative log-softmax scores across the full visible catalog, then gives each query equal weight. RecBole's dataset splitting, single-item loss and evaluator are replaced by the D2 tables and project evaluator. The shared starting configuration uses the latest 64 items, hidden size 64, two blocks, state size 16, convolution width 4, expansion factor 2, dropout 0.2 and seed 42. [Configuration](../configs/mamba4rec.yaml) · [Evaluator](../src/evaluation/ranking.py)

## Native CUDA environment and verification

**Verification status.** The local host has PyTorch 2.8.0+cpu and zero CUDA devices. Its preflight recorded a failed capability check. The CPU failure-reporting test passed and the three native tests were skipped on this host. Native forward/backward, GPU runtime and October validation metrics are pending a GPU run. [Recorded local preflight](../results/model_pilots/mamba4rec/preflight.json)

The authors' archived environment lists Python 3.8.18, `mamba-ssm==1.1.4`, `causal-conv1d==1.1.3.post1` and a commented PyTorch 2.2.1 CUDA 11.8 pin. The project uses a separate wheel target for current Python. The release listings include matching Linux x86-64 CPython 3.11, 3.12 and 3.13 binaries for both C++ ABI choices. The listed wheel availability has been checked. A passing hardware run requires an actual NVIDIA GPU and a `passed` preflight record from that runtime. [Authors' environment](https://github.com/chengkai-liu/Mamba4Rec/blob/44a0c8432b34e3b89d46035b10e8525b1bd368dc/environment.yaml) · [Mamba binary releases](https://github.com/state-spaces/mamba/releases/tag/v2.2.5) · [Convolution binary releases](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.3.post1)

| Component | Colab installation target | What to verify |
|---|---|---|
| Driver and device | NVIDIA GPU, driver compatible with CUDA 12.4 | Select **Runtime → Change runtime type → GPU**. The preflight records the detected GPU, driver and compute capability. |
| Interpreter and Torch | Isolated Linux x86-64 CPython 3.11–3.13, PyTorch 2.6.0 from the CUDA 12.4 wheel index | Use the same interpreter for setup, preflight and training. Inspect `torch.version.cuda` and `torch._C._GLIBCXX_USE_CXX11_ABI`. |
| Native scan and convolution | `mamba-ssm==2.2.5`, `causal-conv1d==1.5.3.post1` | Match **all** wheel tags: Python version, Linux architecture, Torch 2.6, CUDA 12 and C++ ABI. |
| Python support | `transformers==4.48.3`, einops, packaging and Torch's compatible Triton | The notebook installs the model environment separately from D1's CPU environment. |

Open the model pilot notebook, select the Mamba4Rec GPU option, and run the cells in order. Its setup uses published native wheels matched to the interpreter and installed Torch. For example, a CPython 3.12 Torch 2.6 CUDA 12 build with the old C++ ABI needs wheel filenames ending in `cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`. An ABI-true build needs the `TRUE` wheel from each release. The setup and training must see the same native libraries. [Colab pilot notebook](../notebooks/model_pilots.ipynb) · [PyTorch 2.6.0 CUDA installation](https://pytorch.org/get-started/previous-versions/#v260)

With that environment active, these commands run the checks from the repository root. The first uses only synthetic histories. `--smoke` selects a small real D2 training/October-validation pilot.

```bash
python -m src.models.mamba4rec_adapter --preflight --output runs/mamba4rec/preflight.json
python -m scripts.train --model mamba4rec --config configs/mamba4rec.yaml --device cuda --smoke --output-dir runs/mamba4rec/smoke
```

A successful preflight records the concrete environment, a finite multi-positive loss, nonzero gradients in the Mamba projection, transition and output embedding, and a parameter update in the encoder and an output-only item. It compares right-padding/batch invariance at `atol=rtol=1e-4` and saved/reloaded scores at `atol=rtol=1e-5`. A failing preflight exits nonzero and records the exception. The real pilot writes its own `run.json`, metrics, validation Top-10 and model checkpoint. Report the measured runtime and metrics only for completed hardware runs. An interpreter or wheel import check alone cannot establish that selective-scan forward and backward work on the allocated device. [CUDA checks](../tests/test_mamba4rec.py) · [Training entry point](../scripts/train.py)
