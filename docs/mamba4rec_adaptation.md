# Mamba4Rec for monthly product recommendations

[English](mamba4rec_adaptation.md) · [简体中文](mamba4rec_adaptation.zh-CN.md)

The adapter instantiates Chengkai Liu et al.’s `Mamba4Rec` class from commit `44a0c8432b34e3b89d46035b10e8525b1bd368dc` and calls its `forward` during training and evaluation. The repository carries the unmodified recommender and its MIT license. The adapter verifies the pinned SHA-256 hashes, including `5cfb15241ad3608ef21e7abd9739ad03b363c10759dcdf136aa02305392c91f2` for `mamba4rec.py`. [Paper](https://arxiv.org/abs/2403.03900) · [Authors’ code in this checkout](../third_party/mamba4rec/mamba4rec.py) · [MIT license](../third_party/mamba4rec/LICENSE) · [Source manifest](../third_party/mamba4rec/source.json) · [Adapter](../src/models/mamba4rec_adapter.py)

## From a shopping history to scores

1. The D2 history supplies the real item IDs, preserving chronological order and repeat purchases. Keep the most recent `max_length` IDs and pad on the right with 0. For a row `[A, B, A, 0, 0]` the valid length is three and the state at index two represents the customer. An empty history has a zero representation. [Shared histories](../src/data/sequences.py)
2. The project supplies the D2 vocabulary size (including the reserved padding row) and configuration to RecBole’s real `SequentialRecommender` base. The authors’ code owns the item embeddings, Mamba layers, normalization, feed-forward blocks and parameter initialization. It also imports `BPRLoss` during module loading, although the project computes its own monthly loss. [Authors’ source](../third_party/mamba4rec/mamba4rec.py)
3. The authors’ `forward` already returns the state at `length − 1`. The adapter scores D2 candidates using that state and the same embedding rows and dot product as the authors’ `full_sort_predict`. Previously purchased items remain eligible. For inference, unsupported embedding rows use the mean of the supported real item rows for input and output. The runner caches the final representation while scoring successive candidate blocks. [Shared runner and objective](../src/models/sequence_objective.py) · [Monthly protocol](data_protocol.md)
4. The upstream training sample has one purchased target item. Here each customer–month supplies all distinct net-retained positives. The loss averages their negative log-softmax scores across the full visible catalog, then gives each query equal weight. RecBole's dataset splitting, single-item loss and evaluator are replaced by the D2 tables and project evaluator. The shared starting configuration uses the latest 64 items, hidden size 64, two blocks, state size 16, convolution width 4, expansion factor 2, dropout 0.2 and seed 42. [Configuration](../configs/mamba4rec.yaml) · [Evaluator](../src/evaluation/ranking.py)

## Native CUDA environment and verification

**Colab smoke result.** The team reports a successful native preflight and one-epoch smoke run with the authors’ Mamba4Rec (`44a0c84`) and real RecBole 1.2.0 on Colab Python 3.13 and a Tesla T4. Across 16 October validation queries, Recall@10 was 0.02345, NDCG@10 was 0.07513 and sampled AUC was 0.50397. Reload reproduced the ordered Top-10 with a maximum score difference of 0.0. Under the Drive project root, the run directory is `experiments/model_pilots/mamba4rec/20261120T163637Z_smoke_0bb0e341`. The committed notebook displays the separate LRURec GPU run. The archived CPU preflight documents an earlier check on a machine without CUDA. [GPU session records](../results/model_pilots_upstream/colab_gpu_smoke.json) · [Earlier CPU preflight](../results/model_pilots_upstream/mamba4rec/preflight.json)

The authors published a Python 3.8.18 environment with earlier native libraries. The project also retains its earlier RecBole interface check on Python 3.12.14 and PyTorch 2.6.0+cpu. [Authors’ environment](https://github.com/chengkai-liu/Mamba4Rec/blob/44a0c8432b34e3b89d46035b10e8525b1bd368dc/environment.yaml) · [CPU check](../results/model_pilots_upstream/mamba4rec/recbole_cpu_import.json)

The team encountered an unresolved `std::__cxx11::basic_string` symbol when loading native extensions with `torch 2.6.0+cu124` and ABI-FALSE wheels on Python 3.12 and 3.13. To reproduce the successful configuration, choose a Tesla T4 in Colab, select `mamba4rec` and `smoke` in the notebook, and run all cells. Cell 5 installs the following components in one isolated environment. [Pilot notebook](../notebooks/model_pilots.ipynb)

| Component | Validated Colab configuration |
|---|---|
| Interpreter and device | Linux x86-64, CPython 3.13, NVIDIA Tesla T4 |
| PyTorch | Install `torch==2.7.0` from the `https://download.pytorch.org/whl/cu126` index. The build is `2.7.0+cu126`, with `torch._C._GLIBCXX_USE_CXX11_ABI == True`. The CPython 3.13 PyTorch wheel ends in `cp313-cp313-manylinux_2_28_x86_64.whl`. |
| Python dependencies | Install `requirements-mamba4rec.txt`, including `recbole==1.2.0` and `transformers==4.48.3`. The preflight imports the real RecBole base class and loss. |
| Native scan and convolution | Install the official `mamba-ssm==2.2.5` and `causal-conv1d==1.5.3.post1` release wheels ending in `+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl`. Use `--no-deps` for these wheels to preserve the selected PyTorch build. |

[PyTorch 2.7.0 installation](https://pytorch.org/get-started/previous-versions/#v270) · [Project requirements](../requirements-mamba4rec.txt) · [Mamba release](https://github.com/state-spaces/mamba/releases/tag/v2.2.5) · [Convolution release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.3.post1)

With that environment active, these commands run the checks from the repository root. The first uses only synthetic histories. `--smoke` selects a small real D2 training/October-validation pilot.

```bash
python -m src.models.mamba4rec_adapter --preflight --output runs/mamba4rec/preflight.json
python -m scripts.train --model mamba4rec --config configs/mamba4rec.yaml --device cuda --smoke --output-dir runs/mamba4rec/smoke
```

A successful preflight records the concrete environment, a finite multi-positive loss, nonzero gradients in the Mamba projection, transition and output embedding, and a parameter update in the encoder and an output-only item. It compares right-padding/batch invariance at `atol=rtol=1e-4` and saved/reloaded scores at `atol=rtol=1e-5`. A failing preflight exits nonzero and records the exception. The real pilot writes its own `run.json`, metrics, validation Top-10 and model checkpoint. Report the measured runtime and metrics only for completed hardware runs. An interpreter or wheel import check alone cannot establish that selective-scan forward and backward work on the allocated device. [CUDA checks](../tests/test_mamba4rec.py) · [Training entry point](../scripts/train.py)
