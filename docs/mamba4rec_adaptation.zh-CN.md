# Mamba4Rec 月度商品推荐适配

[English](mamba4rec_adaptation.md) · [简体中文](mamba4rec_adaptation.zh-CN.md)

适配器直接实例化 Chengkai Liu 等人在提交 `44a0c8432b34e3b89d46035b10e8525b1bd368dc` 中发布的 `Mamba4Rec` 类，训练与评分均调用作者的 `forward`。仓库同时保存未经修改的推荐模型原文件和 MIT 许可证。适配器会核对固定文件的 SHA-256，`mamba4rec.py` 的摘要为 `5cfb15241ad3608ef21e7abd9739ad03b363c10759dcdf136aa02305392c91f2`。[论文](https://arxiv.org/abs/2403.03900) · [仓库内作者源码](../third_party/mamba4rec/mamba4rec.py) · [MIT 许可证](../third_party/mamba4rec/LICENSE) · [来源清单](../third_party/mamba4rec/source.json) · [适配器](../src/models/mamba4rec_adapter.py)

## 从购买历史到推荐分数

1. D2 提供真实商品 ID 序列，保留时间顺序和复购。截取最近 `max_length` 个真实 ID，在右侧补 0。例如 `[A, B, A, 0, 0]` 的有效长度为三，客户表示取下标二的状态；空历史使用零向量。[共用历史](../src/data/sequences.py)
2. 项目将 D2 的词表行数（含预留的填充行）和配置传入真实 RecBole 的 `SequentialRecommender` 基类。商品嵌入、Mamba 层、归一化、前馈层及参数初始化均由作者代码实现。作者文件在导入时还会加载 `BPRLoss`，项目训练使用自己的月度损失。[作者模型文件](../third_party/mamba4rec/mamba4rec.py)
3. 作者的 `forward` 已取出 `length − 1` 位置的最终状态。项目使用这个状态与 D2 候选商品对应的共享嵌入行做点积，计算方式与作者 `full_sort_predict` 一致。买过的商品仍可被推荐，缺少训练支持的嵌入行在输入和评分时均使用受支持真实商品行的均值。候选分块评分时复用该查询已算出的最终状态。[共用训练器与损失](../src/models/sequence_objective.py) · [月度协议](data_protocol.zh-CN.md)
4. 上游训练样本只有一个目标商品，本项目一个客户–月份可以有多个不同的净保留正例。先在该查询可见的全候选上计算 log-softmax，对全部不同正例的负对数概率取平均，再对查询等权平均。RecBole 的数据切分、单商品损失和评价由 D2 表及统一评价器接管。共同起步配置使用最近 64 个事件、隐藏维度 64、两层、状态维度 16、卷积宽度 4、扩展因子 2、dropout 0.2 与种子 42。[模型配置](../configs/mamba4rec.yaml) · [评价器](../src/evaluation/ranking.py)

## 原生 CUDA 环境及验证

**验证状态。** 已在 Python 3.12.14、NumPy 2.2.6、PyTorch 2.6.0+cpu 环境下实测 RecBole 1.2.0。基类能实例化，两种有效长度的末状态误差为零，BPRLoss 能完成反向计算。原生 Mamba 训练及十月试训分数仍需在 NVIDIA 设备上测量。预检记录真实 RecBole 导入、源文件摘要、原生前向与反向计算、与作者全候选评分的一致性、参数更新和模型重载。引用数值时请核对修订版记录的设备和实测状态。[RecBole CPU 检查](../results/model_pilots_upstream/mamba4rec/recbole_cpu_import.json) · [预检记录](../results/model_pilots_upstream/mamba4rec/preflight.json) · [试训状态](../results/model_pilots_upstream/summary.csv)

作者存档环境列有 Python 3.8.18、`mamba-ssm==1.1.4`、`causal-conv1d==1.1.3.post1`，并在注释内列出 PyTorch 2.2.1 CUDA 11.8。项目为当前 Python 单独选取二进制包。两个官方发布页均已核对，列有适配 Linux x86-64、CPython 3.11、3.12、3.13 和两种 C++ ABI 的成对 wheel。硬件验证通过须以实际 NVIDIA GPU 上的 `passed` 预检记录为准。[作者环境](https://github.com/chengkai-liu/Mamba4Rec/blob/44a0c8432b34e3b89d46035b10e8525b1bd368dc/environment.yaml) · [Mamba 二进制发布](https://github.com/state-spaces/mamba/releases/tag/v2.2.5) · [卷积二进制发布](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.3.post1)

| 组件 | Colab 安装目标 | 核验方式 |
|---|---|---|
| 驱动与设备 | NVIDIA GPU 及兼容 CUDA 12.4 的驱动 | 选择「运行时 → 更改运行时类型 → GPU」。预检记录实际 GPU、驱动及计算能力。 |
| 解释器与 Torch | 独立的 Linux x86-64 CPython 3.11–3.13，从 CUDA 12.4 wheel 源安装 PyTorch 2.6.0 | 安装、预检、训练使用同一解释器。核对 `torch.version.cuda` 与 `torch._C._GLIBCXX_USE_CXX11_ABI`。 |
| 原生扫描与卷积 | `mamba-ssm==2.2.5`、`causal-conv1d==1.5.3.post1` | 两份 wheel 的 Python 版本、Linux 架构、Torch 2.6、CUDA 12、C++ ABI 必须分别匹配。 |
| Python 依赖 | `recbole==1.2.0`、`transformers==4.48.3` 及项目锁定的数值计算软件包 | 先安装 CUDA 版 PyTorch，再装项目的 Mamba4Rec 依赖和匹配的原生 wheel。预检会导入真实 RecBole 的基类和损失类。[运行依赖](../requirements-mamba4rec.txt) |

打开模型试训 notebook，选择 Mamba4Rec 的 GPU 选项，依次执行各单元格。安装步骤会按解释器与 Torch 选择官方原生 wheel。例如 CPython 3.12、Torch 2.6、CUDA 12 且使用旧 C++ ABI 时，两份 wheel 的文件名都应以 `cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl` 结尾；使用新 ABI 时选择各发布中的 `TRUE` 版本。安装与训练必须能读到同一批原生库。[Colab 试训 notebook](../notebooks/model_pilots.ipynb) · [PyTorch 2.6.0 CUDA 安装说明](https://pytorch.org/get-started/previous-versions/#v260)

在该环境中，也可在仓库根目录执行以下命令。第一条仅使用合成历史。第二条通过 `--smoke` 在 D2 的真实训练子集及十月验证子集上试训。

```bash
python -m src.models.mamba4rec_adapter --preflight --output runs/mamba4rec/preflight.json
python -m scripts.train --model mamba4rec --config configs/mamba4rec.yaml --device cuda --smoke --output-dir runs/mamba4rec/smoke
```

预检成功时，记录实际环境、有限的多正例损失、Mamba 投影和状态转移以及输出嵌入的非零梯度，并核对编码器参数和仅作候选的商品行都已更新。右填充及不同批次下的评分误差采用 `atol=rtol=1e-4`，保存和重新加载后的评分误差采用 `atol=rtol=1e-5`。预检失败时返回非零退出码，并记录异常。真实试训单独写入 `run.json`、指标、验证集 Top-10 及模型检查点。仅在硬件运行成功后报告耗时和实测指标。只核对解释器或成功导入 wheel，尚不足以确认分配到的 GPU 能执行选择性扫描的前向与反向计算。[CUDA 检查](../tests/test_mamba4rec.py) · [训练入口](../scripts/train.py)
