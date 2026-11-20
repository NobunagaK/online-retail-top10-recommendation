# Mamba4Rec 月度商品推荐适配

[English](mamba4rec_adaptation.md) · [简体中文](mamba4rec_adaptation.zh-CN.md)

Mamba4Rec 与 MF、LightGBM 和 LRURec 对同一月度候选目录评分。适配器依据 Chengkai Liu 等人仓库的提交 `44a0c8432b34e3b89d46035b10e8525b1bd368dc` 实现推荐模型结构，并调用官方 Mamba-1 选择性状态空间模块。该提交的 PyTorch 源文件 SHA-256 为 `5cfb15241ad3608ef21e7abd9739ad03b363c10759dcdf136aa02305392c91f2`。[论文](https://arxiv.org/abs/2403.03900) · [固定版本源码](https://github.com/chengkai-liu/Mamba4Rec/blob/44a0c8432b34e3b89d46035b10e8525b1bd368dc/mamba4rec.py) · [项目适配器](../src/models/mamba4rec_adapter.py)

## 从购买历史到推荐分数

1. D2 提供真实商品 ID 序列，保留时间顺序和复购。截取最近 `max_length` 个真实 ID，在右侧补 0。例如 `[A, B, A, 0, 0]` 的有效长度为三，客户表示取下标二的状态；空历史使用零向量。[共用历史](../src/data/sequences.py)
2. 输入与输出共用一张商品嵌入表。真实 ID 转成向量后经过 dropout 和层归一化。每一层调用作者使用的 Mamba-1 算子，使用融合的选择性扫描与因果卷积。只有一层时，混合器输出经 dropout 和归一化；堆叠多层时，各层的混合器还在归一化前加上输入作为残差。随后前馈层扩维至四倍隐藏维度，使用 GELU，再投影回原维度并加残差。线性层和嵌入表沿用推荐模型的正态初始化，填充行设为零。[官方 Mamba-1 算子](https://github.com/state-spaces/mamba/blob/v2.2.5/mamba_ssm/modules/mamba_simple.py)
3. 共用训练器取 `length − 1` 处的状态，与各候选商品的输入嵌入计算点积。评分头没有商品偏置。客户买过的商品仍可推荐，填充 ID 和目录外商品均排除。预测时，没有训练支持的商品行在历史编码与候选打分时均使用已获支持的真实商品行均值。[共用训练器与目标函数](../src/models/sequence_objective.py) · [月度协议](data_protocol.zh-CN.md)
4. 上游训练样本只有一个目标商品，本项目一个客户–月份可以有多个不同的净保留正例。先在该查询可见的全候选上计算 log-softmax，对全部不同正例的负对数概率取平均，再对查询等权平均。RecBole 的数据切分、单商品损失和评价由 D2 表及统一评价器接管。共同起步配置使用最近 64 个事件、隐藏维度 64、两层、状态维度 16、卷积宽度 4、扩展因子 2、dropout 0.2 与种子 42。[模型配置](../configs/mamba4rec.yaml) · [评价器](../src/evaluation/ranking.py)

## 原生 CUDA 环境及验证

**验证状态。** 本地运行机的 PyTorch 为 2.8.0+cpu，可见 CUDA 设备数为零。预检记录了设备能力检查失败；该主机的 CPU 失败报告测试已通过，三项原生算子测试因无 GPU 跳过。原生前向与反向计算、GPU 耗时及十月验证指标仍待 GPU 实跑。[本地预检记录](../results/model_pilots/mamba4rec/preflight.json)

作者存档环境列有 Python 3.8.18、`mamba-ssm==1.1.4`、`causal-conv1d==1.1.3.post1`，并在注释内列出 PyTorch 2.2.1 CUDA 11.8。项目为当前 Python 单独选取二进制包。两个官方发布页均已核对，列有适配 Linux x86-64、CPython 3.11、3.12、3.13 和两种 C++ ABI 的成对 wheel。硬件验证通过须以实际 NVIDIA GPU 上的 `passed` 预检记录为准。[作者环境](https://github.com/chengkai-liu/Mamba4Rec/blob/44a0c8432b34e3b89d46035b10e8525b1bd368dc/environment.yaml) · [Mamba 二进制发布](https://github.com/state-spaces/mamba/releases/tag/v2.2.5) · [卷积二进制发布](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.3.post1)

| 组件 | Colab 安装目标 | 核验方式 |
|---|---|---|
| 驱动与设备 | NVIDIA GPU 及兼容 CUDA 12.4 的驱动 | 选择「运行时 → 更改运行时类型 → GPU」。预检记录实际 GPU、驱动及计算能力。 |
| 解释器与 Torch | 独立的 Linux x86-64 CPython 3.11–3.13，从 CUDA 12.4 wheel 源安装 PyTorch 2.6.0 | 安装、预检、训练使用同一解释器。核对 `torch.version.cuda` 与 `torch._C._GLIBCXX_USE_CXX11_ABI`。 |
| 原生扫描与卷积 | `mamba-ssm==2.2.5`、`causal-conv1d==1.5.3.post1` | 两份 wheel 的 Python 版本、Linux 架构、Torch 2.6、CUDA 12、C++ ABI 必须分别匹配。 |
| Python 依赖 | `transformers==4.48.3`、einops、packaging 及与 Torch 配套的 Triton | notebook 将模型环境与 D1 的 CPU 环境分开安装。 |

打开模型试训 notebook，选择 Mamba4Rec 的 GPU 选项，依次执行各单元格。安装步骤会按解释器与 Torch 选择官方原生 wheel。例如 CPython 3.12、Torch 2.6、CUDA 12 且使用旧 C++ ABI 时，两份 wheel 的文件名都应以 `cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl` 结尾；使用新 ABI 时选择各发布中的 `TRUE` 版本。安装与训练必须能读到同一批原生库。[Colab 试训 notebook](../notebooks/model_pilots.ipynb) · [PyTorch 2.6.0 CUDA 安装说明](https://pytorch.org/get-started/previous-versions/#v260)

在该环境中，也可在仓库根目录执行以下命令。第一条仅使用合成历史。第二条通过 `--smoke` 在 D2 的真实训练子集及十月验证子集上试训。

```bash
python -m src.models.mamba4rec_adapter --preflight --output runs/mamba4rec/preflight.json
python -m scripts.train --model mamba4rec --config configs/mamba4rec.yaml --device cuda --smoke --output-dir runs/mamba4rec/smoke
```

预检成功时，记录实际环境、有限的多正例损失、Mamba 投影和状态转移以及输出嵌入的非零梯度，并核对编码器参数和仅作候选的商品行都已更新。右填充及不同批次下的评分误差采用 `atol=rtol=1e-4`，保存和重新加载后的评分误差采用 `atol=rtol=1e-5`。预检失败时返回非零退出码，并记录异常。真实试训单独写入 `run.json`、指标、验证集 Top-10 及模型检查点。仅在硬件运行成功后报告耗时和实测指标。只核对解释器或成功导入 wheel，尚不足以确认分配到的 GPU 能执行选择性扫描的前向与反向计算。[CUDA 检查](../tests/test_mamba4rec.py) · [训练入口](../scripts/train.py)
