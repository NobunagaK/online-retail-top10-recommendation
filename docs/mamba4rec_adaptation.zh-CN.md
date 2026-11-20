# Mamba4Rec 月度商品推荐适配

[English](mamba4rec_adaptation.md) · [简体中文](mamba4rec_adaptation.zh-CN.md)

适配器直接实例化 Chengkai Liu 等人在提交 `44a0c8432b34e3b89d46035b10e8525b1bd368dc` 中发布的 `Mamba4Rec` 类，训练与评分均调用作者的 `forward`。仓库同时保存未经修改的推荐模型原文件和 MIT 许可证。适配器会核对固定文件的 SHA-256，`mamba4rec.py` 的摘要为 `5cfb15241ad3608ef21e7abd9739ad03b363c10759dcdf136aa02305392c91f2`。[论文](https://arxiv.org/abs/2403.03900) · [仓库内作者源码](../third_party/mamba4rec/mamba4rec.py) · [MIT 许可证](../third_party/mamba4rec/LICENSE) · [来源清单](../third_party/mamba4rec/source.json) · [适配器](../src/models/mamba4rec_adapter.py)

## 从购买历史到推荐分数

1. D2 提供真实商品 ID 序列，保留时间顺序和复购。截取最近 `max_length` 个真实 ID，在右侧补 0。例如 `[A, B, A, 0, 0]` 的有效长度为三，客户表示取下标二的状态；空历史使用零向量。[共用历史](../src/data/sequences.py)
2. 项目将 D2 的词表行数（含预留的填充行）和配置传入真实 RecBole 的 `SequentialRecommender` 基类。商品嵌入、Mamba 层、归一化、前馈层及参数初始化均由作者代码实现。作者文件在导入时还会加载 `BPRLoss`，项目训练使用自己的月度损失。[作者模型文件](../third_party/mamba4rec/mamba4rec.py)
3. 作者的 `forward` 已取出 `length − 1` 位置的最终状态。项目使用这个状态与 D2 候选商品对应的共享嵌入行做点积，计算方式与作者 `full_sort_predict` 一致。买过的商品仍可被推荐，缺少训练支持的嵌入行在输入和评分时均使用受支持真实商品行的均值。候选分块评分时复用该查询已算出的最终状态。[共用训练器与损失](../src/models/sequence_objective.py) · [月度协议](data_protocol.zh-CN.md)
4. 上游训练样本只有一个目标商品，本项目一个客户–月份可以有多个不同的净保留正例。先在该查询可见的全候选上计算 log-softmax，对全部不同正例的负对数概率取平均，再对查询等权平均。RecBole 的数据切分、单商品损失和评价由 D2 表及统一评价器接管。共同起步配置使用最近 64 个事件、隐藏维度 64、两层、状态维度 16、卷积宽度 4、扩展因子 2、dropout 0.2 与种子 42。[模型配置](../configs/mamba4rec.yaml) · [评价器](../src/evaluation/ranking.py)

## 原生 CUDA 环境及验证

**Colab 小样本结果。** 据团队实跑记录，作者版本 Mamba4Rec（`44a0c84`）使用真实 RecBole 1.2.0，在 Colab Python 3.13 和 Tesla T4 上通过原生 GPU 预检并完成一轮训练。16 条十月验证查询的 Recall@10 为 0.02345，NDCG@10 为 0.07513，抽样 AUC 为 0.50397。重载后 Top-10 商品及顺序完全复现，最大分数差为 0.0。该次会话在 Drive 项目根目录下的 `experiments/model_pilots/mamba4rec/20261120T163637Z_smoke_0bb0e341`。仓库 notebook 展示另一次 LRURec 的 GPU 运行输出。归档的 CPU 预检记录来自此前一台无 CUDA 的机器。[GPU 会话记录](../results/model_pilots_upstream/colab_gpu_smoke.json) · [此前的 CPU 预检](../results/model_pilots_upstream/mamba4rec/preflight.json)

作者发布的原始环境使用 Python 3.8.18 和早期版本的原生库。项目此前在 Python 3.12.14、PyTorch 2.6.0+cpu 上进行的 RecBole 接口核验也已归档。[作者环境](https://github.com/chengkai-liu/Mamba4Rec/blob/44a0c8432b34e3b89d46035b10e8525b1bd368dc/environment.yaml) · [CPU 接口核验](../results/model_pilots_upstream/mamba4rec/recbole_cpu_import.json)

团队在 Python 3.12 和 3.13 上使用 `torch 2.6.0+cu124` 与 ABI-FALSE 的原生扩展 wheel 时，加载扩展出现 `std::__cxx11::basic_string` 符号解析错误。要复现跑通的配置，请在 Colab 选择 Tesla T4，在 notebook 中选择 `mamba4rec` 与 `smoke`，依次运行全部单元格。第 5 号单元格在同一独立环境中安装下列组件。[试训 notebook](../notebooks/model_pilots.ipynb)

| 组件 | 已实测的 Colab 配置 |
|---|---|
| 解释器与设备 | Linux x86-64、CPython 3.13、NVIDIA Tesla T4 |
| PyTorch | 从 `https://download.pytorch.org/whl/cu126` 安装 `torch==2.7.0`。实际构建为 `2.7.0+cu126`，`torch._C._GLIBCXX_USE_CXX11_ABI == True`。CPython 3.13 的 PyTorch wheel 文件名以 `cp313-cp313-manylinux_2_28_x86_64.whl` 结尾。 |
| Python 依赖 | 安装 `requirements-mamba4rec.txt`，包括 `recbole==1.2.0`、`transformers==4.48.3`。预检会导入真实 RecBole 的基类和损失类。 |
| 原生扫描与卷积 | 安装官方发布的 `mamba-ssm==2.2.5` 和 `causal-conv1d==1.5.3.post1` wheel。两个文件名均以 `+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl` 结尾。为这两个包指定 `--no-deps`，以保留已安装的 PyTorch 构建。 |

[PyTorch 2.7.0 安装说明](https://pytorch.org/get-started/previous-versions/#v270) · [项目依赖](../requirements-mamba4rec.txt) · [Mamba 二进制发布](https://github.com/state-spaces/mamba/releases/tag/v2.2.5) · [卷积二进制发布](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.3.post1)

在该环境中，也可在仓库根目录执行以下命令。第一条仅使用合成历史。第二条通过 `--smoke` 在 D2 的真实训练子集及十月验证子集上试训。

```bash
python -m src.models.mamba4rec_adapter --preflight --output runs/mamba4rec/preflight.json
python -m scripts.train --model mamba4rec --config configs/mamba4rec.yaml --device cuda --smoke --output-dir runs/mamba4rec/smoke
```

预检成功时，记录实际环境、有限的多正例损失、Mamba 投影和状态转移以及输出嵌入的非零梯度，并核对编码器参数和仅作候选的商品行都已更新。右填充及不同批次下的评分误差采用 `atol=rtol=1e-4`，保存和重新加载后的评分误差采用 `atol=rtol=1e-5`。预检失败时返回非零退出码，并记录异常。真实试训单独写入 `run.json`、指标、验证集 Top-10 及模型检查点。仅在硬件运行成功后报告耗时和实测指标。只核对解释器或成功导入 wheel，尚不足以确认分配到的 GPU 能执行选择性扫描的前向与反向计算。[CUDA 检查](../tests/test_mamba4rec.py) · [训练入口](../scripts/train.py)
