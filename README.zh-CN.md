# 在线零售商品 Top-10 推荐

[English](README.md) · [简体中文](README.zh-CN.md)

本 CS5481 项目比较 MF、LightGBM、LRURec 和 Mamba4Rec，目标是根据客户历史为后续月份推荐十种商品。四个模型共用净有效购买的购物篮历史、候选目录和月度标签。MF、LightGBM 和作者版本 LRURec 已完成 CPU 试训。两种序列适配器均调用作者固定版本的推荐模型源码。两种序列模型的 Colab GPU 核验及正式模型比较仍待完成。[研究与代码开发计划 v3.4](docs/CS5481_Project2_Research_Development_Plan_3.4.md)

## 运行说明

### D1 · 从原始交易到训练期购物篮

课程要求解释取消、退货、重复交易明细、缺失客户编号、异常数量或价格，说明如何聚合客户与商品的交互，并按时间构建历史。三份 notebook 使用同一工作簿。[五项课程要求，第 3 页](resource/Group_Project_Tutorial_Recommender_Systems.pdf#page=3) · [异常分析](docs/technical/transaction_exception_analysis.ipynb) · [清洗与课程问题回答](notebooks/project2_colab.ipynb) · [训练期探索](notebooks/training_eda.ipynb)

1. **先看异常分析。** 按原始 Excel 行号核对 C 单、负数量、零价、坏账和库存备注；在新增的匹配表中追查每条商品冲销的原单、余额、时间和地区，并比较几种清洗规则带来的差异。[证据工作簿](results/transaction_exceptions.xlsx)
2. **再做 D1 清洗。** 查看每一步的证据、行数、处理理由以及五项课程问题的回答。A 表保留匿名和已知客户的商品交易；B 表只保留已知客户。两表都按记录中的原价将正负明细各计一次。每笔原购买扣除截止前已发生且匹配到它的所有冲销，包含取消和退货。部分冲销保留余量；已全额冲销的行贡献零件。按客户、发票、商品聚合后，余量为正才产生一次购买事件。[经营账 A](data/processed/signed_ledger_a.csv.gz) · [客户账 B](data/processed/signed_ledger_b.csv.gz) · [10 月 1 日购物篮快照](data/processed/training_basket_events.csv.gz)
3. **最后看训练期 EDA。** 读取 2011 年 10 月 1 日零时的专用快照，覆盖 2010 年 12 月至 2011 年 9 月完成的购物篮。月度活动、单价和数量分布、商品热度、复购和稀疏度均按当时已知的冲销重算。[训练期统计](results/training_eda/summary.json) · [月度统计](results/training_eda/monthly_activity.csv) · [商品排名](results/training_eda/item_popularity.csv)

**在 Colab 运行。** 在「文件 → 打开笔记本 → GitHub」选择仓库和包含该文件的分支，依次打开上述 notebook，在 CPU 上从上往下运行。Drive 项目根目录为 `我的云端硬盘/CS5481_Project2_OnlineRetail/`，原表位于 `data/raw/Online Retail.xlsx`。分析和清洗均直接读取原表；EDA 读取清洗产物。生成的表和图写入同一项目的 `data/processed/` 和 `results/`。

**在本地复现。** 安装下方所列依赖，从仓库根目录按顺序运行。原始工作簿在其他位置时，前两条命令可传 `--workbook`。清洗和 EDA 支持用 `--project-dir` 指定产物根目录。

```bash
python -m scripts.analyze_exceptions
python -m scripts.prepare_data
python -m scripts.analyze_training
python -m scripts.check_net_retention
```

**如何理解取消和退货。** 在同客户、同商品中寻找时间不晚于 C 单且可冲销余量足够的一条最近购买。可跨价格匹配。同分钟先选 Excel 行号在 C 单之前的正单，再取最小行号。英国、指定的 23 个欧洲地区、其他已知地区的取消上限分别为 72、120、240 小时，等于上限仍算取消，超过上限为项目定义的退货。发货、签收及实物退回无法从工作簿观察。匿名、未匹配和未知地区保留独立审计状态。[整表对账](results/reversal_benchmark.csv) · [商品 C 单匹配明细](results/credit_allocations.csv.gz) · [匹配规则差异](results/reversal_match_differences.csv)

全表共有 8,665 条商品 C 单明细，分类结果为取消 1,473 条、按项目定义的退货 5,883 条、已知客户未匹配 1,147 条、匿名 162 条。发票 `541431` 和 `581483` 两笔巨额购买在相应冲销后有效件数均为 0。有 12 条 C 单与原购买记录在同一分钟。

**如何理解金额与时间。** 有效购买按扣除所有已匹配冲销后的剩余件数乘以原购买单价计算。经营账则按每条销售和贷项各自的单价计算净额。目标是净有效成交（Net Valid Purchase）：目标月结束时，客户在该月的合格购物篮中仍至少保有一件该商品，才将其列为正例。同月另一行或另一张发票上有有效余量，仍可构成该客户的正例。这里的“留存”根据工作簿中的销售与冲销推算，实物是否退还或留在客户手中无法核实。截止时间后的 C 单只影响较晚的快照。数据摘要和每份产物的校验值见清单。[清洗产物清单](results/data_manifest.json)

**阈值敏感性核验。** 将英国的 72 小时改为 120 小时，有 608 条贷项改变分类。核验逐项比较冲销分配、全部篮内事件字段，以及八个训练月份和十月验证月在候选筛选前的月末正例。四个主模型都只使用与地区阈值无关的购买事件字段。依赖地区时限的取消／退货标签、退货计数和退货率作审计用。如果以后将它们加入 LightGBM 的验证集特征对照实验，该实验的特征和得分可能随时限改变。[敏感性对照结果](results/net_retention_threshold_check.csv) · [复现入口](scripts/check_net_retention.py)

### D2 · 月度查询与统一评价

完成 D1 后，在 CPU 运行时顺序执行月度查询 notebook，跟随一位客户核对历史、候选商品、目标月正例及共用的评价输入。若要从 D1 已知客户账与冲销分配表重新生成同一组表，在仓库根目录运行下方命令。本次构建覆盖二月至九月的训练目标和十月验证。本次保存十月验证的 AUC 抽样。十一月标签、十一月 AUC 抽样及测试得分留给选模冻结后的测试。[月度查询与评价 notebook](notebooks/monthly_queries_and_evaluation.ipynb) · [数据协议](docs/data_protocol.zh-CN.md) · [查询构建脚本](scripts/build_queries.py)

```bash
python scripts/build_queries.py --project-root .
python -m scripts.audit_label_delay
```

查询构建脚本默认输出到 `data/processed/d2/`。`manifest.json` 记录版本及输入输出的 SHA-256 校验值，`coverage.csv` 给出各组客户的分母，查询、历史、标签、候选目录、正负样本及历史特征分别写入 Parquet 文件。可用 `--output-dir` 写到另一目录以比较两次构建。已生成 5,430 条训练查询；十月有 1,357 位活跃客户，其中 758 位进入主验证，面对同一份含 3,495 种商品的候选目录。notebook 的示例分数用于演示评价器。[D2 清单](data/processed/d2/manifest.json) · [月度覆盖统计](data/processed/d2/coverage.csv)

**如何解释十月与十一月的差异。** 两个月都只使用 2011 年 10 月 1 日零时以前的历史，十月预测下一月（T+1），十一月预测第二个月（T+2）。距上次购买的天数是最后一笔净保留购买到 `history_end` 的实际时间差，保留不足一天的部分。7、30、90 天窗口也都在这一截止点结束。因此，同一客户在两个月共用相同的历史、候选目录及历史特征。将来如果得分有差异，预测跨度、圣诞前备货，以及参与购买的客户和商品发生变化，都是可能因素，仅凭这一对月份无法分离各自的影响。原表在 12 月 9 日 12:50 截止，可观察到的十一月购买事后冲销时间更短。十二月不参与主实验评分。

**如何解读跨月冲销。** 在限制客户资格及候选目录前，二月至十月共有 246,998 个不同的“月份–客户–商品”月末正例，计入后来可观察到的冲销后，其中 758 个归零。项目负责人另行提供的整表核对涵盖二月至十一月，结果为 788/301,922（0.26%），其中十一月为 30 个。这些比率衡量记录中的正例变化。要计算 Recall@10 或 NDCG@10 的变化，还需要逐查询的合格正例及模型排名。模型选择只使用十月验证。[测试前审计表](results/d2_audit/label_delay_pretest.csv) · [跨月冲销审计](scripts/audit_label_delay.py) · [定义与分母](docs/data_protocol.zh-CN.md)

### D3–D5 · 模型试训

在 Colab 的 GitHub 选项中，选择 `feat/upstream-sequence-models` 分支并打开 `notebooks/model_pilots.ipynb`。选择 `smoke`，每次运行一个模型。MF 与 LightGBM 使用 CPU，两种序列模型的 Colab 试训使用 NVIDIA GPU。依次运行所有单元格，notebook 会校验 D2，将 LRURec 源码缓存在 Drive 项目的 `cache/lrurec/` 中，本次运行目录为 `experiments/model_pilots/<模型>/<会话>/`，其中的 `session.log` 和 `model/run.json` 分别记录运行日志与验证结果。[模型试训 notebook](notebooks/model_pilots.ipynb)

**CPU 运行。** 安装下方模型依赖后，从仓库根目录执行以下命令。LRURec 的源码命令会获取作者的固定文件并校验摘要。可用 `LRUREC_CACHE_DIR` 指定持久缓存，或用 `LRUREC_REFERENCE_FILE` 指定已有的同版本离线文件。[MF、LightGBM 试训结果](results/model_pilots/summary.csv) · [序列模型运行记录](results/model_pilots_upstream/summary.csv)

```bash
python -m scripts.train --model mf --device cpu --smoke
python -m scripts.train --model lightgbm --device cpu --smoke
python -m src.models.lrurec_source
python -m scripts.train --model lrurec --device cpu --smoke
```

**GPU 核验。** 在 Colab 中先准备对应模型的独立环境。Mamba4Rec 使用仓库附带的作者 MIT 许可源码和真正的 RecBole 依赖。先用预检核对原生 GPU 前向计算、梯度、参数更新与重载，再运行真实 D2 试训。[LRURec 适配说明](docs/lrurec_adaptation.zh-CN.md) · [Mamba4Rec 运行环境](docs/mamba4rec_adaptation.zh-CN.md)

```bash
python -m scripts.train --model lrurec --device cuda --smoke
python -m src.models.mamba4rec_adapter --preflight --output runs/mamba4rec/preflight.json
python -m scripts.train --model mamba4rec --device cuda --smoke
```

命令行默认读取 `data/processed/d2/`，将每次结果写入 `runs/<模型>/<UTC 时间戳>/`。可用 `--data-dir`、`--output-dir` 改用其他位置。续跑时指定同一输出目录并加上 `--resume`。成功运行后可查看 `metrics.csv`、`query_metrics.csv`、`top10.parquet`、`auc_scores.parquet` 及最佳模型文件（`best.pt` 或 LightGBM 的 `best.txt`）。省略 `--smoke` 或在 notebook 中选择 `baseline` 会用完整训练数据拟合一套配置。D6 将比较预先登记的配置和种子，选定模型后再打开十一月测试标签。[训练入口](scripts/train.py) · [模型配置](configs/)

### 技术文档 · 核对一张发票

1. 在 Google Colab 选择 CPU 运行时，打开交易明细 notebook。在「文件 → 打开笔记本」中，可以选择 GitHub 标签页并粘贴该文件的 GitHub 地址，也可以上传下载的 `.ipynb` 文件。[原始交易明细与购物篮](docs/technical/transaction_records.ipynb)
2. 将原始工作簿放在 `我的云端硬盘/CS5481_Project2_OnlineRetail/data/raw/Online Retail.xlsx`，运行第一个代码单元格并授权挂载 Drive。如果工作簿在其他位置，出现输入框时从 Colab「文件」面板复制 Excel 路径并粘贴。[仓库中的工作簿](<data/raw/Online Retail.xlsx>)
3. 按顺序运行其余单元格。SHA-256 校验确认工作簿与计划中的版本一致，读取结果应为 **541,909 行、8 列**。客户 `17850` 的发票 `536365` 有 **7 行、7 种商品、40 件，总金额 £139.12**。

这份发票核验记录含有 Colab 会话中单独挂载 Drive 的代码格。在本地 Jupyter 中，跳过单独的 `drive.mount` 单元格，从仓库内运行后面的工作簿选择代码即可。

提交时如需这份 notebook 的 Python 脚本，可在 Colab 选择「文件 → 下载 → 下载 .py」。本地已安装 Jupyter nbconvert 时，也可以在仓库根目录执行以下命令。如需在本地运行导出的发票脚本，请在导出前移除单独挂载 Colab Drive 的代码格。脚本会在终端打印发票明细和汇总。保留原 notebook，可一并提交带有排版的技术说明。

```bash
jupyter nbconvert --to python docs/technical/transaction_records.ipynb
python docs/technical/transaction_records.py
```

## 软件环境要求

在仓库根目录安装 CPU 运行环境，或安装已包含 CPU 依赖的开发环境。直接依赖已固定版本，适用于 Python 3.12；本地参考解释器为 Python 3.12.14。[CPU 依赖](requirements.txt) · [测试与 notebook 依赖](requirements-dev.txt)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

如需在本地执行 notebook、导出 Python 或运行测试，请在同一环境运行 `python -m pip install -r requirements-dev.txt`，并在本地 notebook 界面选用此环境的 Python 内核。测试命令为 `python -m pytest -q`。D1、D2 在 CPU 上使用 NumPy、pandas、openpyxl、Matplotlib 和 PyArrow。

在 Linux CPU 上复现模型试训时，还需在当前环境安装模型依赖及 CPU 版 PyTorch。GPU notebook 分别为两种序列模型准备环境，Mamba4Rec 使用其说明中对应的 CUDA 版 PyTorch 和原生二进制包。[模型依赖](requirements-models.txt) · [Mamba4Rec GPU 安装说明](docs/mamba4rec_adaptation.zh-CN.md)

```bash
python -m pip install -r requirements-models.txt
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

## 随机种子

工作簿核验、发票示例、清洗审计及训练期 EDA 均使用确定性计算。D2 用 `42` 抽取共用训练负例，用 `5481` 固定验证 AUC 样本。按模型实验计划，超参数搜索使用训练种子 `42`，入选配置使用 `42`、`2026`、`3407` 复核，计算 ROC AUC 时另用种子 `5481` 抽取负例。

## 数据集引用

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. DOI `10.24432/C5BW33`。许可为 CC BY 4.0。交易明细 notebook 读取原始工作簿，选取一张发票进行核验。[数据来源与署名](https://doi.org/10.24432/C5BW33)

## 结果表格和图形的位置

| 状态 | 位置 | 内容 |
|---|---|---|
| 已核验 | [results/transaction_exceptions.xlsx](results/transaction_exceptions.xlsx) | 筛选前的证据表、历史发票回查、备注线索和客户指标敏感性。 |
| 已核验 | [results/cleaning_audit.csv](results/cleaning_audit.csv) | 五项规则的原始命中数、本步实际移除行数和剩余行数。 |
| 已核验 | [cleaning_steps.csv](results/cleaning_steps.csv)、[merchandise_rules.csv](results/merchandise_rules.csv) | 清洗各步的计数和商品分类依据。 |
| 已核验 | [data_manifest.json](results/data_manifest.json) | 输入工作簿的标识和本次清洗摘要。 |
| 已核验 | [row_decisions.csv.gz](data/processed/row_decisions.csv.gz) | 全部原始明细的 Excel 行号与清洗决定。 |
| 已核验 | [purchase_lines.csv.gz](data/processed/purchase_lines.csv.gz)、[basket_events.csv.gz](data/processed/basket_events.csv.gz) | 保留的购买明细和按客户、发票、商品聚合的购物篮事件。 |
| 已核验 | [经营账 A](data/processed/signed_ledger_a.csv.gz) · [客户账 B](data/processed/signed_ledger_b.csv.gz) · [10 月 1 日购物篮快照](data/processed/training_basket_events.csv.gz) | 正负经营／客户明细，以及 10 月 1 日可见的有效购买。 |
| 已核验 | [整表对账](results/reversal_benchmark.csv) · [商品 C 单匹配明细](results/credit_allocations.csv.gz) · [匹配规则差异](results/reversal_match_differences.csv) | 按原始 Excel 行号核验冲销分配、分类与分母。 |
| 本地已核验 | [训练期 EDA 输出目录](results/training_eda/) | 训练窗口总量及分母、月度统计、分布与商品热度表和图，格式为 JSON、CSV 和 PNG。 |
| 运行后可查看 | [交易明细 notebook](docs/technical/transaction_records.ipynb) 的输出单元格 | 工作表行列数、发票明细和购物篮汇总。 |
| 已本地核验 | [D2 清单](data/processed/d2/manifest.json) · [月度覆盖统计](data/processed/d2/coverage.csv) · [Parquet 表](data/processed/d2/) | 各截止点的共用输入、客户覆盖分母、可复现的训练与验证抽样。 |
| CPU 试训记录 | [MF、LightGBM](results/model_pilots/summary.csv) | 同表保留 `69fcba3` 的旧版独立实现 LRURec，供历史对照。 |
| 作者版本序列模型 | [运行状态与指标](results/model_pilots_upstream/summary.csv) | 按模型保存 LRURec CPU 试训及 Mamba4Rec 环境检查，以各行的设备和完成状态为准。 |
| 计划生成 | `results/test_metrics.csv` | 在冻结的测试协议下比较四个模型。 |
| 计划生成 | `results/cohort_coverage.csv`、`results/segment_metrics.csv`、`results/case_study.csv` | 客户覆盖、分组指标和个案推荐结果。 |
| 计划生成 | `experiments/exp_XXX/diagnostics_panel.png` | 各次实验的诊断图。 |

## 技术文档与 Canvas 提交材料

技术报告草稿汇总已实测的 D1、D2 结果，供后续整理 Canvas 提交稿。数据协议给出复现查询和指标的定义，研究计划说明后续模型实验及其验收标准。[技术报告草稿](docs/technical_report.zh-CN.md) · [数据协议](docs/data_protocol.zh-CN.md) · [研究计划](docs/CS5481_Project2_Research_Development_Plan_3.4.md)

项目自编文档分别提供英文版和中文版。技术文档 notebook 的每个 Markdown 单元格先呈现完整的英文标题和正文，再呈现完整的中文标题和正文。两种语言共用同一份代码。[技术文档](docs/technical/transaction_records.ipynb)

Canvas 提交包将包含技术报告、源代码、README，以及项目采用且课程允许提交的演示材料。后续每份交付物先核对课程文件第 3 页的相应要求，为每项回答标明可复现的证据，再说明项目额外完成的分析和仍待测量的指标。[项目二评分要求](resource/Group_Project_Tutorial_Recommender_Systems.pdf#page=3)
