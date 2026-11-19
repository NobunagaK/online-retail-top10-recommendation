# 在线零售商品 Top-10 推荐

[English](README.md) · [简体中文](README.zh-CN.md)

本 CS5481 项目比较 MF、LightGBM、LRURec 和 Mamba4Rec，目标是根据客户历史为后续月份推荐十种商品。数据准备流程逐行记录清洗决定，并生成四种模型共用的正价购买购物篮事件。研究计划记录后续要实现的训练和评估流程。[研究与代码开发计划 v3.4](docs/CS5481_Project2_Research_Development_Plan_3.4.md)

## 运行说明

### D1 · 筛选之前先分析业务情形

先打开特殊交易分析 notebook，选择 CPU，用原始工作簿从上往下运行。17 张 Excel 工作表的计算过程均可在代码单元格中查看。输出 Excel 的 `C_invoices`、`Q_nonpositive`、`P_nonpositive` 三张表分别列出 C 前缀、数量非正和单价非正的完整明细。三张表可以重叠。`sign_combinations` 用互斥分组核对总数，`ExcelRow` 可定位原始工作表行号。1,336 条非 C 负数量明细有单独的表，可以按时间追查发票与商品、查阅原始备注并对照案例时间线。价格分析列出有客户编号的全部 40 条零价明细，逐行检查同票是否还有正价商品，并用发票 `550188` 对照商品 `22636` 的两种单价。后续章节分析异常记录对客户活动统计的影响。[分析 notebook](docs/technical/transaction_exception_analysis.ipynb) · [可筛选的 Excel 结果](results/transaction_exceptions.xlsx)

运行后，Drive 项目目录下会生成 `results/transaction_exceptions.xlsx`。本地可在仓库根目录执行 `python -m scripts.analyze_exceptions`。此次调查读取全部原始行，包含完全重复的行。下方五步审计衡量的是从正向购买样本剔除记录时影响了多少行。两项操作都保留原始工作簿。

### D1 · 生成正价购买明细和购物篮事件

在 Colab 选择 CPU，打开项目主 notebook，从上到下运行，按提示挂载 Drive。默认输入路径为 `我的云端硬盘/CS5481_Project2_OnlineRetail/data/raw/Online Retail.xlsx`。notebook 核验工作簿，读取 `Online Retail` 工作表，在代码单元格中展示清洗与聚合的计算过程。生成的文件保存在项目目录下，具体位置见下方结果表。[项目主 notebook](notebooks/project2_colab.ipynb)

要在本地复现，使用已安装 pandas 和 openpyxl 的 Python 3.11 或更高版本，在仓库根目录执行以下命令。

```bash
python -m scripts.prepare_data
```

如原始工作簿位于其他位置，可用 `--workbook` 指定路径。用 `--project-dir` 指定生成文件的项目目录，用 `--output` 单独指定五步审计 CSV 的路径。[本地运行入口](scripts/prepare_data.py)

**如何核对行数。** 原表共有 **541,909 行交易明细**。同一行可能符合多项条件。五步基线审计中，`raw_flagged` 统计原表中符合该项条件的全部行，`rows_removed` 把每一行只计入它最先命中的步骤。五步共排除 **149,217 行**，进入商品筛选前剩 **392,692 行**。在 `cleaning_steps.csv` 中，先看进入本步的行数 `rows_before`，再核对本步排除数 `rows_removed` 和剩余数 `rows_after`。`evidence_sheets` 给出作为依据的异常工作表或训练期编码登记表，`reason_en` 和 `reason_zh` 用中英文说明理由。用 `row_decisions.csv.gz` 可按 Excel 原始行号回查处理决定。商品规则进一步排除 **1,632 行**，最后得到 **391,060 行正价购买明细**。[逐步清洗与证据](results/cleaning_steps.csv) · [五步基线审计](results/cleaning_audit.csv)

**如何使用清洗结果。** `purchase_lines.csv.gz` 的每一行是一条保留的正价购买明细。`basket_events.csv.gz` 的每一行对应一位客户在一张发票中的一种商品。聚合时分别累加件数和各行金额（数量 × 单价）。该客户在同张发票上最后一条保留明细的时间为整篮完成时间。全表共得到 **386,236 个事件**，涉及 **18,402 个购物篮、4,334 位客户和 3,658 种商品**。MF、LightGBM、LRURec 和 Mamba4Rec 共用这份购物篮事件表构建购买历史。训练和评估属于后续步骤。

### D1 · 探索训练期购买数据

在 Colab 选择 CPU，打开训练期 EDA notebook，从上往下运行。使用 `data/processed/` 中的两份 D1 产物和 `results/data_manifest.json` 内的校验值，或使用 notebook 提供的经校验的同版本文件。在本地复现时，从仓库根目录运行下方命令。如果清洗产物位于其他项目文件夹，用 `--project-dir` 指定该目录。程序在所选项目的 `results/training_eda/` 下写入表格和图形。[训练期 EDA notebook](notebooks/training_eda.ipynb) · [本地运行入口](scripts/analyze_training.py)

```bash
python -m scripts.analyze_training
```

**统计范围。** 选取 2010 年 12 月 1 日起、2011 年 10 月 1 日前完成的正价商品购物篮，按这些购物篮取对应的原始购买明细。训练窗口共含 **262,597 条购买明细、260,199 个购物篮–商品事件和 13,081 个购物篮**。十月验证、十一月测试的活动在该窗口之外。

**如何解读输出。**

1. 从月度表查看活跃客户、购物篮、事件、件数与金额（英镑）的变化。同一客户可在多个月活跃，把每月客户数相加会重复计算回头客。[月度统计表](results/training_eda/monthly_activity.csv) · [月度交易量图](results/training_eda/monthly_volume.png)
2. 阅读分布时先确认统计单位。单价和购买件数取自每条明细；篮大小为一张客户发票中不同商品编码数；客户历史长度为该客户的购物篮–商品事件数。正价商品单价中位数为 £1.95，篮大小中位数为 15 种商品，客户历史长度中位数为 34 个事件。[分布摘要](results/training_eda/distribution_summary.csv) · [各取值频数](results/training_eda/distributions.csv) · [分布图](results/training_eda/size_distributions.png)
3. 商品热度按购买该商品的不同客户数排序。最热门的 351 种商品占训练期事件的 46.71%。3,613 位客户与 3,501 种商品可组成 12,649,113 个客户–商品对，实际观察到 188,368 个，**稀疏度为 98.51%**。若同一客户在完成时间**严格更早**的购物篮中买过同一商品，本次事件计作复购，共 **71,824 / 260,199 个事件（27.60%）**。下一阶段 D2 将按目标月构造查询与标签，编写统一评价器。[商品热度表](results/training_eda/item_popularity.csv) · [商品热度图](results/training_eda/item_popularity.png) · [汇总及分母](results/training_eda/summary.json)

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

| 组件 | 运行环境 |
|---|---|
| 工作簿分析 | Google Colab CPU，或本地 Python 与 Jupyter。分析代码已在本地用 Python 3.12.14、pandas 2.2.3、openpyxl 3.1.5 验证。 |
| 训练期 EDA | Python、pandas 和 Matplotlib。在 CPU 上使用 Jupyter 或 Colab 运行 notebook。 |
| 模型实验（计划） | MF、LightGBM 使用 Colab CPU，LRURec、Mamba4Rec 使用 Colab NVIDIA GPU。各模型实际验证通过的依赖版本将随训练代码记录。 |

## 随机种子

工作簿核验、发票示例、清洗审计及训练期 EDA 均使用确定性计算。按实验计划，超参数搜索使用训练种子 `42`，入选配置使用 `42`、`2026`、`3407` 复核，计算 ROC AUC 时另用种子 `5481` 抽取负例。

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
| 本地已核验 | [训练期 EDA 输出目录](results/training_eda/) | 训练窗口总量及分母、月度统计、分布与商品热度表和图，格式为 JSON、CSV 和 PNG。 |
| 运行后可查看 | [交易明细 notebook](docs/technical/transaction_records.ipynb) 的输出单元格 | 工作表行列数、发票明细和购物篮汇总。 |
| 计划生成 | `results/test_metrics.csv` | 在冻结的测试协议下比较四个模型。 |
| 计划生成 | `results/cohort_coverage.csv`、`results/segment_metrics.csv`、`results/case_study.csv` | 客户覆盖、分组指标和个案推荐结果。 |
| 计划生成 | `experiments/exp_XXX/diagnostics_panel.png` | 各次实验的诊断图。 |

## 技术文档与 Canvas 提交材料

项目自编文档分别提供英文版和中文版。技术文档 notebook 的每个 Markdown 单元格先呈现完整的英文标题和正文，再呈现完整的中文标题和正文。两种语言共用同一份代码。[技术文档](docs/technical/transaction_records.ipynb)

Canvas 提交包将包含技术报告、源代码、README，以及项目采用且课程允许提交的演示材料。按研究计划，可复用的数据处理、模型和评估逻辑写成 Python 模块，notebook 负责运行与解释。
