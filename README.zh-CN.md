# 在线零售商品 Top-10 推荐

[English](README.md) · [简体中文](README.zh-CN.md)

本 CS5481 项目比较 MF、LightGBM、LRURec 和 Mamba4Rec，目标是根据客户历史为后续月份推荐十种商品。当前可核验原始工作簿、复查一张发票、调查特殊交易的业务背景，并审计五条拟用于购买样本的剔除规则。研究计划记录后续要实现的训练和评估流程。[研究与代码开发计划](docs/CS5481_Project2_Research_Development_Plan_3.3.md)

## 运行说明

### D1 · 筛选之前先分析业务情形

先打开特殊交易分析 notebook，选择 CPU，用原始工作簿从上往下运行。17 张 Excel 工作表的计算过程均可在代码单元格中查看。输出 Excel 分别列出 C 前缀、数量非正和单价非正的完整明细，保留原始 Excel 行号。这三个视图允许重叠，另有互斥的符号交叉表用来核对总数。1,336 条非 C 负数量明细有单独的表，可以按时间追查发票与商品、查阅原始备注并对照案例时间线。价格分析列出有客户编号的全部 40 条零价明细，逐行检查同票是否还有正价商品，并用发票 `550188` 对照商品 `22636` 的两种单价。后续章节分析异常记录对客户活动统计的影响。[分析 notebook](docs/technical/transaction_exception_analysis.ipynb) · [可筛选的 Excel 结果](results/transaction_exceptions.xlsx)

运行后，Drive 项目目录下会生成 `results/transaction_exceptions.xlsx`。本地可在仓库根目录执行 `python -m scripts.analyze_exceptions`。此次调查读取全部原始行，包含完全重复的行。下方五步审计衡量的是从正向购买样本剔除记录时影响了多少行。两项操作都保留原始工作簿。

### D1 · 交易明细清洗审计

在 Colab 选择 CPU，打开项目主 notebook，按顺序运行单元格。输入沿用下方交易明细技术文档使用的 Drive 工作簿。程序加载指定版本的审计代码、核验原始工作簿，在 Drive 项目目录下生成 `results/cleaning_audit.csv`。如果 Colab 提示上传 `clean.py`，从单元格显示的 GitHub 链接下载原始文件，在上传框选中即可。[项目主 notebook](notebooks/project2_colab.ipynb)

本地复现时，使用已安装 pandas 和 openpyxl 的 Python 3.11 或更高版本，在仓库根目录执行：

```bash
python -m scripts.prepare_data
```

按步骤顺序读 CSV。`raw_flagged` 统计原表中符合规则的所有行，`rows_removed` 只统计到这一步才被移除的行。五步合计移除 **149,217 行**，原表 **541,909 行**中还剩 **392,692 行**，供后续筛选有效商品和聚合购物篮。[审计表](results/cleaning_audit.csv) · [规则实现](src/data/clean.py)

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
| 模型实验（计划） | MF、LightGBM 使用 Colab CPU，LRURec、Mamba4Rec 使用 Colab NVIDIA GPU。各模型实际验证通过的依赖版本将随训练代码记录。 |

## 随机种子

工作簿核验、发票示例及清洗审计都使用确定性计算。按实验计划，超参数搜索使用训练种子 `42`，入选配置使用 `42`、`2026`、`3407` 复核，计算 ROC AUC 时另用种子 `5481` 抽取负例。

## 数据集引用

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. DOI `10.24432/C5BW33`。许可为 CC BY 4.0。交易明细 notebook 读取原始工作簿，选取一张发票进行核验。[数据来源与署名](https://doi.org/10.24432/C5BW33)

## 结果表格和图形的位置

| 状态 | 位置 | 内容 |
|---|---|---|
| 已核验 | [results/transaction_exceptions.xlsx](results/transaction_exceptions.xlsx) | 筛选前的证据表、历史发票回查、备注线索和客户指标敏感性。 |
| 已核验 | [results/cleaning_audit.csv](results/cleaning_audit.csv) | 五项规则的原始命中数、本步实际移除行数和剩余行数。 |
| 运行后可查看 | [交易明细 notebook](docs/technical/transaction_records.ipynb) 的输出单元格 | 工作表行列数、发票明细和购物篮汇总。 |
| 计划生成 | `results/test_metrics.csv` | 在冻结的测试协议下比较四个模型。 |
| 计划生成 | `results/cohort_coverage.csv`、`results/segment_metrics.csv`、`results/case_study.csv` | 客户覆盖、分组指标和个案推荐结果。 |
| 计划生成 | `experiments/exp_XXX/diagnostics_panel.png` | 各次实验的诊断图。 |

## 技术文档与 Canvas 提交材料

项目自编文档分别提供英文版和中文版。技术文档 notebook 的每个 Markdown 单元格先呈现完整的英文标题和正文，再呈现完整的中文标题和正文。两种语言共用同一份代码。[技术文档](docs/technical/transaction_records.ipynb)

Canvas 提交包将包含技术报告、源代码、README，以及项目采用且课程允许提交的演示材料。按研究计划，可复用的数据处理、模型和评估逻辑写成 Python 模块，notebook 负责运行与解释。
