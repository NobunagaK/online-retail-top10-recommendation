# 在线零售商品 Top-10 推荐

[English](README.md) · [简体中文](README.zh-CN.md)

本 CS5481 项目比较 MF、LightGBM、LRURec 和 Mamba4Rec，目标是根据客户历史为后续月份推荐十种商品。当前可运行的部分核验原始工作簿并复查一张发票。研究计划记录后续要实现的训练和评估流程。[研究与代码开发计划](docs/CS5481_Project2_Research_Development_Plan_3.3.md)

## 运行说明

1. 在 Google Colab 选择 CPU 运行时，打开交易明细 notebook。在「文件 → 打开笔记本」中，可以选择 GitHub 标签页并粘贴该文件的 GitHub 地址，也可以上传下载的 `.ipynb` 文件。[原始交易明细与购物篮](docs/technical/transaction_records.ipynb)
2. 将原始工作簿放在 `我的云端硬盘/CS5481_Project2_OnlineRetail/data/raw/Online Retail.xlsx`，运行第一个代码单元格并授权挂载 Drive。如果工作簿在其他位置，出现输入框时从 Colab「文件」面板复制 Excel 路径并粘贴。[仓库中的工作簿](<data/raw/Online Retail.xlsx>)
3. 按顺序运行其余单元格。SHA-256 校验确认工作簿与计划中的版本一致，读取结果应为 **541,909 行、8 列**。客户 `17850` 的发票 `536365` 有 **7 行、7 种商品、40 件，总金额 £139.12**。

本地运行时，在仓库根目录启动 Jupyter 并打开同一份 notebook。代码会自动在仓库的 `data/raw/` 中查找工作簿。

提交时如需这份 notebook 的 Python 脚本，可在 Colab 选择「文件 → 下载 → 下载 .py」。本地已安装 Jupyter nbconvert 时，也可以在仓库根目录执行以下命令。脚本会在终端打印发票明细和汇总。保留原 notebook，可一并提交带有排版的技术说明。

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

工作簿和发票核验使用确定性计算。按实验计划，超参数搜索使用训练种子 `42`，入选配置使用 `42`、`2026`、`3407` 复核，计算 ROC AUC 时另用种子 `5481` 抽取负例。

## 数据集引用

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. DOI `10.24432/C5BW33`。许可为 CC BY 4.0。交易明细 notebook 读取原始工作簿，选取一张发票进行核验。[数据来源与署名](https://doi.org/10.24432/C5BW33)

## 结果表格和图形的位置

| 状态 | 位置 | 内容 |
|---|---|---|
| 运行后可查看 | [交易明细 notebook](docs/technical/transaction_records.ipynb) 的输出单元格 | 工作表行列数、发票明细和购物篮汇总。 |
| 计划生成 | `results/test_metrics.csv` | 在冻结的测试协议下比较四个模型。 |
| 计划生成 | `results/cohort_coverage.csv`、`results/segment_metrics.csv`、`results/case_study.csv` | 客户覆盖、分组指标和个案推荐结果。 |
| 计划生成 | `experiments/exp_XXX/diagnostics_panel.png` | 各次实验的诊断图。 |

## 技术文档与 Canvas 提交材料

项目自编文档分别提供英文版和中文版。可运行的技术文档按英文在前、中文在后排列说明，两种语言共用一份代码。[技术文档](docs/technical/transaction_records.ipynb)

Canvas 提交包将包含技术报告、源代码、README，以及项目采用且课程允许提交的演示材料。按研究计划，可复用的数据处理、模型和评估逻辑写成 Python 模块，notebook 负责运行与解释。
