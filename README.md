# Online Retail Top-10 Recommendation

[English](README.md) · [简体中文](README.zh-CN.md)

This CS5481 project compares MF, LightGBM, LRURec, and Mamba4Rec for recommending ten products to a customer for a future month. The data-preparation workflow traces each source row through cleaning and produces the paid basket events that all four models will use. The research plan describes the training and evaluation workflow under development. [Research and development plan v3.4](docs/CS5481_Project2_Research_Development_Plan_3.4.en.md)

## Run instructions

### D1 · Investigate business context before filtering

Open the exception analysis notebook on a CPU runtime and run it from top to bottom with the original workbook. Its code cells calculate all 17 Excel worksheets. In the Excel output, `C_invoices`, `Q_nonpositive` and `P_nonpositive` contain overlapping row-level views. `sign_combinations` gives mutually exclusive counts for reconciliation, and `ExcelRow` points back to the source worksheet. The 1,336 negative-quantity lines outside C invoices have their own sheet. Strictly earlier invoice/SKU matches, literal notes, and case timelines let the reader inspect the evidence for each interpretation. The price analysis displays all 40 zero-price lines with known customers, checks each invoice for priced lines, and traces both entries for SKU `22636` on invoice `550188`. The notebook also measures how exceptional lines affect customer activity. [Analysis notebook](docs/technical/transaction_exception_analysis.ipynb) · [Filterable results workbook](results/transaction_exceptions.xlsx)

The notebook writes `results/transaction_exceptions.xlsx` inside the Drive project. Locally, run `python -m scripts.analyze_exceptions` from the repository root. The analysis reads all original rows including duplicates. The five-rule audit below counts how many rows would be excluded from a positive-purchase sample. Both procedures leave the original workbook intact.

### D1 · Prepare paid purchase lines and basket events

Open the project notebook in a Colab CPU runtime and run its cells in order. Mount Drive when prompted. The default input is `My Drive/CS5481_Project2_OnlineRetail/data/raw/Online Retail.xlsx`. The notebook verifies the workbook, reads the `Online Retail` worksheet, and shows the cleaning and aggregation calculations in its code cells. It writes the tables listed below under the project folder. [Project notebook](notebooks/project2_colab.ipynb)

To reproduce the outputs locally, run this command from the repository root with Python 3.11 or later, pandas and openpyxl installed.

```bash
python -m scripts.prepare_data
```

Use `--workbook` to select a different path to the original workbook and `--project-dir` to choose the project folder for generated files. The optional `--output` sets the path of the five-rule audit CSV. [Local entry point](scripts/prepare_data.py)

**Reading the counts.** The raw denominator is **541,909 transaction lines**. A single line can match multiple conditions. The five-rule baseline audit counts these overlaps in `raw_flagged`. Its `rows_removed` column attributes each line to the first applicable step. The five steps exclude **149,217** lines and leave **392,692** before merchandise filtering. In `cleaning_steps.csv`, read each step against `rows_before`, then follow `rows_removed` and `rows_after`. The `evidence_sheets` column names the supporting worksheets or the training-period code register, and `reason_en` and `reason_zh` explain it. Trace a particular Excel row in `row_decisions.csv.gz`. Merchandise exclusions account for another **1,632 lines**, leaving **391,060 paid purchase lines**. [Detailed steps and reasons](results/cleaning_steps.csv) · [Five-rule baseline audit](results/cleaning_audit.csv)

**Using the cleaned data.** Each row of `purchase_lines.csv.gz` is one retained, positively priced purchase line. Each row of `basket_events.csv.gz` is one distinct product in one customer's invoice. The event sums quantities and line amounts (quantity × unit price) and uses the basket's completion time, the latest retained line timestamp on the invoice for that customer. The full workbook yields **386,236 events** in **18,402 baskets** for **4,334 customers** and **3,658 SKUs**. The basket events supply the common purchase history for MF, LightGBM, LRURec and Mamba4Rec. Training and evaluation are later stages.

### Technical reference · Inspect an invoice

1. Open the transaction notebook in Google Colab on a CPU runtime. In **File → Open notebook**, use the GitHub tab and the notebook's GitHub URL, or upload the downloaded `.ipynb` file. [Transaction records and baskets](docs/technical/transaction_records.ipynb)
2. Put the original workbook in `My Drive/CS5481_Project2_OnlineRetail/data/raw/Online Retail.xlsx`. Run the first code cell and authorize the Drive mount. For another location, copy the Excel file's path from Colab's Files pane when prompted. [Workbook in this repository](<data/raw/Online Retail.xlsx>)
3. Run the remaining cells in order. The SHA-256 check identifies the workbook used in the plan. Expect **541,909 rows and 8 columns**. Invoice `536365` for customer `17850` has **7 lines, 7 distinct products, 40 units and a total of £139.12**.

The saved invoice notebook includes the Drive mount cell from the Colab session. In local Jupyter, skip the standalone `drive.mount` cell and run the following workbook-selection cell from the checkout.

To export this notebook as a Python script at submission time, choose **File → Download → Download .py** in Colab. Alternatively, with Jupyter nbconvert installed locally, run the following from the repository root. For local execution of the exported invoice script, remove the standalone Colab mount cell before export. The script prints the invoice lines and totals to the terminal. Keep the notebook alongside the script to preserve its formatted documentation.

```bash
jupyter nbconvert --to python docs/technical/transaction_records.ipynb
python docs/technical/transaction_records.py
```

## Software requirements

| Component | Environment |
|---|---|
| Workbook analysis | Google Colab with a CPU, or local Jupyter with Python. The analysis was verified locally with Python 3.12.14, pandas 2.2.3, and openpyxl 3.1.5. |
| Model experiments (planned) | Colab CPU for MF and LightGBM, Colab with an NVIDIA GPU for LRURec and Mamba4Rec. Tested model-specific dependency versions will accompany the training code. |

## Random seeds

The workbook check, invoice example, and cleaning audit are deterministic. Planned experiments use training seed `42` for hyperparameter search, seeds `42`, `2026`, and `3407` to check selected configurations, and a separate seed `5481` to sample negatives for ROC AUC.

## Dataset citation

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. DOI `10.24432/C5BW33`. Licensed under CC BY 4.0. The transaction notebook reads the original workbook and selects one invoice for inspection. [Dataset and attribution](https://doi.org/10.24432/C5BW33)

## Result tables and figures

| Status | Location | Contents |
|---|---|---|
| Verified | [results/transaction_exceptions.xlsx](results/transaction_exceptions.xlsx) | Evidence sheets, invoice traces, description clues, and customer-level sensitivity before filtering. |
| Verified | [results/cleaning_audit.csv](results/cleaning_audit.csv) | Raw matches, net removals, and remaining rows for all five D1 audit rules. |
| Verified | [cleaning_steps.csv](results/cleaning_steps.csv), [merchandise_rules.csv](results/merchandise_rules.csv) | Step counts and the reasons for merchandise classifications. |
| Verified | [data_manifest.json](results/data_manifest.json) | Source workbook identity and a summary of the cleaning run. |
| Verified | [row_decisions.csv.gz](data/processed/row_decisions.csv.gz) | Source row numbers and cleaning decisions across the original worksheet. |
| Verified | [purchase_lines.csv.gz](data/processed/purchase_lines.csv.gz), [basket_events.csv.gz](data/processed/basket_events.csv.gz) | Retained purchase lines and aggregated invoice–product events. |
| Available on execution | Output cells in [the transaction notebook](docs/technical/transaction_records.ipynb) | Workbook dimensions, invoice lines, and basket totals. |
| Planned | `results/test_metrics.csv` | Four-model comparison on the frozen test protocol. |
| Planned | `results/cohort_coverage.csv`, `results/segment_metrics.csv`, `results/case_study.csv` | Population coverage, metrics by group, and individual recommendations. |
| Planned | `experiments/exp_XXX/diagnostics_panel.png` | Diagnostic figure for an experiment. |

## Technical documentation and Canvas submission

Project-authored documents have English and Chinese editions. Each Markdown cell in a technical notebook presents the full English section followed by the full Chinese section. Both versions share the same code. [Technical reference](docs/technical/transaction_records.ipynb)

The Canvas package will include the technical report, source code, README, and any permitted demonstration materials used. The plan places reusable data, model, and evaluation code in Python modules and uses notebooks to run and explain them.
