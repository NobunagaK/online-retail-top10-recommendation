# Online Retail Top-10 Recommendation

[English](README.md) · [简体中文](README.zh-CN.md)

This CS5481 project compares MF, LightGBM, LRURec, and Mamba4Rec for recommending ten products to a customer for a future month. The project currently verifies the raw workbook, inspects an invoice, investigates exceptional transactions, and audits five proposed row-removal rules. The research plan describes the training and evaluation workflow under development. [Research and development plan](docs/CS5481_Project2_Research_Development_Plan_3.3.en.md)

## Run instructions

### D1 · Investigate business context before filtering

Open the exception analysis notebook on a CPU runtime and run it from top to bottom with the original workbook. Its code cells calculate all 17 Excel worksheets. Its Excel output contains overlapping row-level views for C invoices, nonpositive quantities, and nonpositive prices, a disjoint sign table for reconciliation, and the original Excel row numbers. The 1,336 negative-quantity lines outside C invoices have their own sheet. Strictly earlier invoice/SKU matches, literal notes, and case timelines let the reader inspect the evidence for each interpretation. The price analysis displays all 40 zero-price lines with known customers, checks each invoice for priced lines, and traces both entries for SKU `22636` on invoice `550188`. The notebook also measures how exceptional lines affect customer activity. [Analysis notebook](docs/technical/transaction_exception_analysis.ipynb) · [Filterable results workbook](results/transaction_exceptions.xlsx)

The notebook writes `results/transaction_exceptions.xlsx` inside the Drive project. Locally, run `python -m scripts.analyze_exceptions` from the repository root. The analysis reads all original rows including duplicates. The five-rule audit below counts how many rows would be excluded from a positive-purchase sample. Both procedures leave the original workbook intact.

### D1 · Audit transaction rows

Open the project notebook in a Colab CPU runtime and run the cells in order. Use the same Drive workbook as the transaction reference below. The notebook loads the versioned audit code, verifies the original workbook, then writes `results/cleaning_audit.csv` in the Drive project folder. If Colab prompts for `clean.py`, download the raw file from the GitHub link shown in the cell and select it in the upload dialog. [Project notebook](notebooks/project2_colab.ipynb)

To reproduce the audit locally, run the following from the repository root using Python 3.11 or later with pandas and openpyxl installed.

```bash
python -m scripts.prepare_data
```

Read the CSV in step order. `raw_flagged` counts every match on the original sheet. `rows_removed` counts only rows that reach that step. The five removal counts sum to **149,217**, leaving **392,692** of the **541,909** rows for subsequent merchandise filtering and basket aggregation. [Audit table](results/cleaning_audit.csv) · [Rule implementation](src/data/clean.py)

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
| Available on execution | Output cells in [the transaction notebook](docs/technical/transaction_records.ipynb) | Workbook dimensions, invoice lines, and basket totals. |
| Planned | `results/test_metrics.csv` | Four-model comparison on the frozen test protocol. |
| Planned | `results/cohort_coverage.csv`, `results/segment_metrics.csv`, `results/case_study.csv` | Population coverage, metrics by group, and individual recommendations. |
| Planned | `experiments/exp_XXX/diagnostics_panel.png` | Diagnostic figure for an experiment. |

## Technical documentation and Canvas submission

Project-authored documents have English and Chinese editions. Each Markdown cell in a technical notebook presents the full English section followed by the full Chinese section. Both versions share the same code. [Technical reference](docs/technical/transaction_records.ipynb)

The Canvas package will include the technical report, source code, README, and any permitted demonstration materials used. The plan places reusable data, model, and evaluation code in Python modules and uses notebooks to run and explain them.
