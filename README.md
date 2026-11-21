# Online Retail Top-10 Recommendation

[English](README.md) · [简体中文](README.zh-CN.md)

This CS5481 project compares MF, LightGBM, LRURec, and Mamba4Rec for recommending ten products to a customer for a future month. All four models share the same net-retained basket histories, candidate catalogs and monthly labels. Both sequence models use the authors’ pinned recommender code. D6 completed 48 formal fits. October validation selected LightGBM configuration `06`. On 935 eligible November queries it achieved **0.36294 ± 0.00220 NDCG@10** (three-seed mean ± sample SD). [Validation metrics](results/validation_metrics.csv) · [Locked test metrics](results/test_metrics.csv) · [Research and development plan v3.4](docs/CS5481_Project2_Research_Development_Plan_3.4.en.md)

## Run instructions

### D1 · From source transactions to training baskets

Project 2 asks how cancellations, returns, duplicate lines, missing customer IDs and anomalous quantities or prices are handled, how customer–product interactions are aggregated, and how history is built chronologically. Use the three notebooks in order. [Project 2 requirements, p. 3](resource/Group_Project_Tutorial_Recommender_Systems.pdf#page=3) · [Exception analysis](docs/technical/transaction_exception_analysis.ipynb) · [Cleaning and rubric answers](notebooks/project2_colab.ipynb) · [Training EDA](notebooks/training_eda.ipynb)

1. **Investigate exceptions.** Trace C invoices, negative quantities, zero prices, bad debt and inventory notes to the original Excel rows. The allocation tables identify each merchandise credit’s sale, remaining reversible units, elapsed time and region, and show which lines change under alternative rules. [Evidence workbook](results/transaction_exceptions.xlsx)
2. **Run D1 cleaning.** Review each rule, its evidence, affected rows and the five assignment answers. Ledger A includes merchandise transactions for anonymous and identified customers. Ledger B includes identified customers. Both count each signed line once at its recorded price. For each sale, deduct every linked credit issued before the cutoff, including returns. A positive remainder contributes one purchase event per customer, invoice and SKU after aggregation. A fully offset line contributes zero units. [Store ledger A](data/processed/signed_ledger_a.csv.gz) · [Customer ledger B](data/processed/signed_ledger_b.csv.gz) · [October 1 basket snapshot](data/processed/training_basket_events.csv.gz)
3. **Explore the training period.** The 1 October 2011 snapshot covers baskets completed from December 2010 through September 2011. Monthly activity, price and quantity distributions, item popularity, repeat purchases and sparsity use credits known at that cutoff. [Training counts](results/training_eda/summary.json) · [Monthly totals](results/training_eda/monthly_activity.csv) · [Item rankings](results/training_eda/item_popularity.csv)

**Run in Colab.** In **File → Open notebook → GitHub**, select this repository and the branch containing the files. Open the notebooks above in order and run all cells on CPU. The default project root is `My Drive/CS5481_Project2_OnlineRetail/`, with the workbook at `data/raw/Online Retail.xlsx`. The exception and cleaning notebooks each start from that workbook. The EDA notebook reads the cleaning outputs. Generated files go to `data/processed/` and `results/` under the project root.

**Run locally.** Install the dependencies listed below and run these commands from the repository root. The first two accept `--workbook` for a different workbook location. Cleaning and EDA accept `--project-dir` for another output root.

```bash
python -m scripts.analyze_exceptions
python -m scripts.prepare_data
python -m scripts.analyze_training
python -m scripts.check_net_retention
```

**Interpreting credits.** Each identified merchandise credit can claim a single sale for the same customer and SKU at or before its timestamp. Choose the latest sale with sufficient reversible units. Prices may differ. Within a minute, prefer an eligible Excel row before the credit, then the smallest eligible row number. Classify elapsed times up to and including 72 hours in the UK, 120 hours in the specified 23 European regions, or 240 hours in other known regions as cancellations. Longer intervals are operational returns. Shipment, delivery and physical receipt are unobserved. Keep anonymous, unmatched and unknown-region cases identifiable. [Reconciliation](results/reversal_benchmark.csv) · [Line-level credit matches](results/credit_allocations.csv.gz) · [Matching differences](results/reversal_match_differences.csv)

The full-workbook run classified 8,665 merchandise credit lines as 1,473 cancellations, 5,883 operational returns, 1,147 unmatched with an identified customer, and 162 anonymous. Both large purchases on invoices `541431` and `581483` have zero effective units after their linked credits. Twelve credits match purchases in the same recorded minute.

**Interpreting amounts and dates.** The effective purchase amount values the remaining units at the original sale price. A Net Valid Purchase means the customer has at least one unit of that SKU remaining in a qualifying target-month basket by the exclusive month-end cutoff. If one line is fully credited, another line or basket can still establish that positive. Signed merchandise revenue sums each sale and credit at its recorded price. Retention is inferred from the recorded sales and credits. Physical possession is unobserved. A later credit can update a later snapshot only. The manifest records the source, summaries and artifact checksums. [Data manifest](results/data_manifest.json)

**Testing threshold sensitivity.** Replaying the UK boundary at 120 hours instead of 72 reclassifies 608 credits. The audit compares sale allocations, all basket-event columns and month-end positive pairs before the candidate filter in eight training months and October validation. The four primary models use the same threshold-independent event fields. Cancellation/return status, lag-based return counts and rates are audit fields. If LightGBM later uses them in a validation ablation, its features and scores can depend on the thresholds. [Sensitivity results](results/net_retention_threshold_check.csv) · [Reproduction command](scripts/check_net_retention.py)

### D2 · Monthly queries and shared evaluation

After D1, open the monthly query notebook on a CPU and run its cells in order. It traces one customer's history, catalog, target-month positives and shared evaluation inputs. To rebuild the same tables from the D1 signed customer ledger and credit allocations, run the following from the repository root. The build uses February–September training targets and October validation. The builder writes the fixed October AUC sample. November labels, the November AUC sample and test scores are reserved for the frozen test run. [Monthly queries and evaluation notebook](notebooks/monthly_queries_and_evaluation.ipynb) · [Data protocol](docs/data_protocol.md) · [Query builder](scripts/build_queries.py)

```bash
python scripts/build_queries.py --project-root .
python -m scripts.audit_label_delay
```

The query builder writes to `data/processed/d2/`. Look in `manifest.json` for versions and SHA-256 digests, `coverage.csv` for the population denominators, and the separate Parquet files for queries, histories, labels, catalogs, pairs and historical features. Use `--output-dir` to write a second build for comparison. The saved build contains 5,430 training queries. Of the 1,357 customers active in October, 758 qualify for primary validation over the shared catalog of 3,495 items. The notebook’s worked scores illustrate the evaluator. [D2 manifest](data/processed/d2/manifest.json) · [Monthly coverage](data/processed/d2/coverage.csv)

**Why October and November differ.** Both use histories strictly before 1 October 2011. October is the next month (T+1), November is the second (T+2). Recency is elapsed days from the last retained purchase to `history_end`, including fractional days. The 7-, 30- and 90-day windows end at that same cutoff. A customer present in both periods therefore has the same history, catalog and historical features. A future difference in scores could reflect the longer horizon, the Christmas run-up or changes in the buyers and products observed. This single pair of months cannot isolate their effects. The workbook ends on 9 December at 12:50, leaving a shorter window for later credits against November purchases. December is outside the main evaluation.

**How to read delayed credits.** Before customer eligibility and catalog filtering, 758 of 246,998 distinct month–customer–SKU positives in February–October reach zero when later observed credits are included. The project owner supplied a separate February–November workbook count of 788/301,922 (0.26%), including 30 for November. These fractions describe changes in recorded labels. Computing a change in Recall@10 or NDCG@10 would require eligible per-query labels and model rankings. Model selection uses October validation alone. [Pre-test audit table](results/d2_audit/label_delay_pretest.csv) · [Label-delay audit](scripts/audit_label_delay.py) · [Definitions and denominators](docs/data_protocol.md)

### D3–D5 · Model pilots

In Colab’s GitHub tab, open `notebooks/model_pilots.ipynb` from the revision containing the GPU setup described below. Choose `smoke` and one model per session. MF and LightGBM use CPU. Choose an NVIDIA GPU for either sequence model’s Colab pilot. Run all cells. The notebook verifies D2, caches LRURec in Drive under `cache/lrurec/`, and prints the Drive run directory `experiments/model_pilots/<model>/<session>/` containing `session.log` and `model/run.json`. [Model pilot notebook](notebooks/model_pilots.ipynb) · [Open the verified notebook in Colab](https://colab.research.google.com/github/NobunagaK/online-retail-top10-recommendation/blob/8a66b9f1dee361969a2afc0c085ed9e5a166cb39/notebooks/model_pilots.ipynb)

**CPU runs.** Install the model dependencies below, then use the following commands to run each CPU pilot. For LRURec, the source command obtains and verifies the authors’ pinned file. Use `LRUREC_CACHE_DIR` for a persistent cache or `LRUREC_REFERENCE_FILE` for an existing exact copy. [MF and LightGBM pilot results](results/model_pilots/summary.csv) · [Sequence model runs](results/model_pilots_upstream/summary.csv)

```bash
python -m scripts.train --model mf --device cpu --smoke
python -m scripts.train --model lightgbm --device cpu --smoke
python -m src.models.lrurec_source
python -m scripts.train --model lrurec --device cpu --smoke
```

**GPU verification.** Select Python 3.13 and a Tesla T4. Cell 5 installs PyTorch 2.7.0 from the CUDA 12.6 wheel index (`2.7.0+cu126`) in the isolated Mamba4Rec environment, reads its C++ ABI flag and installs matching native wheels with `--no-deps`. LRURec uses Colab’s GPU PyTorch installation. Mamba4Rec uses the repository’s MIT-licensed copy of the authors’ recommender and real RecBole. Its preflight checks the native GPU forward pass, gradients, optimizer update and checkpoint reload, before training on D2. [LRURec adaptation](docs/lrurec_adaptation.md) · [Mamba4Rec setup](docs/mamba4rec_adaptation.md)

```bash
python -m scripts.train --model lrurec --device cuda --smoke
python -m src.models.mamba4rec_adapter --preflight --output runs/mamba4rec/preflight.json
python -m scripts.train --model mamba4rec --device cuda --smoke
```

The CLI reads `data/processed/d2/` and writes each run to `runs/<model>/<UTC timestamp>/`. Use `--data-dir` or `--output-dir` for other locations. To resume, specify the same output directory and add `--resume`. Completed runs include `metrics.csv`, `query_metrics.csv`, `top10.parquet`, `auc_scores.parquet` and the best model (`best.pt` or LightGBM's `best.txt`). Omitting `--smoke`, or choosing `baseline` in the notebook, fits one configuration to the full training data. D6 separately compares registered configurations and seeds before opening November test labels. [Training CLI](scripts/train.py) · [Model configurations](configs/)

### D6 · Tuning, freezing and locked testing

1. Open the D6 notebook, select `plan` and run all cells on CPU to inspect the 32 registered configurations. D2 data comes from the pinned Git checkout. [D6 notebook](notebooks/d6_tuning.ipynb) · [Registered grid](configs/d6_search.yaml)
2. Choose `all` for each model. Use CPU for MF and LightGBM and an NVIDIA GPU for either sequence model in Colab. By default, the runner checks existing full baselines under `experiments/model_pilots/<model>/` on Drive. It verifies the configuration, data, per-query metrics and artifacts before reuse. For a matched comparison on a new GPU, clear `REUSE_BASELINE` before the model’s first search to run `01` and `08` in the same environment. To resume, select the same model and `all`. The `screen` and `confirm` options run the two stages separately.
3. After all four models finish, select `freeze` once. Mean October validation NDCG@10 over three seeds determines the selected configurations and checkpoints. Then select `test` once per model. November scoring loads frozen weights and uses the October 1 history cutoff. [Selection and paired comparisons](docs/CS5481_Project2_Research_Development_Plan_3.4.en.md) · [Course specification, p. 3](resource/Group_Project_Tutorial_Recommender_Systems.pdf#page=3)

Each D6 model’s `run.json` and the consolidated `tuning.csv` contain separate `optimizer` and `lr_scheduler` fields. The records include the effective values when a baseline YAML omits these keys.

The repository’s `results/` holds the formal experiment. To reproduce it independently, use the same new output directory throughout. The example shows LRURec. Train each of the four models on its designated device before freezing once, then score each model. For a fresh Colab run, set `DRIVE_FOLDER` to a new project folder. D7 reads the frozen predictions from the original Drive project.

```bash
python -m scripts.tune --stage plan --results-dir runs/d6_reproduction
python -m scripts.tune --stage all --model lrurec --device cuda --pilots-dir experiments/model_pilots --results-dir runs/d6_reproduction
python -m scripts.freeze_experiment --results-dir runs/d6_reproduction
python -m scripts.evaluate --model lrurec --device cuda --results-dir runs/d6_reproduction
```

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

From the repository root, install the CPU runtime or the development environment, which also includes the CPU runtime. The direct dependency versions are pinned for Python 3.12. The local reference interpreter is Python 3.12.14. [CPU requirements](requirements.txt) · [Tests and notebook requirements](requirements-dev.txt)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For local notebook execution, Python export or tests, run `python -m pip install -r requirements-dev.txt` in the same environment. In a local notebook frontend, select that environment’s Python kernel. To run the test suite, use `python -m pytest -q`. The D1/D2 commands use NumPy, pandas, openpyxl, Matplotlib and PyArrow on CPU.

For the Linux CPU model pilots, install the additional model packages and the CPU PyTorch build in the active environment. The GPU notebook sets up each sequence model separately, including the CUDA-enabled PyTorch build and native wheels documented for Mamba4Rec. [Model requirements](requirements-models.txt) · [Mamba4Rec GPU setup](docs/mamba4rec_adaptation.md)

```bash
python -m pip install -r requirements-models.txt
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

## Random seeds

The workbook check, invoice example, cleaning audit, and training-period EDA use deterministic calculations. D2 uses seed `42` for shared training negatives and `5481` for the fixed validation AUC sample. D6 used training seed `42` for hyperparameter search, seeds `42`, `2026`, and `3407` to check selected configurations, and a separate seed `5481` to sample negatives for ROC AUC.

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
| Verified | [Store ledger A](data/processed/signed_ledger_a.csv.gz) · [Customer ledger B](data/processed/signed_ledger_b.csv.gz) · [October 1 basket snapshot](data/processed/training_basket_events.csv.gz) | Signed merchandise transactions and the effective October 1 snapshot. |
| Verified | [Reconciliation](results/reversal_benchmark.csv) · [Line-level credit matches](results/credit_allocations.csv.gz) · [Matching differences](results/reversal_match_differences.csv) | Credit allocations, classifications and controls traceable to individual Excel lines. |
| Verified locally | [Training EDA results](results/training_eda/) | Training-window totals and denominators, monthly volume, distributions and product popularity in JSON, CSV and PNG formats. |
| Available on execution | Output cells in [the transaction notebook](docs/technical/transaction_records.ipynb) | Workbook dimensions, invoice lines, and basket totals. |
| Verified locally | [D2 manifest](data/processed/d2/manifest.json) · [Monthly coverage](data/processed/d2/coverage.csv) · [Parquet tables](data/processed/d2/) | Cutoff-specific inputs, cohort counts, and deterministic training and validation samples. |
| CPU pilot evidence | [MF and LightGBM](results/model_pilots/summary.csv) | Includes the earlier independent LRURec implementation at `69fcba3` for comparison. |
| Authors’ sequence models | [Run status and metrics](results/model_pilots_upstream/summary.csv) | See the pilot metrics by device and the Mamba4Rec setup notes for the GPU session records. |
| Archived D6 tuning | [Registration](results/d6_registration.json) · [48 fits](results/tuning.csv) · [October validation](results/validation_metrics.csv) | 32 screening fits and 16 additional seed runs. MF and LightGBM used CPU. Both sequence models used an A100 for their formal baselines and tuning. |
| Archived D6 selection and test | [Frozen selection](results/selection.json) · [Selection SHA-256](results/selection_digest.json) · [November test](results/test_metrics.csv) | Checksums for the twelve selected checkpoints and three-seed test results. Drive keeps weights in `results/d6_runs/` and per-query test evidence in `results/d6_test/` for D7 metric reconstruction and case studies. |
| Planned | `results/cohort_coverage.csv`, `results/segment_metrics.csv`, `results/case_study.csv` | Population coverage, metrics by group, and individual recommendations. |
| Planned | `experiments/exp_XXX/diagnostics_panel.png` | Diagnostic figure for an experiment. |

## Technical documentation and Canvas submission

The technical report draft covers D1 and D2 data evidence and D6 validation and locked test results. D7 will add segment findings and recommendation examples. The data protocol defines how to reproduce each query and metric. The research plan describes the model experiments and their acceptance criteria. [Technical report draft](docs/technical_report.md) · [Data protocol](docs/data_protocol.md) · [Research plan](docs/CS5481_Project2_Research_Development_Plan_3.4.en.md)

Project-authored documents have English and Chinese editions. Each Markdown cell in a technical notebook presents the full English section followed by the full Chinese section. Both versions share the same code. [Technical reference](docs/technical/transaction_records.ipynb)

The Canvas package will include the technical report, source code, README, and any permitted demonstration materials used. For each new deliverable, check the relevant requirements on page 3, link each answer to a reproducible result, and identify the team’s additional analyses and still-unmeasured quantities. [Project 2 rubric](resource/Group_Project_Tutorial_Recommender_Systems.pdf#page=3)
