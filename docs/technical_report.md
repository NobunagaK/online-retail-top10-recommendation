# Technical report draft · Top-10 product recommendation on UCI Online Retail

[English](technical_report.md) · [简体中文](technical_report.zh-CN.md)

This project asks which ten products a customer will buy in a later calendar month and still have a positive recorded quantity for at month-end. We have completed transaction preparation (D1), monthly query construction and evaluator checks (D2). MF, LightGBM and the authors’ LRURec have completed CPU pilots. Both sequence adapters use pinned public recommender code and completed Colab T4 smoke validation. October validation selected LightGBM. Its frozen November predictions reached NDCG@10 of 0.36294 and Recall@10 of 0.18715, averaged over three training seeds. We use D. Chen (2015), *Online Retail*, UCI Machine Learning Repository, DOI 10.24432/C5BW33, licensed CC BY 4.0. The transformations below derive from the supplied 541,909-line, eight-column workbook. [Dataset and attribution](https://doi.org/10.24432/C5BW33) · [Course specification, p. 3](../resource/Group_Project_Tutorial_Recommender_Systems.pdf#page=3)

## From transaction lines to purchase history

| Cleaning question | Evidence and decision |
|---|---|
| How are cancellations handled? | The workbook has 9,288 C-prefixed lines. For a merchandise credit with a known customer, we look for the most recent sale line for that customer and SKU with enough remaining units. Each match reserves those units once. Customer 12346 bought 74,215 units on invoice 541431. Credit C541433 reversed all 74,215 sixteen minutes later. A subsequent snapshot gives that purchase zero units. |
| How are returns identified? | The workbook records credit notes without dispatch, receipt or return-reason fields. We use elapsed-time proxies of 72 hours for the UK, 120 for the specified European destinations and 240 for other known destinations. Of 8,665 eligible merchandise credits, 1,473 fall within the cancellation threshold, 5,883 fall beyond it, 1,147 have a known customer and no eligible sale, and 162 have no customer ID. All linked credits reduce the original line's units when observed. Matching alone cannot confirm a physical return. |
| Which duplicates are removed? | We remove 5,268 exact repetitions across all eight original fields, keeping the first worksheet row. Remaining lines with the same customer, invoice and SKU are combined, summing units and the individual line amounts. A purchase on another invoice remains a separate event. |
| What happens to missing customer IDs? | 135,080 source lines lack a customer ID. The store ledger retains eligible anonymous lines for accounting. Customer histories and credit matching require an identified customer. |
| How are anomalous quantities and prices handled? | The 1,336 non-C negative-quantity lines all have zero price and missing customer IDs. They create no customer purchase. We also exclude bad-debt adjustments, zero prices and training-evidenced non-merchandise codes such as postage. Positive wholesale quantities remain eligible. For example, the 74,215-unit line above is removed from the later history because of its linked credit. |

Each basket closes at the latest original eligible purchase-line timestamp on its customer invoice. A snapshot uses only baskets completed and credits observed strictly before its cutoff. At the 1 October cutoff we obtain **260,919 retained lines, 258,532 basket-item events and 13,001 baskets**. Quantities and amounts preserve partial credits and the original sale price. Signed revenue separately values sales and credits at each document's own price. [Cleaning counts and source hashes](../results/data_manifest.json) · [Credit reconciliation](../results/reversal_reconciliation.csv) · [Row-level evidence](../results/transaction_exceptions.xlsx)

## Forecast and evaluation design

The October 1 history contains purchases from 3,608 customers. Of these, 2,023 contributed the **5,430 eligible customer–month training queries** from February through September 2011. For each query, the historical catalog contains items retained before the month starts. A label is one distinct customer–SKU purchase in the target month with units still outstanding at the exclusive target end. In October, **758 of 1,357 active customers** had an eligible training query and at least one purchase in the fixed catalog of **3,495 SKUs**. These 758 customers contribute **25,051 in-catalog positives** to primary validation. The **43,978** distinct October positives across all active customers include catalog-external items and purchases outside this cohort. [Monthly coverage](../data/processed/d2/coverage.csv) · [Data protocol](data_protocol.md)

October validation (T+1) and November testing (T+2) use the same history ending at 00:00 on 1 October. For customer 17811, the last eligible basket was at 10:08 on 27 September. Recency is **85 hours 52 minutes, or 3.578 days**, for both forecasts. The 7-, 30- and 90-day purchase windows also end on 1 October. Calendar demand can move even while those features stay fixed. For example, SKU 22910, a Christmas paper-chain kit, appeared in 17 training baskets in August and 104 in September. One product illustrates a potential seasonal effect. The longer forecast horizon, changing customer mix and holiday demand could all contribute to October–November score differences. Two evaluation months cannot isolate their individual effects. [Customer features](../data/processed/d2/customer_features.parquet) · [Training basket events](../data/processed/training_basket_events.csv.gz) · [Product descriptions](../data/processed/training_purchase_lines.csv.gz)

MF and LightGBM share within-query training negatives. LightGBM groups candidates by customer and target month. LRURec and Mamba4Rec use the same time-ordered history, preserving repeat purchases. All four models ranked every item in the same query catalog. The primary metrics are query-macro Recall@10 and NDCG@10, with sampled query-macro ROC AUC as a supplement. October validation alone guided model and hyperparameter selection before November test evaluation. [Inputs and sampling seeds](../data/processed/d2/manifest.json) · [Evaluator](../src/evaluation/ranking.py) · [Experiment plan](CS5481_Project2_Research_Development_Plan_3.4.en.md)

## Model implementations and small-sample validation

MF scores customer–item pairs with a latent-factor dot product and biases. LightGBM ranks customer–month groups with LambdaRank. LRURec trains the authors’ `LRU` (commit `e02a559`) on left-padded retail histories. Its final state scores eligible items through the shared item embeddings and biases. Mamba4Rec instantiates the authors’ `Mamba4Rec` (commit `44a0c84`) with RecBole, using right-padded item IDs and lengths. D2 supplies the chronological split, full cutoff catalog, monthly multi-positive loss inputs and repeat-purchase eligibility to both models. [LRURec adaptation](lrurec_adaptation.md) · [Mamba4Rec adaptation](mamba4rec_adaptation.md)

The CPU pilot uses seed 42, **206 training queries and 16 October validation queries**. Each query ranks **3,495 candidates**. Recall and NDCG use all candidates. Sampled ROC AUC uses all **428 in-catalog positives** and **4,280 fixed negatives** across the 16 queries. Metrics are averaged equally across queries.

| Model | Recall@10 | NDCG@10 | Sampled ROC AUC |
|---|---:|---:|---:|
| MF, CPU | 0.01162 | 0.06116 | 0.67143 |
| LightGBM, CPU | 0.16824 | 0.32521 | 0.91526 |
| LRURec, authors’ `LRU`, CPU | 0.02911 | 0.09083 | 0.55049 |
| LRURec, authors’ `LRU`, Tesla T4 | 0.02911 | 0.09298 | 0.55936 |
| Mamba4Rec, Tesla T4 (owner report) | 0.02345 | 0.07513 | 0.50397 |
| LRURec, independent implementation at `69fcba3` (historical) | 0.02896 | 0.09704 | 0.53316 |

The authors’ LRURec completed one CPU epoch and reproduced its scores and Top-10 exactly after reload. Of 3,495 candidates, 114 used mean fallback. These products accounted for 46 validation positives and zero Top-10 hits. The historical row shows what the independent implementation scored on the same pilot cohort. Formal model selection used the full cohort of 758 eligible October customers reported below. The real RecBole dependency passed a separate CPU check. Both T4 pilots covered 16 October queries and reproduced Top-10 with zero score change after reload. The notebook records the LRURec run. The project owner reported the separate Mamba4Rec preflight and pilot from Drive. [T4 run sources](../results/model_pilots_upstream/colab_gpu_smoke.json) · [RecBole CPU check](../results/model_pilots_upstream/mamba4rec/recbole_cpu_import.json) · [Baseline and historical runs](../results/model_pilots/summary.csv) · [Upstream model pilot records](../results/model_pilots_upstream/summary.csv)

## D6 tuning, model selection and locked test

Each model screened eight configurations in October with seed 42. The top two configurations ran again with seeds 2026 and 3407, giving 48 completed fits. MF and LightGBM used AMD EPYC 7B12 CPUs. Both sequence models ran all formal configurations on NVIDIA A100-SXM4-40GB GPUs. The table compares October validation NDCG@10 for configuration `01` and the selected configuration at the same seed on the same hardware. Earlier T4 baseline runs had separately reported scores of 0.16168 for LRURec and 0.14422 for Mamba4Rec. The table uses the A100 baselines from the D6 ledger. [Preregistration](../results/d6_registration.json) · [48-run ledger](../results/tuning.csv)

| Model | Frozen configuration and key settings | Configuration 01 (seed 42) | Selected (seed 42) | Difference |
|---|---|---:|---:|---:|
| LightGBM | `06` up to 31 leaves per tree, L2 regularization 1 | 0.35220 | 0.35367 | +0.00147 |
| MF | `03` 64 latent dimensions | 0.20795 | 0.23660 | +0.02865 |
| LRURec | `07` length 256, hidden size 128, dropout 0.3, learning rate 0.0003, AdamW with cosine schedule, zero weight decay | 0.15683 | 0.24387 | +0.08704 |
| Mamba4Rec | `03` length 128, hidden size 64, two layers, state size 16, AdamW with fixed rate | 0.13903 | 0.23321 | +0.09418 |

The original baselines ran once each at seed 42. At this seed, LightGBM’s selected configuration raised NDCG@10 by 0.00147, while Recall@10 fell by 0.00044 and sampled AUC fell by 0.00007. The other three models improved on all three metrics. The November test evaluated only the frozen winners. [Baseline and selected scores for all metrics, with all 32 tuning decisions](d6_tuning.md)

Of 758 October queries, 497 had more than 64 historical events and 276 had more than 128. Seed-42 trials `02/03/04` varied only the history cap (64, 128, 256). LRURec attained October NDCG@10 of 0.23393, 0.23259 and 0.24142. Mamba4Rec attained 0.23022, 0.23321 and 0.23180. Extending the cap from 64 to 256 therefore gained 0.00750 for LRURec and 0.00158 for Mamba4Rec. A second comparison (`01` vs `08`) kept length 64 and patience 10 while changing Adam with a fixed rate to AdamW with a cosine schedule. The gains were 0.07691 and 0.09066, respectively. This comparison measures the joint effect of changing the optimizer and the schedule. All these contrasts use one validation seed. [Registered grid](../configs/d6_search.yaml) · [Monthly coverage](../data/processed/d2/coverage.csv)

**October selection and November evaluation.** Seed 42 placed LightGBM `05` first (0.356386). Across the three seeds, `06` had the higher mean October NDCG@10 (0.354744 vs 0.352988 for `05`) and won under the registered rule. The same validation criterion selected LightGBM overall. [Eight LightGBM configurations](lightgbm_features.md#how-d6-selected-configuration-06)

We froze three checkpoints per model before opening November labels and scored those same checkpoints in November. The table gives query-macro metrics as the mean ± sample standard deviation across the three seeds. The standard deviation describes variation across training seeds. [Frozen selection and checkpoint hashes](../results/selection.json) · [Selection digest](../results/selection_digest.json)

| Period | Model | NDCG@10 | Recall@10 | Sampled ROC AUC |
|---|---|---:|---:|---:|
| October validation | LightGBM | 0.35474 ± 0.00107 | 0.17319 ± 0.00107 | 0.90336 ± 0.00011 |
| October validation | MF | 0.23514 ± 0.00127 | 0.11559 ± 0.00087 | 0.84397 ± 0.00059 |
| October validation | LRURec | 0.24059 ± 0.00375 | 0.11717 ± 0.00153 | 0.83454 ± 0.00082 |
| October validation | Mamba4Rec | 0.23148 ± 0.00173 | 0.11274 ± 0.00307 | 0.83319 ± 0.00220 |
| November test | LightGBM | 0.36294 ± 0.00220 | 0.18715 ± 0.00053 | 0.90148 ± 0.00005 |
| November test | MF | 0.23002 ± 0.00540 | 0.11451 ± 0.00136 | 0.85346 ± 0.00023 |
| November test | LRURec | 0.22974 ± 0.00257 | 0.11473 ± 0.00235 | 0.83859 ± 0.00105 |
| November test | Mamba4Rec | 0.22133 ± 0.00182 | 0.11132 ± 0.00318 | 0.83779 ± 0.00214 |

Both months shared the frozen October 1 catalog of 3,495 items. The primary cohort contained 758 eligible customers in October and 935 in November. The fixed AUC sample contained 25,051 positives and 241,129 negatives in October, and 30,879 positives and 292,245 negatives in November. Each month had zero skipped AUC queries. Supplementary recall also counted otherwise eligible active customers who bought only out-of-catalog items, giving 759 and 936 customers, respectively. [October validation](../results/validation_metrics.csv) · [November test](../results/test_metrics.csv)

**LightGBM ranked first in both months.** Its November NDCG@10 exceeded MF by 0.13292, or 57.8% relative to MF. LightGBM also scored 0.00820 higher in November than in October. All four models used the same pre-October purchase history, including the transactions underlying LightGBM’s 7-, 30- and 90-day features. Customer mix, forecast horizon and seasonal demand remain possible explanations for the month-to-month difference. [Feature definitions](lightgbm_features.md) · [Locked test results](../results/test_metrics.csv)

**MF and LRURec ranked differently by metric.** LRURec led MF on October NDCG@10. In November, MF led by 0.00028 on NDCG@10 and 0.01487 on sampled AUC, while LRURec led by 0.00022 on Recall@10. Recall counts relevant items anywhere in the Top-10, NDCG gives earlier hits more credit, and sampled AUC measures how often a purchased item outscores a sampled negative. Establishing statistical equivalence would require more than the spread across three training seeds. D7 will compare scores for the same customers by history length and by repeat versus first-observed purchases. [Metric definitions](../src/evaluation/ranking.py) · [October validation](../results/validation_metrics.csv) · [November test](../results/test_metrics.csv)

**What each model sees.** LightGBM receives explicit customer–item purchase counts, elapsed time since purchase and recent item popularity. The sequence models receive ordered item IDs and lengths. Repeated IDs can convey popularity implicitly. The sequence inputs omit elapsed time, quantity and store-wide purchase counts over fixed windows. A training basket contained a median of 15 distinct SKUs (90th percentile 43). Within an invoice, D2 sorts items by StockCode for reproducibility. The workbook supplies no behavioral order for these items. These differences motivate the D7 comparisons. Aggregate scores alone cannot establish how much any specific feature or within-basket ordering contributed. [Feature dictionary](lightgbm_features.md) · [Sequence construction](../src/data/sequences.py) · [Basket-size distribution](../results/training_eda/distribution_summary.csv)

### Evaluation coverage and the ten-item recall limit

| Measure under the frozen protocol | October validation | November test |
|---|---:|---:|
| Primary customers / customers with a net valid purchase that month | 758 / 1,357 (55.86%) | 935 / 1,657 (56.43%) |
| Primary customers buying more than ten in-catalog items | 585 / 758 | 720 / 935 |
| Maximum primary Recall@10 with complete knowledge of the targets | 53.30% | 55.04% |
| Maximum primary Recall@10 when restricted to previously seen items | 33.37% | 35.24% |
| Targets absent from that customer’s pre-October history / all primary targets | 14,408 / 25,051 (57.51%) | 17,030 / 30,879 (55.15%) |

**Only 445 customers belonged to both primary cohorts.** Another 313 appeared only in October and 490 only in November. All 490 had eligible training records. To see whether LightGBM’s month-to-month gain also holds for the same customers, D7 can compare paired scores for those 445 and report the two month-specific groups separately. [Cohort and target calculation](data_protocol.md#reproduce-the-evaluation-cohorts-and-targets)

**For a customer who bought 20 distinct in-catalog items that month, ten correct recommendations yield 50% recall.** The upper bound in the table averages each customer’s best attainable recall with ten slots. Catalog coverage imposes a further limit. Counting out-of-catalog purchases in supplementary Recall@10 lowers LightGBM’s November score from 18.71% to 17.55%. Adding just the one customer with only out-of-catalog purchases and zero hits would give 18.69%. Most of the drop comes from counting out-of-catalog purchases by customers already in the primary cohort. [Locked test metrics](../results/test_metrics.csv) · [Recall definition](../src/evaluation/ranking.py)

**Repeat buying and product discovery often coexist for the same customer.** In November, 824 of 935 customers bought both previously seen and previously unseen items relative to their history frozen on October 1. An item absent at that cutoff could have been bought in October. Of the 10,643 October validation targets that repeated an earlier purchase, 5,947 (55.88%) were visible within the last 64 events and 9,106 (85.56%) within the last 256. The longer history exposed a previous purchase for 3,159 additional targets across 373 customers. Determining which of these targets gained a Top-10 hit requires the models’ frozen rankings. [History and target calculation](data_protocol.md#reproduce-the-evaluation-cohorts-and-targets)

### Model size and training time

| Model | Baseline `01` size | Selected model size | Selected fits, training seconds (range over three seeds) |
|---|---:|---:|---:|
| MF | 182,160 real scalars | `03` · 358,800 real scalars | 66.9–69.7 (CPU) |
| LRURec | 361,193 real scalars | `07` · 981,033 real scalars | 97.0–108.7 (A100) |
| Mamba4Rec | 355,904 real scalars | `03` · 355,904 real scalars | 48.7–56.9 (A100) |
| LightGBM | 328 trees (seed 42) | `06` · 226, 297, 348 trees (seeds 42, 2026, 3407) | 170.0–237.2 (CPU) |

We used the actual D2 vocabulary (3,496 SKU IDs plus padding) to count MF’s factors and biases and to instantiate the pinned LRURec. Each complex LRURec element counts as two real scalars. For Mamba4Rec, summing the pinned model’s layer dimensions gives 223,808 item-embedding scalars and 132,096 other scalars in both configurations. Increasing its history cap from 64 to 128 lets it read more events with the same number of parameters. LightGBM size is expressed in saved trees. Its 21 inputs and each tree’s leaf limit describe the model structure but do not give a comparable scalar parameter count. [D2 vocabulary](../data/processed/d2/item_vocab.parquet) · [MF factors](../src/models/mf.py) · [LRURec adaptation](lrurec_adaptation.md) · [Mamba4Rec layers](../third_party/mamba4rec/mamba4rec.py) · [Mamba block](https://github.com/state-spaces/mamba/blob/v2.2.5/mamba_ssm/modules/mamba_simple.py)

Times are the logged `train_seconds` for the selected fits. LightGBM’s fitting loop includes a validation check at each boosting round. The elapsed times depend on the device and the number of rounds or epochs run before early stopping. [Run ledger](../results/tuning.csv) · [Timing implementation](../scripts/train.py)

## Delayed feedback and label stability

Our audit counts a **month–customer–SKU** once if it has positive retained units at month-end. The numerator counts these positives whose remaining units for that purchase month reached zero by the final observed timestamp, 9 December 2011 at 12:50. The audit precedes customer and catalog filtering. An early-October sale has weeks to accumulate credits before the label freezes. An otherwise outstanding 31 October purchase stays positive for October even if a 1 November credit fully reverses it. Across the training months, 653 of 203,020 positives reached zero. In October the figure was **105 of 43,978 (0.239%)**. Together, the reproducible February–October audit finds **758 of 246,998 (0.307%)**. The project owner separately reported **788 of 301,922 (0.261%)** for February–November, including 30 November positives. November has only the 1–9 December tail for post-month follow-up, so later credits are right-censored. This descriptive check leaves the frozen labels intact. Estimating the impact on ranking metrics requires the affected eligible queries and their actual recommendation ranks. [Pre-test label audit](../results/d2_audit/label_delay_pretest.csv) · [Audit method and owner-reported November figure](data_protocol.md#month-end-labels-and-delayed-feedback)

## What the model comparison must explain

| Topic | What the data show | Evidence needed from trained models |
|---|---|---|
| Sparsity | Before 1 October, 187,452 of 12,609,960 possible customer–SKU pairs had a purchase (98.51% sparsity). | Compare Recall@10 and NDCG@10 for short and long histories, with customer counts. |
| Popularity bias | The top 350 training SKUs by distinct customer count account for 46.70% of training events. | Compare their share of recommendation slots with their share of actual purchases in the same cohort, and measure long-tail hits. |
| Repeat purchases | 71,073 of 258,532 training events (27.49%) repeat a SKU from a strictly earlier basket of the same customer. | Compare hits on repeat and first-observed purchases, keeping both eligible. |
| Cold start | In October, 356 active customers had no eligible history, and 1,787 positive customer–SKU pairs fell outside the catalog. | Report eligibility coverage alongside primary scores and supplementary Recall with out-of-catalog items in its denominator. |
| Cancellations and returns | Linked credits change the effective quantity and can eliminate a target entirely, as invoice 541431 illustrates. | For a fixed customer case, present the history, target, each model's Top-10 and at least one miss. Check the credit available at the relevant cutoff before interpreting a hit. |

D7 will use the per-customer predictions, fixed AUC sample and target labels on Drive to reconstruct the D6 scores and complete the subgroup, coverage and customer case analyses above. [Training evidence](../results/training_eda/summary.json) · [Coverage by month](../data/processed/d2/coverage.csv)
