# LightGBM feature dictionary

[简体中文](lightgbm_features.zh-CN.md)

Each row describes one customer and candidate SKU at the query’s exclusive history cutoff. The 21 numeric columns come from the three D2 history tables. D2 first applies credits visible by the cutoff, then groups surviving items into completed baskets. October validation and November test share the October 1 history. [D2 data protocol](data_protocol.md) · [Feature implementation](../src/features/asof.py)

| Prefix | Fields after the prefix | Meaning |
|---|---|---|
| `customer_` | `invoice_count`, `mean_basket_size`, `amount_gbp` | Number of completed baskets, mean distinct SKUs per basket and total net purchase value at original sale prices. |
| `item_` | `distinct_customers`, `invoice_count`, `weighted_unit_price_gbp` | Distinct customers and baskets with this SKU. The price is its total net purchase value divided by its total retained units, in GBP per unit. |
| `pair_` | `previously_purchased`, `invoice_count`, `quantity` | Whether this customer previously retained this SKU, how many baskets contained it, and how many units remain across those baskets. |
| Each of the three prefixes | `days_since_last_purchase` | Fractional days from the most recent applicable completed basket to the exclusive cutoff. |
| Each of the three prefixes | `invoices_7d`, `invoices_30d`, `invoices_90d` | Distinct applicable baskets completed within the preceding 7, 30 or 90 days. Include the lower boundary and exclude the cutoff. |

An unseen customer–SKU pair has a purchase flag, counts and quantity of zero. Its recency is missing, which LightGBM handles natively. A candidate seen for the first time in September receives its observed October 1 item statistics. Customer IDs, SKU IDs, labels and credit-class fields are excluded from the feature matrix.

Each training group contains one customer-month, all its candidate positives and its shared D2 sampled negatives. During early stopping, the model scores the complete candidate catalog for each selected October customer. Within each group, equal scores are ordered by exact StockCode. The project callback computes binary NDCG@10 for each group and averages equally across customers. The fit returns the best validation NDCG@10 and records the score for every boosting iteration. The saved native model contains the selected prefix of trees. [Ranker implementation](../src/models/lightgbm_ranker.py) · [Baseline configuration](../configs/lightgbm.yaml)

With LightGBM 4.6.0 on CPU, a synthetic check gave every item the same score in each of two 12-item catalogs. The library metric and the project metric both returned macro NDCG@10 of 0.6934264036172708. A separate fixture verifies that reloading the selected native model reproduces predictions in the supplied candidate order. [Regression tests](../tests/test_lightgbm_ranker.py)

## How D6 selected configuration `06`

All eight trials shared the training data, 21 features and `lambdarank` objective. Each fit had a limit of 1,000 trees and stopped after 50 rounds without improvement in October validation NDCG@10. Every score below uses the same 758 October queries and 3,495 candidates per query. Only the two screening finalists ran with the additional seeds 2026 and 3407. Three-seed results give the mean ± sample standard deviation. [Registered configurations](../configs/d6_search.yaml) · [Per-fit records](../results/tuning.csv)

| Configuration | `num_leaves` | `learning_rate` | `min_child_samples` | `reg_lambda` | Seed 42 NDCG@10 | Three-seed NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| `01` | 31 | 0.05 | 20 | 0 | 0.352203 | — |
| `02` | 15 | 0.05 | 20 | 0 | 0.351645 | — |
| `03` | 63 | 0.05 | 20 | 0 | 0.351067 | — |
| `04` | 31 | 0.03 | 20 | 0 | 0.352807 | — |
| `05` | 31 | 0.05 | 50 | 0 | 0.356386 | 0.352988 ± 0.003152 |
| `06` | 31 | 0.05 | 20 | 1 | 0.353668 | 0.354744 ± 0.001069 |
| `07` | 15 | 0.10 | 100 | 10 | 0.350883 | — |
| `08` | 63 | 0.03 | 50 | 1 | 0.350040 | — |

`num_leaves` limits the leaves per tree, `learning_rate` scales each new tree’s contribution, `min_child_samples` sets the minimum leaf sample threshold, and `reg_lambda` applies an L2 penalty to leaf outputs. Configurations `02–06` each change one setting from baseline `01`. Configurations `07` and `08` change several settings together. Both scored below the baseline at seed 42.

**Configuration `05` led at seed 42. Configuration `06` won on the registered three-seed mean.** Its mean was 0.001756 higher and its sample standard deviation across the three seeds was smaller. The frozen `06` models retained 226, 297 and 348 trees for seeds 42, 2026 and 3407 respectively. [Finalist scores and frozen tree counts](../results/selection.json)
