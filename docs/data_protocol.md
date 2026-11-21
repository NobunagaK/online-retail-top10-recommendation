# Monthly purchase queries and evaluation

[English](data_protocol.md) · [简体中文](data_protocol.zh-CN.md)

A query asks which ten products to recommend to one customer for a particular month. The target is a purchase in that month with units still outstanding at the close of the month. MF, LightGBM, LRURec and Mamba4Rec share the same query, catalog and relevant items. Data are from D. Chen (2015), *Online Retail*, UCI, CC BY 4.0. [Course specification, p. 3](../resource/Group_Project_Tutorial_Recommender_Systems.pdf#page=3) · [Dataset DOI](https://doi.org/10.24432/C5BW33) · [Executed notebook](../notebooks/monthly_queries_and_evaluation.ipynb)

## What each date controls

| Target | History and candidate catalog | Labels |
|---|---|---|
| February–September 2011, eight training months | All eligible completed baskets before the first of the target month. December 2010 and January 2011 supply the initial history. | Close each target month separately. |
| October 2011, validation | Before 1 October 2011 at 00:00. | Purchases in October, with linked credits observed before 1 November. |
| November 2011, final test | The same 1 October snapshot. | Purchases in November, with linked credits observed before 1 December. Real test labels and test AUC pairs are reserved for the frozen experiment. |

Every interval includes its start and excludes its end. Each historical snapshot uses the D1 signed customer ledger, linked credit quantities and the original eligible invoice completion times. Within a target month, keep a customer–SKU pair once if any basket for that pair still has positive units after deducting **all** visible matched credits. Full, partial and mixed credits follow one quantity rule. A later purchase of the same SKU can establish a positive after an earlier purchase has been reversed. The regional cancellation/return classification is for audit. The December 2011 tail ends at 12:50 on 9 December. [D1 manifest](../results/data_manifest.json) · [Snapshot and label implementation](../src/data/purchase_snapshot.py)

## Recency and the forecast horizon

For a customer or a customer–SKU pair with prior activity, recency in days is the elapsed time from its last eligible basket to `history_end`. The 7-, 30- and 90-day windows also end at `history_end`. For customer `17811`, the last basket in the permitted snapshot was at 10:08 on 27 September. Both forecasts therefore use **3.578 days** of recency (85 hours and 52 minutes). A pair with no prior purchase has missing recency and zero past purchases. For identical customers and items, the October and November historical feature values agree exactly. Different customers can qualify for evaluation in each month, so their aggregate feature distributions can still differ.

October is the next complete month after the training cutoff (T+1). November is one month farther ahead (T+2). When interpreting the eventual scores, report cohort coverage, target mix and the two forecast horizons alongside them. Pre-Christmas purchasing is a plausible source of November demand differences. With one validation month and one test month, the separate effects of seasonality, horizon and customer mix cannot be identified from their scores alone.

## Cohorts, catalogs and inputs

- In a training month, a customer needs both a history before month start and a target positive in that month's historical catalog. For evaluation, the customer must have had at least one eligible training query and must buy at least one in-catalog item in the target month for the primary metrics. Products the customer has previously bought remain candidates.
- Preserve every target-month positive for coverage, including catalog-external SKUs. Report active customers, unseen customers, customers with history but no eligible training query, catalog-external-only buyers and inactive training customers. Supplementary Recall gives equal weight to active customers with an eligible training query, counts all their distinct net purchases in the denominator and assigns zero if all are catalog-external.
- MF and LightGBM share all positive items and up to five different training negatives per positive from the same query catalog, sampled without replacement after excluding all target positives. A negative represents an unobserved purchase during that month. Record the sampling seed. A LightGBM ranking group corresponds to one query.
- The two sequence models share original variable-length histories. Order baskets by their completion time, break time ties by invoice number and sort SKU codes inside each basket. Count one event per SKU in each basket and keep repetitions across baskets. Item IDs are stable as the permitted vocabulary grows. Zero is padding. Runtime truncation takes the most recent events (initial limit 64). Both models get the same real item IDs and target positives. Their planned training objective averages the negative log-probabilities of all positive items over the query's visible catalog.
- Catalog membership is determined by each cutoff. An earlier training snapshot can contain a SKU that disappears after subsequent credits. Its stable ID still identifies that history. Report eligible items that lack actual training support when fitting the model; keep them in the evaluation catalog.

## Built development data

The eight training months produce **5,430 customer–month queries**. October has **1,357 active customers**. Of these, **758** satisfy the primary evaluation rules, with **25,051** distinct in-catalog positives across those customers and a catalog of **3,495 SKUs**. The 43,978 positives used in the delayed-feedback audit include purchases outside this evaluation cohort. The stable training vocabulary contains 3,496 SKUs because an item present at an earlier cutoff can leave a later catalog after credits. [Monthly coverage](../data/processed/d2/coverage.csv) · [Files, schemas and input hashes](../data/processed/d2/manifest.json)

## Ranking and reproducibility

Score every item in the query catalog, keep ten distinct SKUs and break ties by exact StockCode. Recall@10 divides hits by the number of distinct in-catalog positives. NDCG@10 discounts a hit at rank `r` by `1/log2(r+1)` and divides by the best possible discounted sum for that query. Average both metrics equally across primary queries. Use unsampled full-catalog scores for ranking.

For the supplementary query-macro ROC AUC, keep all positives and draw up to ten negatives per positive from the same catalog without replacement. Seed 5481, the split and query identifier determine each draw. Reuse the same pairs across models and training seeds. Compute AUC from scores before truncating to ten, awarding half credit to a tie. Record single-class queries separately and report a missing AUC if no query has both classes. [Evaluator](../src/evaluation/ranking.py) · [Project experiment plan](CS5481_Project2_Research_Development_Plan_3.4.en.md)

## Month-end labels and delayed feedback

Closing each month gives earlier purchases more time to accumulate credits. An October purchase fully reversed on 1 November still contributes an October positive if its balance was positive at the October close. D2 uses this same observation rule for every month. The transaction file ends on 9 December, leaving a November purchase at most nine days of **post-month** follow-up. A comparison using every later recorded credit therefore gives different months different follow-up times.

A retrospective audit compares the original month-end positives to the same purchase month's balance at the end of the workbook. Its unit is one **month–customer–SKU** before any evaluation-cohort or catalog filtering. The February–September training months lose 653 of 203,020 such positives. October loses 105 of 43,978 (0.239%). Together, February–October lose 758 of 246,998 (0.307%). September's 149 losses accrue over the rest of the observed workbook, including credits in November and December. This descriptive audit is separate from the model inputs. [Month-by-month audit](../results/d2_audit/label_delay_pretest.csv) · [Audit implementation](../scripts/audit_label_delay.py)

The project owner's independent February–November workbook check reported **788 of 301,922 positives (0.261%)**, including 30 November positives. The ten-month figure includes the held-out month and is attributed to that check. The default D2 run constructs training and validation only. For the report, state that few monthly labels changed under the longer, unequal observation periods. Measuring a change in Recall@10 or NDCG@10 requires ranked predictions and the corresponding evaluation cohorts. The label-change fraction alone cannot establish its magnitude.

## Reproduce the evaluation cohorts and targets

Run this Python block from the repository root in the project environment after model selection. It reads the D2 October tables and rebuilds November labels with the same D1 ledger, credit allocations and cutoff code used in the locked evaluation. The archived primary-query and positive totals provide independent checks. Each target is one distinct in-catalog SKU per eligible customer and month. “Previously seen” uses the customer's entire net purchase history before October 1, including for November. A length cap counts the last events **before** deduplicating SKUs. [Locked label construction](../scripts/evaluate.py) · [October metrics](../results/validation_metrics.csv) · [November metrics](../results/test_metrics.csv)

```python
from pathlib import Path

import pandas as pd

from scripts.evaluate import november_tables
from src.data.monthly_queries import HISTORY_END
from src.models.pilot_data import PilotData

root = Path.cwd()
data = PilotData(root / "data/processed/d2")
histories = data.read("histories")
history = histories.loc[histories.history_end.eq(HISTORY_END)].set_index("customer_id").history_item_ids
october = {name: data.read(name).query("split == 'validation'") for name in ("cohorts", "labels")}
november = november_tables(root, data)


def customer_counts(items, past):
    return {"targets": len(items), "repeats": len(items.intersection(past)),
            "last64": len(items.intersection(past[-64:])), "last256": len(items.intersection(past[-256:]))}


customers = {}
for split, tables in [("validation", october), ("test", november)]:
    cohorts = tables["cohorts"]
    labels = tables["labels"].query("eligible and in_catalog")
    targets = labels.groupby("customer_id").item_id.agg(set)
    customers[split] = set(targets.index)
    counts = pd.DataFrame([customer_counts(items, history.at[c]) for c, items in targets.items()])
    size, repeat = counts.targets, counts.repeats
    last64, last256 = counts.last64, counts.last256
    scores = pd.read_csv(root / "results" / f"{split}_metrics.csv").set_index("model")
    assert scores.primary_queries.eq(len(targets)).all() and scores.auc_positive_items.eq(size.sum()).all()
    print(split, "primary / active", len(targets), int(cohorts.active.sum()))
    print("over ten / recall ceiling", int((size > 10).sum()), (size.clip(upper=10) / size).mean())
    print("repeat-only recall ceiling", (repeat.clip(upper=10) / size).mean())
    print("new to frozen history / all targets", int(size.sum() - repeat.sum()), int(size.sum()))
    print("buyers of both", int((repeat.gt(0) & repeat.lt(size)).sum()))
    print("repeat targets: full, last 64, last 256", int(repeat.sum()), int(last64.sum()), int(last256.sum()))
    print("customers gaining repeat evidence", int((last256 > last64).sum()))
    lightgbm = scores.loc["lightgbm"]
    print("recall: primary, adding zero-only customers, supplementary",
          lightgbm.recall_at_10_mean,
          lightgbm.recall_at_10_mean * lightgbm.primary_queries / lightgbm.supplementary_queries,
          lightgbm.supplementary_recall_at_10_mean)
print("shared primary customers", len(customers["validation"] & customers["test"]))
```
