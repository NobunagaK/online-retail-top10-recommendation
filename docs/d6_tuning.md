# D6 tuning decisions and baseline comparisons

[English](d6_tuning.md) · [简体中文](d6_tuning.zh-CN.md)

D6 measured how much each model gained over its original configuration and selected one configuration per model for the November test. The course calls for model-specific tuning on a common chronological split. The team chose eight settings per model so each search could include the original baseline, controlled comparisons and combinations that might work well together. The archived registration records the exact values before the formal search. The explanations below describe what each comparison tests. [Course specification, p. 3](../resource/Group_Project_Tutorial_Recommender_Systems.pdf#page=3) · [Research plan, section 5](CS5481_Project2_Research_Development_Plan_3.4.en.md) · [D6 registration](../results/d6_registration.json)

## What improved over the baseline?

Both sides of each arrow use seed 42, 758 October customers and the same 3,495 candidates. Both MF fits and both LightGBM fits ran on CPU. Both sides of each sequence-model comparison ran on an A100. The left side is the original configuration `01`, and the right side is the selected configuration evaluated at the same seed.

| Model | Selected | NDCG@10, baseline → selected | Δ NDCG@10 | Recall@10, baseline → selected | Sampled AUC, baseline → selected |
|---|---|---:|---:|---:|---:|
| MF | `03` | 0.20795 → 0.23660 | +0.02865 | 0.10496 → 0.11568 | 0.83818 → 0.84382 |
| LightGBM | `06` | 0.35220 → 0.35367 | +0.00147 | 0.17304 → 0.17261 | 0.90333 → 0.90326 |
| LRURec | `07` | 0.15683 → 0.24387 | +0.08704 | 0.06907 → 0.11858 | 0.82282 → 0.83405 |
| Mamba4Rec | `03` | 0.13903 → 0.23321 | +0.09418 | 0.05796 → 0.10921 | 0.82082 → 0.83522 |

LightGBM illustrates why the selection metric matters. At seed 42 its selected configuration gained 0.00147 on NDCG@10, while Recall@10 fell 0.00044 and sampled AUC fell 0.00007. NDCG rewards putting a relevant item nearer the top. All three metrics rose for MF, LRURec and Mamba4Rec in these comparisons. [Validation metrics at full precision](../results/validation_metrics.csv)

The first eight runs per model used seed 42. The two leading configurations then added seeds 2026 and 3407, giving 32 initial fits and 16 additional fits. Mean October NDCG@10 across the three seeds determined each winner. All four baselines appeared only at seed 42. November evaluated the four frozen winners, so these data quantify a baseline-to-tuned improvement on October validation only. [Run ledger](../results/tuning.csv) · [Frozen finalists](../results/selection.json) · [November test](../results/test_metrics.csv)

## Matrix factorization

MF learns one vector and one bias per customer and per item. The dimension `d` controls how many latent preferences a vector can represent. The learning rate controls Adam’s step size. The loss averages binary cross-entropy over the purchase positives and sampled negatives within each customer–month. An L2 term penalizes both the customer and item vectors and their biases. This penalty matters in a sparse matrix where many items have few observations. All eight settings used the same sampled pairs, a batch target of 1,024 pairs that keeps queries together, up to 100 epochs, and early stopping after 10 epochs without an improvement in October NDCG@10. [MF implementation](../src/models/mf.py) · [Baseline](../configs/mf.yaml)

| ID | Dimension | Learning rate | L2 | What the comparison tests | October NDCG@10, seed 42 |
|---|---:|---:|---:|---|---:|
| `01` | 32 | 0.001 | 1e-5 | Reference for the lecture-model implementation. | 0.207951 |
| `02` | 16 | 0.001 | 1e-5 | Halve the dimension to test a more compact representation. | 0.176720 |
| `03` | 64 | 0.001 | 1e-5 | Double the dimension to test whether more latent factors help distinguish items. | 0.236599 |
| `04` | 32 | 0.003 | 1e-5 | Triple the step size to test faster updates under the same stopping rule. | 0.204345 |
| `05` | 32 | 0.001 | 1e-6 | Reduce the L2 penalty tenfold to allow larger learned factors and biases. | 0.207866 |
| `06` | 32 | 0.001 | 1e-4 | Increase the L2 penalty tenfold to test stronger shrinkage. | 0.208707 |
| `07` | 16 | 0.003 | 1e-4 | Test faster updates jointly with fewer factors and stronger shrinkage. | 0.172575 |
| `08` | 64 | 0.003 | 1e-6 | Test the same faster updates with more factors and weaker shrinkage. | 0.231155 |

At seed 42, changing only the dimension from 32 to 64 gave the largest single-setting gain over `01` (+0.02865 NDCG@10). Settings `07` and `08` each change three parameters from baseline, so each score describes the combination. Both finalists used 64 dimensions. `03` scored 0.235145 ± 0.001273 across seeds, ahead of `08` at 0.231334 ± 0.004155. [Run ledger](../results/tuning.csv) · [Frozen selection](../results/selection.json)

## LightGBM

All settings used LambdaRank to rank candidate products from the same 21 features. Each fit could add up to 1,000 trees, with 50 rounds of patience on full-catalog October NDCG@10. The reference `01` used up to 31 leaves per tree, learning rate 0.05, a minimum leaf sample threshold of 20 and L2 regularization of 0. Each comparison below is relative to `01`. [Feature definitions, exact settings and scores](lightgbm_features.md#how-d6-selected-configuration-06)

| ID | What the change tests | October NDCG@10, seed 42 |
|---|---|---:|
| `01` | Establish the reference against which to measure each change. | 0.352203 |
| `02` | Cap each tree at 15 leaves. Test whether fewer splits suffice to represent the feature interactions. | 0.351645 |
| `03` | Allow 63 leaves. Test whether finer partitions improve the ranking. | 0.351067 |
| `04` | Lower the learning rate to 0.03. Each new tree contributes a smaller correction to the ensemble. | 0.352807 |
| `05` | Raise the minimum leaf sample threshold to 50. Test whether requiring more evidence for a split improves ranking. | 0.356386 |
| `06` | Set L2 regularization to 1. Test whether shrinking leaf outputs helps while keeping the baseline tree structure settings. | 0.353668 |
| `07` | Use 15 leaves, a minimum leaf sample threshold of 100 and L2 of 10 with learning rate 0.10. Test fewer partitions and stronger regularization together with larger updates. | 0.350883 |
| `08` | Use 63 leaves, a minimum leaf sample threshold of 50 and L2 of 1 with learning rate 0.03. Test greater capacity together with smaller updates and regularization. | 0.350040 |

Only `02–06` isolate one parameter. The scores for `07` and `08` measure joint changes. `05` led the seed-42 screen at 0.356386. The registered three-seed mean selected `06` (0.354744 ± 0.001069) over `05` (0.352988 ± 0.003152). [Ranker implementation](../src/models/lightgbm_ranker.py) · [Frozen selection](../results/selection.json)

## LRURec and Mamba4Rec

A cap of 64 events truncates the histories of 497 of the 758 October queries. Doubling the cap to 128 reduces this count to 276, and doubling it again to 256 reduces it to 117. These three settings test whether making more past purchases visible improves ranking. Each formal baseline (`01`) used Adam, a fixed learning rate of 0.001, weight decay 0.01, hidden dimension 64, two layers, dropout 0.2 and early-stopping patience 10. Mamba4Rec started with state dimension 16. Keeping batch size 64 and the maximum of 100 epochs fixed gave all eight configurations the same batching and training budget. [Baseline LRURec](../configs/lrurec.yaml) · [Baseline Mamba4Rec](../configs/mamba4rec.yaml) · [October histories](technical_report.md)

Hidden dimension controls the width of each event representation. For Mamba4Rec, `d_state` controls the recurrent state size per inner channel, and the layer count determines how many blocks process the sequence. Dropout randomly discards a fraction of activations during training. The grid tests lighter (0.1) and stronger (0.3) dropout around the 0.2 baseline.

Configurations `02–07` used AdamW and allowed 15 epochs without an improvement in October NDCG@10 before stopping. The longer patience gave slower configurations more chances to improve. AdamW applies weight decay separately from the gradient update. The implementation exempts LayerNorm parameters, biases, LRURec’s `params_log` and Mamba’s designated state parameters to avoid directly shrinking them. A cosine schedule lowers the learning rate over the epoch budget to test whether smaller late updates improve ranking. Mamba4Rec’s convolution width 4 and expansion factor 2 were held fixed while testing other aspects of its recurrent state. [Shared training implementation](../src/models/sequence_objective.py) · [Registered grid](../configs/d6_search.yaml)

| ID | Configuration and question tested | LRURec October NDCG@10, seed 42 | Mamba4Rec October NDCG@10, seed 42 |
|---|---|---:|---:|
| `01` | Reproduce the original 64-event baseline to measure the total gain from tuning. | 0.156831 | 0.139031 |
| `02` | Keep 64 events and a fixed rate. Introduce AdamW and patience 15 as the reference for the length comparison. | 0.233929 | 0.230215 |
| `03` | Change only the cap from 64 to 128 relative to `02`. Test whether additional past purchases improve ranking. | 0.232591 | 0.233214 |
| `04` | Change only the cap from 64 to 256 relative to `02`. Test the longest registered history. | 0.241425 | 0.231797 |
| `05` | Add cosine scheduling to `03`. This isolates the schedule at length 128 under AdamW. | 0.232454 | 0.232934 |
| `06` | Starting from `05`, lower dropout to 0.1 and weight decay to 0.0001. LRURec widens to 128 hidden units. Mamba4Rec keeps 64 hidden units, uses one layer with state dimension 32 and raises the rate to 0.003. Test a wider LRU encoder and a shallower Mamba encoder with more state per layer. | 0.226330 | 0.225484 |
| `07` | Use 256 events, 128 hidden units, dropout 0.3, zero weight decay and a smaller initial rate of 0.0003 with cosine. LRURec keeps two layers and Mamba4Rec uses one layer with state dimension 32. Test longer history and greater width with slower updates and dropout as the regularizer. | 0.243874 | 0.179688 |
| `08` | Match `01` on architecture, length, initial rate and patience 10. Change only the optimizer/decay policy and the schedule to AdamW with cosine to measure their combined effect. | 0.233742 | 0.229694 |

At seed 42, moving from 64 to 256 events (`02→04`) gained **0.00750** for LRURec and **0.00158** for Mamba4Rec. Adding cosine at length 128 (`03→05`) changed their scores by **−0.00014** and **−0.00028**. The larger gains for `01→08` (**+0.07691** and **+0.09066**) apply to the joint optimizer and schedule change. `01→02` also changed patience. The grid gives no isolated estimate for patience, decay exemptions or the individual changes bundled in `06/07`. [48-run ledger](../results/tuning.csv)

With three seeds, LRURec `07` averaged 0.240593 against 0.234647 for `04`. Mamba4Rec `03` averaged 0.231480 against 0.231333 for `05`. The predeclared validation criterion therefore selected a longer, wider LRURec with cosine and a length-128 Mamba4Rec at a fixed rate. [Frozen selections](../results/selection.json)
