# LRURec for monthly retail purchases

[简体中文](lrurec_adaptation.zh-CN.md)

The training path instantiates Yue et al.’s `LRU` from commit `e02a559bd8c42342f5c53d69cefea2589becb22c`. The authors’ embedding and linear recurrent blocks encode the history. The starting configuration takes the latest 64 purchase events, with 64 hidden dimensions and two layers. [Paper](https://doi.org/10.1145/3616855.3635760) · [Authors’ implementation](https://github.com/yueqirex/LRURec/blob/e02a559bd8c42342f5c53d69cefea2589becb22c/model/lru.py) · [Project adapter](../src/models/lrurec_adapter.py) · [Configuration](../configs/lrurec.yaml)

## Obtain the authors’ code

The pinned repository has no explicit software redistribution license for `model/lru.py`. The project distributes its adapter and source-fetch command. That command obtains the file from the authors and checks SHA-256 `149e0ccfe03c7eba7a51bf405d179f8a3f395011c0f730169aec2b59c9c156b1`. Set `LRUREC_CACHE_DIR` to a persistent local directory before running the commands below. In Colab the notebook uses the project’s Drive cache. For an existing offline copy, `LRUREC_REFERENCE_FILE` can point directly to that exact source file. [Source loader](../src/models/lrurec_source.py) · [Upstream tree](https://github.com/yueqirex/LRURec/tree/e02a559bd8c42342f5c53d69cefea2589becb22c)

```bash
python -m src.models.lrurec_source
python -m src.models.lrurec_source --offline
python -m scripts.train --model lrurec --device cpu --smoke
```

## From a customer history to ranked products

1. D2 constructs one item sequence per customer and history cutoff from net-retained basket events. Repeat purchases across invoices remain in sequence. For both October validation and November testing, available history ends immediately before 1 October 2011. A history of three purchases can enter the model as `[0, 0, A, B, A]`. The adapter applies the authors’ embedding, masks, internal power-of-two padding and recurrent blocks, then selects the final valid state. [Data protocol](data_protocol.md)
2. During evaluation, each query is encoded once. For each requested candidate, the adapter takes the dot product of the final state with the corresponding row in `embedding.token.weight` and adds `model.bias`. This yields the authors’ final-position scores before the original evaluator masks previously seen items. It allows a customer to buy A again and supports scoring the catalog in chunks. [Authors’ scoring head](https://github.com/yueqirex/LRURec/blob/e02a559bd8c42342f5c53d69cefea2589becb22c/model/lru.py#L72-L87) · [Authors’ evaluator](https://github.com/yueqirex/LRURec/blob/e02a559bd8c42342f5c53d69cefea2589becb22c/trainer/lru.py#L39-L49)
3. The project trains on a whole target month. It averages the negative log-softmax score of the customer’s distinct net-retained positives within the eligible catalog, giving each customer–month equal weight. For inference, each unsupported embedding row uses the mean of trained input or loss rows, and each unsupported item bias uses the mean of rows supported by the training loss. Row zero is reserved for padding. The masks come from the current run. [Monthly loss and support masks](../src/models/sequence_objective.py)
4. With fixed weights and dropout disabled, compare scores for the same real candidates with the authors’ full forward pass, including a repeat purchase and histories of unequal lengths. Separately check that splitting a candidate list into chunks gives the same scores, that the optimizer updates the encoder and scoring head, and that reloading reproduces Top-10. The CPU comparison at seed 23 gave a maximum absolute difference of 1.49 × 10⁻⁸ across 33 scores (three histories and eleven candidates). [Tests](../tests/test_lrurec.py) · [Upstream model pilot records](../results/model_pilots_upstream/summary.csv)

The upstream model’s real CPU pilot used seed 42, 206 training queries and 16 October validation queries. After one epoch it achieved Recall@10 of 0.02911, NDCG@10 of 0.09083 and sampled ROC AUC of 0.55049. The validation scores reproduced exactly after reload. [Run record](../results/model_pilots_upstream/lrurec/run.json)

Using that checkpoint, we also checked all 55,920 raw October scores against the authors’ complete forward pass with the same weights and independent application of the fallback rules. With chunks of 256 candidates, the adapter encoded each history once. The maximum absolute difference was 3.58 × 10⁻⁷, and all 16 ordered Top-10 lists matched. Reproduce this comparison with `python -m scripts.check_lrurec_scores --run-dir <run-directory>` using the directory containing `best.pt` and `run.json`. [Score comparison](../results/model_pilots_upstream/lrurec/author_score_check.json)

The earlier independent implementation and its CPU results are archived at commit `69fcba3`. [Original CPU run](https://github.com/NobunagaK/online-retail-top10-recommendation/blob/69fcba3e9ffe8e1f558078434f240ff8a4b1eeb0/results/model_pilots/lrurec/run.json)

## D6 training configuration

The shared runner reads `optimizer` and `lr_scheduler`. The baseline defaults to `adam / none`, preserving the original Adam updates and fixed rate. Selecting `adamw` exempts LayerNorm, biases and LRURec’s `params_log` from weight decay. Selecting `cosine` steps the learning rate once per epoch and saves scheduler progress with the checkpoint. D6 pairs `01/08` at the original baseline settings and varies only history length across `02/03/04`. [Registered grid](../configs/d6_search.yaml) · [D6 notebook](../notebooks/d6_tuning.ipynb)

Running the pinned D5 code and the D6 defaults on the same CPU with seed 42 and 16 real October validation queries produced NDCG@10 of 0.09083041086975355 in both runs. All 30 network tensors, 160 Top-10 rows and 4,708 sampled AUC scores matched exactly. The extended runner also loaded the D5 checkpoint directly. [Original D5 pilot](../results/model_pilots_upstream/lrurec/run.json)
