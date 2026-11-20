# LRURec adaptation for monthly retail purchases

[简体中文](lrurec_adaptation.zh-CN.md)

LRURec represents the customer's purchase sequence with stacked linear recurrent units. Each unit projects a real item embedding into a complex state twice the size of the embedding. For each event it applies a learned diagonal transition, adds the scaled input, projects the real component back to the embedding width, and applies residual normalization and a position-wise feed-forward network. The final valid state scores a candidate by its dot product with that item's input embedding plus a learned item bias. The default monthly configuration uses the last 64 events, 64 hidden dimensions and two layers. [Yue et al., WSDM 2024](https://doi.org/10.1145/3616855.3635760) · [Project adapter](../src/models/lrurec_adapter.py) · [Configuration](../configs/lrurec.yaml)

## What changes for this project

- D2 supplies a chronological, repeat-preserving sequence of positive net-retained basket events. Histories are left padded with ID zero. The adapter zeros the contribution of padding to each recurrence. Its prefix scan doubles the covered interval at each step, accepts arbitrary sequence widths, and returns states at the original positions. The common runner selects the last valid state. [D2 protocol](data_protocol.md) · [Shared sequence handling](../src/models/sequence_objective.py)
- The original task predicts the next item. Here the supervised example is one customer-month. Training uses a softmax over its full cutoff catalog and the mean negative log probability of all distinct catalog positives. The four models share candidate and label definitions. Previously purchased items keep their scores, and the evaluator resolves ties by StockCode. PAD stays outside the candidate set. [Monthly evaluator](../src/evaluation/ranking.py)
- The input and scoring head share item embeddings. For evaluation the common runner applies mean supported-row fallback to the tied embedding and separately to the output bias using its own support mask. The original item IDs and candidate membership are preserved. For model-size comparisons, PyTorch counts each complex projection weight as one parameter element; its real and imaginary parts contribute two trainable real scalars. [Research plan, `3](CS5481_Project2_Research_Development_Plan_3.4.en.md)

## Source and parity check

The architecture is independently implemented from *Linear Recurrent Units for Sequential Recommendation* by Zhenrui Yue, Yueqi Wang, Zhankui He, Huimin Zeng, Julian McAuley and Dong Wang. The authors' source at revision `e02a559bd8c42342f5c53d69cefea2589becb22c` provides the comparison implementation. The parity check reads a separate local copy downloaded from the authors' repository. [Official encoder and raw scoring head](https://github.com/yueqirex/LRURec/blob/e02a559bd8c42342f5c53d69cefea2589becb22c/model/lru.py) · [Original evaluator](https://github.com/yueqirex/LRURec/blob/e02a559bd8c42342f5c53d69cefea2589becb22c/trainer/lru.py)

From a project environment with PyTorch and pytest installed, run the following from the project root. The optional upstream test checks the source file's SHA-256, loads identical complex transitions, projections, normalizations, embeddings and bias into both networks, disables dropout and compares every real item's raw final-position score. The examples include a one-event history, repeats and unequal left padding with a sequence width of five. Floating-point tolerances are `atol=2e-6` and `rtol=2e-5`. On CPU with PyTorch 2.8.0, all 33 raw scores matched within a maximum absolute difference of 1.49 × 10⁻⁸. The other tests cover a direct time-step recurrence, padding, checkpoint restoration and encoder/head gradients. [Tests](../tests/test_lrurec.py)

```bash
git clone https://github.com/yueqirex/LRURec.git /tmp/lrurec-reference
git -C /tmp/lrurec-reference checkout --detach e02a559bd8c42342f5c53d69cefea2589becb22c
LRUREC_REFERENCE_FILE=/tmp/lrurec-reference/model/lru.py python -m pytest -q tests/test_lrurec.py
```
