# 用 LRURec 预测月度净有效购买

[English](lrurec_adaptation.md)

训练入口直接实例化 Yue 等人在提交 `e02a559bd8c42342f5c53d69cefea2589becb22c` 中发布的 `LRU` 类，用作者的商品嵌入层和线性循环模块编码购买历史。初始配置使用最近 64 个购买事件、64 维隐藏状态和两层循环单元。[论文](https://doi.org/10.1145/3616855.3635760) · [作者实现](https://github.com/yueqirex/LRURec/blob/e02a559bd8c42342f5c53d69cefea2589becb22c/model/lru.py) · [项目适配器](../src/models/lrurec_adapter.py) · [配置](../configs/lrurec.yaml)

## 获取作者源码

固定提交中未找到适用于 `model/lru.py` 的明确软件再分发许可。项目提供适配器和获取命令，由运行者从作者仓库获取原文件。脚本核对 SHA-256 `149e0ccfe03c7eba7a51bf405d179f8a3f395011c0f730169aec2b59c9c156b1`。本地可将 `LRUREC_CACHE_DIR` 设为长期保留的目录，再执行下列命令。Colab notebook 使用项目的 Drive 缓存。如果已有离线副本，可用 `LRUREC_REFERENCE_FILE` 直接指定该版本源文件。[源码获取脚本](../src/models/lrurec_source.py) · [作者固定提交](https://github.com/yueqirex/LRURec/tree/e02a559bd8c42342f5c53d69cefea2589becb22c)

```bash
python -m src.models.lrurec_source
python -m src.models.lrurec_source --offline
python -m scripts.train --model lrurec --device cpu --smoke
```

## 从客户购买历史到排序

1. D2 根据每位客户在截止点可见的净留存购物篮商品事件构造时间序列，保留跨发票复购。十月验证和十一月测试共用 2011 年 10 月 1 日零时以前的历史。购买三次的历史可以补齐为 `[0, 0, A, B, A]`。适配器使用作者的商品嵌入、掩码、内部补齐到 2 的幂次的规则和循环模块，取最后一个有效位置的状态。[数据协议](data_protocol.zh-CN.md)
2. 评价时每条查询只编码一次。适配器用末状态与候选商品的 `embedding.token.weight` 行做点积，再加上 `model.bias` 中对应的偏置。该值对应作者评价器屏蔽历史商品之前的末位置分数。客户此前买过的 A 仍有资格进入 Top-10，候选目录也可以分块评分。[作者评分层](https://github.com/yueqirex/LRURec/blob/e02a559bd8c42342f5c53d69cefea2589becb22c/model/lru.py#L72-L87) · [作者评价器](https://github.com/yueqirex/LRURec/blob/e02a559bd8c42342f5c53d69cefea2589becb22c/trainer/lru.py#L39-L49)
3. 项目按月训练。先在当时可见的候选目录上计算 log-softmax，再对目标月内去重的净留存正例取平均负对数概率，各客户–月份等权。推断时，无训练信号的商品嵌入使用本次训练输入或损失支持的所有嵌入行的均值，商品偏置只使用进入过损失的偏置行计算均值。编号 0 专用于填充。[月度损失与训练支持掩码](../src/models/sequence_objective.py)
4. 固定权重并关闭 dropout，对含复购、长度不同的历史逐个比较项目适配器和作者完整前向计算的候选分数。另行检查分块与完整目录打分是否一致、编码器及评分头能否更新、重载后能否复现 Top-10。用种子 23 在 CPU 上核对 3 条历史和 11 个候选的 33 个分数，最大绝对误差为 1.49 × 10⁻⁸。[测试](../tests/test_lrurec.py) · [作者模型试训记录](../results/model_pilots_upstream/summary.csv)

作者版本在真实 CPU 小样本上使用种子 42，包含 206 条训练查询和 16 条十月验证查询。一轮训练后，Recall@10 为 0.02911，NDCG@10 为 0.09083，抽样 ROC AUC 为 0.55049。重载权重后验证分数完全复现。[运行记录](../results/model_pilots_upstream/lrurec/run.json)

使用这份权重，独立应用均值回退规则后，我们还逐一对照了作者完整前向计算与项目适配器的 55,920 个十月候选分数。每块取 256 种候选商品，每条历史只编码一次。最大绝对误差为 3.58 × 10⁻⁷，16 条查询的 Top-10 商品及顺序全部一致。可运行 `python -m scripts.check_lrurec_scores --run-dir <运行目录>` 复核，其中运行目录包含 `best.pt` 和 `run.json`。[逐分对照结果](../results/model_pilots_upstream/lrurec/author_score_check.json)

此前的独立实现及其 CPU 结果保存在提交 `69fcba3`。[旧版 CPU 试训记录](https://github.com/NobunagaK/online-retail-top10-recommendation/blob/69fcba3e9ffe8e1f558078434f240ff8a4b1eeb0/results/model_pilots/lrurec/run.json)
