# 用 LRURec 预测月度净有效购买

[English](lrurec_adaptation.md)

LRURec 使用堆叠的线性循环单元表示客户的购买序列。每个单元先把商品的实数嵌入投影到宽度为两倍的复数状态。读取一次购买事件时，用可学习的对角转移更新状态，加上缩放的当前输入，再投影回商品嵌入的宽度并取实部。随后执行残差归一化和逐位置前馈网络。用最后一个有效状态与候选商品的输入嵌入做点积，再加商品偏置，即得到原始候选分数。初始配置使用最近 64 个事件、64 维嵌入和两层循环单元。[Yue 等，WSDM 2024](https://doi.org/10.1145/3616855.3635760) · [项目适配器](../src/models/lrurec_adapter.py) · [配置](../configs/lrurec.yaml)

## 月度任务的适配

- D2 按时间顺序提供净有效购物篮事件，同一客户在不同购物篮中复购同一商品时保留每次事件。序列用编号 0 左填充，适配器将每个填充位置对循环状态的贡献设为零。并行前缀计算每一步将涵盖区间扩大一倍，支持任意序列宽度，并输出与各原位置对应的状态。共用运行器取最后一个有效状态。[D2 数据协议](data_protocol.zh-CN.md) · [共享序列处理](../src/models/sequence_objective.py)
- 原任务预测下一个商品。本项目按“客户–月份”构造监督样本，在截止时可见的完整目录上计算 softmax，再对该月去重后的目录内正例取平均负对数概率。四种模型共用候选及正例定义。买过的商品可以再次被推荐，分数相同按商品编码排序。PAD 不进入候选目录。[月度评价器](../src/evaluation/ranking.py)
- 输入与评分使用同一张商品嵌入表。推断时，共享运行器依据训练支持记录，对缺少支持的嵌入行使用有支持行的均值；输出偏置根据独立的支持掩码计算回退均值。商品原编号与候选资格保持一致。比较模型大小时，PyTorch 把一个复数投影权重计作一个参数元素，其实部和虚部则合计两个可学习的实数标量。[研究计划，第 3 节](CS5481_Project2_Research_Development_Plan_3.4.md)

## 上游来源与分数核验

本适配器依据 Zhenrui Yue、Yueqi Wang、Zhankui He、Huimin Zeng、Julian McAuley、Dong Wang 的论文《Linear Recurrent Units for Sequential Recommendation》独立实现。对照基准使用作者仓库版本 `e02a559bd8c42342f5c53d69cefea2589becb22c`，对照源码由运行者从作者仓库获取。[官方编码器与原始评分层](https://github.com/yueqirex/LRURec/blob/e02a559bd8c42342f5c53d69cefea2589becb22c/model/lru.py) · [官方评价器](https://github.com/yueqirex/LRURec/blob/e02a559bd8c42342f5c53d69cefea2589becb22c/trainer/lru.py)

在已安装 PyTorch 和 pytest 的项目环境中，从项目根目录执行下列命令。可选上游测试先核对作者源文件的 SHA-256，再给两个实现载入相同的复数转移、投影、归一化、嵌入及偏置参数，关闭 dropout，逐个比较所有真实商品在最后有效位置的原始分数。样例覆盖单事件、重复商品和宽度为 5 的不同左填充长度。浮点容差为 `atol=2e-6`、`rtol=2e-5`。在 PyTorch 2.8.0 的 CPU 环境中，实测 33 个原始分数，最大绝对差为 1.49 × 10⁻⁸。其他测试分别验证逐时间步复算、填充、模型恢复和编码器与评分层梯度。[测试代码](../tests/test_lrurec.py)

```bash
git clone https://github.com/yueqirex/LRURec.git /tmp/lrurec-reference
git -C /tmp/lrurec-reference checkout --detach e02a559bd8c42342f5c53d69cefea2589becb22c
LRUREC_REFERENCE_FILE=/tmp/lrurec-reference/model/lru.py python -m pytest -q tests/test_lrurec.py
```
