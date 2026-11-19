# CS5481 项目二｜在线零售商品推荐研究与代码开发计划

[English](CS5481_Project2_Research_Development_Plan_3.4.en.md) · [简体中文](CS5481_Project2_Research_Development_Plan_3.4.md)

**Project 2 · Online Retail Recommendation · v3.4 · 2026-11-20**

给定客户在截止日前的购买历史，为指定的后续月份推荐 10 种商品，目标是在该月购买、且到月末扣除已匹配冲销后仍有正余量的商品。正式实验并列比较 **MF、LightGBM、LRURec 和 Mamba4Rec**。四个模型各自训练、独立给同一批客户和候选商品打分，都参加验证选模和最终测试。

**课程口径与进度**　项目 PDF 第 3 页原文写有 “exactly three”。按组员向任课老师确认的最新口径，三个是最低数量，可以增加。团队保留课堂 MF、指定的 LightGBM 和近期模型 LRURec，并将同在近期算法清单中的 Mamba4Rec 纳入正式实验。月份、预算和评分细则是本组的研究设计。团队已核对课程材料、原始 Excel 和上游源码，并在本地用完整工作簿复现了异常分析、逐行清洗和购物篮聚合。D1 训练期 EDA 覆盖 2010 年 12 月 1 日至 2011 年 9 月 30 日完成的正价商品购物篮，第 2 节记录实测结果。D2 在 CPU 上构建各截止点的历史、候选、查询及独立存储的训练与验证标签。下一项模型试训是 MF。各模型的 Colab 兼容性、实际耗时及排名成绩仍待测量。

[研究计划](#research) · [代码开发计划](#development) · [评分与提交清单](#submission) · [材料与来源](#sources)

<a id="research"></a>

## 一、研究计划

### 1. 用实验回答的三个问题

| 研究问题 | 实验方法 | 报告中应看到的证据 |
|---|---|---|
| 面对同一批客户和商品，哪种方法的 Top-10 排序最好？ | 用同一验证集比较课堂矩阵分解 MF、使用显式特征的 LightGBM、使用线性循环的 LRURec 和使用选择性状态空间模型的 Mamba4Rec。验证选模后冻结四个模型。 | 验证与测试表列出四个模型的 NDCG@10、Recall@10，并单列抽样 ROC AUC，同时注明评估客户数、候选商品数和配置。 |
| 两种序列模型分别在哪类购买历史上更有用？四种方法通常漏掉什么？ | 在相同历史长度上限、标签、候选和评价代码下比较 LRURec 与 Mamba4Rec，并把四种方法按短／长历史、复购／首次购买、热门／长尾分组。 | 各组人数与指标、热门商品曝光、冷启动覆盖，以及同一客户的真实购买与四列 Top-10。 |
| 推荐质量的差异值得多少计算成本？ | 记录训练和全候选推理耗时。两种序列模型尽量用同型号 Colab GPU、相同批量和精度，登记实际参数量、有效序列长度及峰值显存。 | 质量与耗时对比表。硬件不同的运行各自标注设备，报告可直接比较的结果。 |

四种方法的特征与训练目标有差异。两种序列模型的共同起步配置（最近 64 个事件、隐藏维 64、2 层、同一月度损失和种子 42）纳入各自搜索预算，并单列这对配置的验证结果与参数量。主表另报各模型独立调优的配置，注明实际序列长度。可选的顺序消融保留篮内商品，仅打乱各历史中的订单顺序，在训练样本上重新拟合，用相同干预下的验证查询比较结果。

### 2. 数据来源、清洗与探索

数据为 UCI **Online Retail**。上传工作簿 `Online Retail.xlsx` 的工作表为 `Online Retail`，共有 **541,909 行、8 列**，时间范围为 **2010-12-01 08:26 至 2011-12-09 12:50**。原表含 25,900 个不同发票号、4,372 个非空客户 ID、4,070 个不同商品代码。报告和 README 使用 D. Chen（2015）、数据 DOI `10.24432/C5BW33` 和 CC BY 4.0 署名，并注明所做清洗。[数据主页](https://archive.ics.uci.edu/dataset/352/online+retail) · [数据 DOI](https://doi.org/10.24432/C5BW33)

每条原始明细有 Excel 行号。异常工作簿记录原表标记、符合条件的商品正负明细和各笔商品 C 单的冲销去向。清洗 notebook 按五项课程问题解释处理理由，可定位每一行。原始工作簿 SHA-256 为 `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`。[五项课程要求，第 3 页](../resource/Group_Project_Tutorial_Recommender_Systems.pdf#page=3) · [异常分析](../docs/technical/transaction_exception_analysis.ipynb) · [清洗与课程问题回答](../notebooks/project2_colab.ipynb) · [训练期探索](../notebooks/training_eda.ipynb)

| 原表证据 | 正价购买样本的处理规则与理由 |
|---|---|
| 按 8 列比较有 5,268 条完全重复的额外明细。135,080 行缺客户编号，涉及 3,710 张发票，同票没有其他明细能提供编号。 | 每组完全相同记录保留首条。匿名商品交易进入整体经营账 A。个人历史与客户账 B 要求已知客户编号。缺描述时仍可用商品编码识别品项。 |
| 原表 9,288 条 C 单明细的数量均为负。去重并核实单价和商品身份后有 8,665 条商品 C 明细。严格早于、同价且数量恰好相反的原表候选共有 2,816 条，仅对应原始证据查询。 | 对每条 C 单，查找购买时间不晚于冲销、同客户同商品、可冲销余量足够的最近一条正单。同分钟优先选择位于 C 单之前的 Excel 行，再按行号升序择一。每次匹配消耗可冲销余额。英国、指定 23 个欧洲地区、其他已知地区分别以 72、120、240 小时为取消上限，含边界；超过门槛按项目口径归为退货。工作簿未记录发货、签收或实物退回。 |
| 1,336 条非 C 负数量明细全部零价且缺客户编号，备注有盘亏和报损。原表无零数量行。 | 这批明细按项目规则列入库存核销。商品 C 明细按其实际金额进入经营账。匿名或匹配失败的贷项没有可扣减的原正单；地区未知但已匹配到原单的贷项照常扣减，地区审计单列。 |
| 两条负价和一条正价会计明细均是坏账调整。2,515 条零价中有 40 条数量为正且有客户编号。发票 `550188` 的商品 `22636` 有 8 件付费和 1 件另记零价。 | 商品账排除坏账及零价行，保留同票付费商品，各行先按自身单价计额。有效大额采购正常保留。两笔完全冲销 `C541433` 和 `C581484` 生效后，对应大单从当时及以后的购物篮中移除。 |
| 训练期逐项审查 38 个编码，冻结 18 个精确排除编码。数字编码 `23444` 是次日运费，`23574` 是包装费；`M` 和 `m` 的原文只有 `Manual`，三条 `B` 明细均写着 `Adjust bad debt`。 | 依据训练期证据排除运费、费用、折扣以及无法识别具体商品的手工记录等。基础过滤后仍有 279 条 `Manual` 在商品筛选时剔除，三条 `B` 在账表按客户身份分开之前，即按 A 前缀坏账单剔除。保留靠垫配套垫 `PADS` 的 3 条训练期正价购买（单价 £0.001）。保留原始编码的大小写与后缀，同名商品不自动合并。对所有时间段应用同一冻结规则。[商品规则与原表例证](../results/merchandise_rules.csv) |

**全表对账。** 整体经营账 A 有 531,112 行，净额 £9,770,043.163；已知客户账 B 有 399,563 行，净额 £8,264,201.933。保留原表不足一便士的单价精度。8,665 条商品 C 单中，取消 1,473 条，按项目定义的退货 5,883 条，已知客户未匹配 1,147 条，缺客户编号 162 条。全期回顾口径共有 **388,476 条有效购买明细、383,669 次商品事件、18,280 个购物篮、4,323 位客户、3,644 种商品**。五步基础过滤的 392,692 行保留作旧口径对照。[整表对账](../results/reversal_benchmark.csv) · [商品 C 单匹配明细](../results/credit_allocations.csv.gz) · [匹配规则差异](../results/reversal_match_differences.csv)

**聚合与生效时间。** 自每条匹配贷项可见时从原正单扣减对应件数。取消与退货均按同一规则扣减。单行全额冲销后件数为零，部分冲销则保留余量。该商品在同一篮中所有行的剩余件数合计为正，才保留商品事件。剩余件数乘原购买单价得到有效购买金额，经营净额则按每条正负明细各自的单价入账。按客户、发票及商品汇总，有正余量的一种商品在一篮中计一次事件。整篮完成时间按冲销调整前的合格购买明细确定。[经营账 A](../data/processed/signed_ledger_a.csv.gz) · [客户账 B](../data/processed/signed_ledger_b.csv.gz) · [10 月 1 日购物篮快照](../data/processed/training_basket_events.csv.gz)

**四个模型共用的输入。** 四个模型共用各截止点的购物篮定义。D2 每个月先重放当时可见的正负交易及冲销分配，选择已完成的篮子，再派生各模型的输入。直接从全期回顾表按日期切片会引入后来的冲销。四个主模型都只使用与地区阈值无关的购买事件字段。依赖地区时限的取消／退货标签、退货计数和退货率作审计用。如果以后将它们加入 LightGBM 的验证集特征对照实验，该实验的特征和得分可能随时限改变。MF 与 LightGBM 使用客户–商品交互和历史特征，包括购买过该商品的发票数、累计件数和最后购买时间。按发票计次可避免一次大额采购被当作多次偏好表达。LRURec 与 Mamba4Rec 共用一份按时间排序的商品 ID 序列，跨篮复购保留。词表、分位点和其他需拟合的转换均以该快照允许的历史为准。

**D1 训练期探索性分析。** 截至 2011 年 10 月 1 日零时（不含），**260,919 条有效明细形成 258,532 次商品事件和 13,001 个购物篮**。3,608 位客户、3,495 种商品共有 12,609,960 个可能组合，观察到 187,452 个，稀疏度为 98.51%。其中 71,073 次事件（27.49%）在同一客户更早的篮中有同一 SKU。分析使用截止点快照核算月度活动、价格数量分布、购物篮大小、历史长度和商品热度。D2 按各训练截止点重建历史。英国时限取 72 或 120 小时的对照在八个训练月及十月验证月中得到一致的原单分配、购买历史和月末正例。[D2 数据协议](data_protocol.zh-CN.md) · [阈值核验](../results/net_retention_threshold_check.csv) · [训练期统计](../results/training_eda/summary.json) · [月度统计](../results/training_eda/monthly_activity.csv) · [商品排名](../results/training_eda/item_popularity.csv)

### 3. 时间切分、候选商品与标签

课程要求按时间切分，并从训练数据计算客户历史、商品热度和其他特征。沿用上一版的完整自然月设计。下表区间均含起点、不含终点。

| 用途 | 时间范围 | 可以使用的信息 |
|---|---|---|
| 训练 | `[2010-12-01, 2011-10-01)` | 拟合模型、建立历史快照、词表及训练用统计。 |
| 验证 | `[2011-10-01, 2011-11-01)` | 十月交易只产生验证标签，用于早停、调参及选择最终模型。 |
| 最终测试 | `[2011-11-01, 2011-12-01)` | 配置冻结后打开十一月标签，报告四个模型的测试指标。 |
| 不完整尾月 | `[2011-12-01, 2011-12-10)` | 实测最后交易为 12 月 9 日 12:50，事后冲销只能追踪到此时。这个不完整月份不纳入训练、验证或测试。 |

**训练样本。** 以 2011 年 2 月至 9 月的每个月初作为训练快照。查询为 `(customer_id, target_month)`，须有月初之前的购买历史。该快照的目录 `I(q)` 只含当时已出现的有效商品，正例 `Y_q` 为目标月内买过的不同有效商品与 `I(q)` 的交集。训练清单仅保留 `Y_q` 非空的查询，另计被排除的客户与新商品。四个模型共用查询、目录与正例，MF 和 LightGBM 还共用训练负例。每个目标月均按月末截止点重放有正余量的购物篮事件，再按客户和 SKU 去重；某一行全额冲销后，同月另一次有余量的购买仍可保住商品正例，不用此后发生的冲销修改已冻结的月度标签。

**验证和测试的历史。** 两次预测都使用 `history_end = 2011-10-01 00:00` 之前已完成的购物篮及已发生的冲销。十月交易产生验证标签，不进入十一月输入或模型重训。对客户或客户–商品对，`days_since_last_purchase` 等于最后一次有净余量的购买篮至 `history_end` 的实际天数，保留不足一天的部分。7、30、90 天的发票数均统计 `[history_end − N 天, history_end)`。例如，9 月 30 日中午的购买对两次预测都是 0.5 天前。同一键的历史特征在两月完全相同，两个月实际入选评价的客户仍可不同。

**预测跨度与季节性需求。** 从同一个历史截止点预测，十月是下一个完整月份（T+1），十一月为第二个月（T+2）。若未来观察到两月排名得分有差异，预测跨度、购买客户或商品构成的变化、圣诞前备货都是可能解释。报告应将两个跨度、客户覆盖和得分并列。只有一个验证月和一个测试月，无法分离这些因素的影响。

**跨月冲销与右删失。** 各月标签统一冻结在其目标月结束的排他截止点。审计先在候选和客户资格筛选前，按“月份、客户、SKU”去重，再使用截至 12 月 9 日 12:50 已记录的冲销复核各目标月的购买余额。二月至九月有 653/203,020 个月末正例归零，十月为 105/43,978。仅看九月是 149/37,001。导致这些正例最终归零的贷项中，122 个在十月，26 个在十一月，1 个在十二月。越接近数据末端的购买，可观察的后续时长越短。十一月购买的月后冲销仅能再看十二月一日至九日。[测试前审计表](../results/d2_audit/label_delay_pretest.csv) · [审计计算](../scripts/audit_label_delay.py)

项目负责人另行提供的工作簿核对涵盖二月至十一月，结果为 788/301,922（0.26%），其中十一月为 30 个。这个口径包含留出的测试月，衡量的是客户资格与候选筛选前的记录变化。要估计查询等权 Recall@10 或 NDCG@10 的变化，还要取得每条查询的合格正例和推荐排名。选择模型、超参数和特征仅依据十月验证。[数据协议](data_protocol.zh-CN.md)

**评价对象。** 统一候选集 `I_train` 包含整个训练期清洗后出现过的全部有效商品，允许推荐客户以前买过的商品。每个窗口选取在至少一个有效训练查询中出现过、且当月买过至少一个候选商品的客户。主指标的相关集合 `Y` 为当月真实购买集合与 `I_train` 的交集，同一商品当月多次购买只计一次。四个模型在每个窗口使用相同的查询和候选清单。评估时逐个或分批打分，全候选都要参与排名。

**目录内但缺少训练信号的商品。** 最后一个训练快照在 9 月 1 日，评估目录截止到 9 月 30 日。10 月 1 日快照中有 **3,495 种商品**，其中 **114 种**在该快照中最早保留的购买落在 9 月。各商品实际是否获得训练信号，按各截止点的输入和候选集核对。保留其原 ID、购买历史和候选资格。MF、LRURec、Mamba4Rec 对商品参数表记录训练支持掩码（真正进入过训练输入或损失计算的行，共享矩阵取二者并集，PAD 除外）。每次验证或测试打分时，未获得过训练数据信号的嵌入和商品偏置使用各自表中已获支持商品行的均值。此规则也用于序列输入。LightGBM 按该商品截止时可得的历史特征正常评分。记录受影响的商品数、正例数和命中数，三项指标使用同一次评分输出。

**覆盖统计。** 对每个窗口报告活跃客户、训练期未见客户、已有历史但没有有效训练查询的客户、只买目录外商品的客户，以及训练客户中当月无购买的人数。主表报告可评估客户占全部活跃客户的比例。另对有有效训练查询且当月有购买的所有客户计算补充 Recall@10，其分母含该月全部不同有效商品，只买目录外商品的客户得 0 分，对该组客户等权平均。单列目录外正例占比和目录内但缺少训练信号的商品。

例如，一位客户在九月底前买过 A，十月买了 A、B、B。如果 A 和 B 都在训练目录，十月标签就是 `{A, B}`。A 是一次有效复购，B 计作一个相关商品。若 B 到十月才首次出现在有效购买记录中，主指标的标签变为 `{A}`，B 仍计入新商品覆盖统计及全商品分母的补充 Recall。

### 4. 四个模型的输入、训练目标与输出

| 模型及课程位置 | 输入与训练方式 | 推荐分数的含义与实现重点 |
|---|---|---|
| **MF**，课堂模型，Lecture 7 第 34–35 页 | 从客户–商品嵌入点积加偏置得到 `s(u,i)`。用训练月份的二元购买标签、采样负例、按查询归一的二元交叉熵和 L2 正则拟合。 | 为客户对全候选计算偏好分数。报告说明课件的低秩结构如何用于隐式购买反馈，以及采样分数的排序用途。 |
| **LightGBM**，项目二指定的树模型 | 使用 `LGBMRanker(objective="lambdarank")`。一个客户的一个目标月是一组，组内每行对应一个候选商品，标签为该月是否买过该商品。 | 为每个客户–商品对输出排序分数。解释历史频次、近因和复购特征。特征、标签、查询组和训练负采样均写入报告。 |
| **LRURec**，近期模型，WSDM 2024 | 官方线性循环编码器读取截止日前的商品 ID 序列。用最后一个有效状态学习目标月的正商品集合。 | 用原商品评分层给全候选打分。对照 Mamba4Rec 时固定输入序列和监督方式。 |
| **Mamba4Rec**，第二个近期模型，2024 | 官方 Mamba 层根据输入更新序列状态。使用与 LRURec 相同的客户历史、月度正例及损失函数。 | 取最后一个有效位置，以官方商品嵌入评分方式对相同候选打分。独立保存验证结果、权重和 Top-10。 |

**MF 与 LightGBM 的负采样。** 保留查询全部正例。从该快照目录减去本月全部正例后，以固定种子均匀无放回采样，起步为每个正例最多 5 个负例，组内去重。负例表示当月未观察到购买。训练特征与负样本生成均不得读取验证或测试交易。将每组正负个数与采样 seed 持久化。验证和测试对全目录评分。

**LightGBM 特征与分组。** 客户特征用订单数、平均篮大小、消费额、最近购买时间。商品特征用购买客户数、订单数和训练期价格统计。客户–商品特征用是否买过、历史发票数、最后一次有净余量的购买到 `history_end` 的实际天数。7/30/90 天发票计数同样以 `history_end` 截止。构造每月样本时先截取允许历史，再做聚合。按 `query_id` 连续排序，传给 `group` 的值是各组行数，必须满足 `sum(group) == len(X)`。全候选验证和推理按客户分批，节省内存。[LightGBM 排序接口](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)

**两种序列模型共用的数据与训练目标**

1. 以 `(customer_id, history_end)` 建立可复用的历史序列。不同购物篮按发票时间排序，同一时间按规范化的发票号打破并列，同篮内按 `StockCode` 固定排序。这个篮内顺序仅用于复现。一张发票里同一商品计一个事件，跨发票复购保留。各训练截止点首次可见的商品依次追加 ID。新增 ID 按最早净保留购物篮的时间排序，同一时间按精确 StockCode 打破并列。商品后来退出某月目录仍保留原 ID，真实 ID 从 1 开始，0 专用于填充。
2. 存储变长的 `history_item_ids` 和原始长度。模型加载时取最近 `L` 个事件并计算有效长度 `seq_len`，共同起步值为 `L=64`。完整历史与目标月标签分别保存。两种模型可以按各自实现补齐序列，必须读到同一批真实商品、取到最后一个有效状态。记录各组历史长度、截断比例和截断点落在同一购物篮内部的查询占比。同篮商品的固定次序仅用于复现。
3. 以一个客户的一个目标月为一条监督样本。设模型给候选商品的分数为 `s_q`，仅对当时可见目录 `I(q)` 做 softmax，对当月正商品的负对数概率取平均，每个查询等权。两种序列模型用同一目标。填充 ID 和目录外商品在归一化前排除，曾购买商品保留为合法候选。

$$
\mathcal L(q)=-\frac{1}{|Y_q|}\sum_{i\in Y_q}\log\frac{\exp(s_{q,i})}{\sum_{j\in I(q)}\exp(s_{q,j})}
$$

**LRURec 接口。** 已核对作者提交 `e02a559bd8c42342f5c53d69cefea2589becb22c`。按作者实现左补 0，读取最右的有效状态，用共享商品嵌入点积加商品偏置打分。上游常规评价路径把历史商品分数设为极低值，项目评价器须绕开这一步，仅排除 PAD 和目录外商品。从同一组权重核对项目的末位置分数与上游屏蔽前分数一致。用历史 `[A,B,A]`、目标含 `A` 的合成例验证 A 保留原分数并有资格进入 Top-10。[作者评分头](https://github.com/yueqirex/LRURec/blob/e02a559bd8c42342f5c53d69cefea2589becb22c/model/lru.py#L72-L87) · [作者评价器](https://github.com/yueqirex/LRURec/blob/e02a559bd8c42342f5c53d69cefea2589becb22c/trainer/lru.py#L39-L49) · [论文](https://doi.org/10.1145/3616855.3635760)

**Mamba4Rec 接口。** 已核对作者提交 `44a0c8432b34e3b89d46035b10e8525b1bd368dc`，保留选择性状态空间层、序列表示和商品嵌入打分。官方代码以 `item_seq_len - 1` 取状态，每条训练样本读取一个目标商品。适配器将真实 ID 放在序列左侧、在右侧补 0，并将分数交给上述整月多正例损失和统一评价器。跳过 RecBole 的默认切分与样本生成，明确保留复购候选。通过测试核对编码器与评分头均获得梯度，验证时按相同候选顺序输出分数。[作者代码](https://github.com/chengkai-liu/Mamba4Rec/blob/44a0c8432b34e3b89d46035b10e8525b1bd368dc/mamba4rec.py) · [论文](https://arxiv.org/abs/2403.03900)

### 5. 调参、冻结与最终测试

**统一指标。** 四个模型均在完整候选目录上计算两个主要排名指标。`Recall@10 = Top-10 中命中的不同相关商品数 / |Y|`，其中 $Y$ 为该客户目标月内实际购买且在候选目录中的不同商品集合。NDCG@10 对第 $r$ 名的命中赋予 $1/\log_2(r+1)$ 的权重，再除以 $\min(10,|Y|)$ 个相关商品排在最前面时的理想折扣和。两个指标均先逐客户–月份查询计算，再对查询等权平均。Top-10 商品去重，分数并列按固定的 `StockCode` 顺序排列。

**抽样客户–商品对的 ROC AUC（补充指标）。** 题目允许额外报告此项，本组决定在验证和最终测试中都报告。对每个客户–月份查询，保留 $Y$ 中全部商品为正例，从同一候选目录中该月未观察到购买的商品里均匀、无放回抽取负例。若该查询有 $P$ 个正例、其余候选有 $M$ 件，则一次抽取 $\min(10P,M)$ 个不同负例。按稳定商品顺序，用独立抽样种子 `auc_sample_seed=5481`、分区和 `query_id` 确定样本。验证、测试各存一份清单，四个模型及全部训练种子共用各自分区的清单。测试清单在 `selection.json` 冻结后生成。

**AUC 的计算与用途。** 使用截取 Top-10 之前的原始商品分数。在同一查询内，每个正例与每个抽中负例比较，正例分数更高计 1 分，同分计 0.5 分，更低计 0 分。先求该查询所有比较的平均分，再对同时有正、负例的查询等权平均，结果列标为 `Sampled ROC AUC (query-macro)`。只含单一类别的查询单独计数，仅从 AUC 均值中排除。若整个分区无有效查询，该项记为 `NA`。AUC 用于解释购买与未观察到购买的商品能否被区分，早停、调参和最终选模采用下表及步骤 3 的验证排名规则。

**LightGBM 早停。** 验证集按客户–月份提供完整候选与 `eval_group`。设 `eval_at=(10,)`，只监控一个验证指标。先用含同分的样例核对内置 NDCG 与项目评价器的并列裁决及宏平均值。如果结果不同，关闭内置指标，改用项目的 NDCG@10 评价函数和早停回调。预测、保存与重载统一使用最佳迭代数。[LightGBM 排序参数](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html) · [早停回调](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html)

下表是可执行的起步值，实际试验的配置、迭代数与耗时逐条登记。

| 模型 | 起步配置 | 验证集搜索变量 | 停止规则 |
|---|---|---|---|
| MF | 维度 32，学习率 0.001，L2 为 1e-5，batch 1024 | 维度 16/32/64，学习率 0.001/0.003，L2 为 1e-6/1e-5/1e-4 | 上限 100 个 epoch，验证 NDCG@10 连续 10 轮未提升时早停。 |
| LightGBM | 31 叶，学习率 0.05，最小叶样本数 20 | 叶数 15/31/63，学习率 0.03/0.05/0.1，最小叶样本数 20/50/100，L2 为 0/1/10 | 上限 1,000 棵树，50 轮无提升早停。使用第 5 节规定的全候选验证 NDCG@10 早停，记录并使用最佳树数。 |
| LRURec | 隐藏维 64，2 层，序列长度 64，学习率 0.001，batch 64 | 隐藏维和序列长度各选 32/64/128，学习率 0.0003/0.001，dropout 为 0/0.2 | 上限 100 个 epoch，验证 NDCG@10 连续 10 轮未提升时早停。 |
| Mamba4Rec | 隐藏维 64，2 层，序列长度 64，`d_state=16`、`d_conv=4`、`expand=2`，学习率 0.001，batch 64 | 优先在固定 `L=64` 下取 `d_state` 为 16/32、学习率 0.0003/0.001、dropout 为 0/0.2 的预登记组合，控制在统一预算内 | 上限 100 个 epoch，验证 NDCG@10 连续 10 轮未提升时早停。 |

1. 先用四个模型各自的小样本试训估算训练和全候选验证的成本。正式搜索前预登记组合。默认每个模型最多 8 组（含基准配置），共最多 32 组，统一以 `seed=42` 搜索。若 Colab 预算紧，在搜索前统一缩减到每模型最多 4 组，共最多 16 组。种子复核另外计入运行次数。
2. 常规预算下，每模型验证表现最好的两组用 `42、2026、3407` 三个种子复核。缩减预算下复核每模型的第一组。复用已完成的同种子结果。常规预算最多 48 次正式拟合（32 次初筛加 16 次复核），缩减预算最多 24 次（16 加 8），另计试训、可选消融和失败重跑。LightGBM 如为确定性配置，注明种子方差可能为零。
3. 按三种子**验证集 NDCG@10 均值**选择各模型配置，再用同一规则选出一个最终模型。并列时先比较验证 Recall@10 均值。若剩余并列项均在相同硬件上运行，再比较训练耗时。其余并列项按预登记顺序裁决，模型按 MF、LightGBM、LRURec、Mamba4Rec 排列，同模型内按配置登记顺序。选择判断使用未四舍五入的数值。
4. 把四套配置、各训练种子的最佳权重或迭代数、最终入选模型、评价协议和代码版本写入 `selection.json`。用各模型选定的权重汇总验证结果，再执行一次锁定的整套测试。验证和测试结果表均列出四个模型的 NDCG@10、Recall@10 和补充列 `Sampled ROC AUC (query-macro)`，每项报告三个训练种子的均值与样本标准差（`ddof=1`），反映训练随机性。表注写明各分区的有效 AUC 查询数、跳过查询数及参与计算的正负样本数。完整报告并讨论测试和验证排名之间的差异。

### 6. 结果解释的最低交付

| 主题 | 要计算或展示的证据 | 要回答的问题 |
|---|---|---|
| 稀疏性与历史长度 | 训练期稀疏度，按短、中、长历史分组的指标及人数。历史长度分界取训练分位数并冻结。 | 数据稀疏时，哪种模型的推荐最容易失效？ |
| 热门偏差与覆盖 | 按训练期不同购买客户数降序、同值按 `StockCode` 排序，取目录前 10%（数量向上取整）为热门。报告热门商品所占推荐位比例、推荐过的不同商品数占目录的比例和其余商品的命中情况。 | 分数优势是否伴随过度集中于热门商品？ |
| 复购与首次购买 | 按客户完整可用历史划分复购与首次购买，各自报告正例数、有效查询数及分组 Recall@10。对该组正例非空的查询，以该组命中数除以该组正例数，再等权平均。 | 哪种方法擅长补货式复购？哪些新购最难预测？ |
| 新客户、新商品 | 给出第 3 节的两类覆盖和可服务客户数，并说明引入商品内容等改进能解决哪一类限制。 | 主表中缺少谁？是否高估了系统实际可服务的范围？ |
| 取消、退货与具体失误 | 用经营净额与购物篮调整说明预测目标。按统一规则列出同一客户的历史、目标月实际购买及四组 Top-10。对有冲销记录的案例标出相应截止点可见的取消或退货，至少展示一个真实的失误。 | 哪些购买进入了评价？模型是否再次推荐了客户退过的商品？用实际命中、漏荐和项目退货代理定义的局限解释结果。 |

案例先按客户 ID 的固定哈希从不同历史长度组各选一人，再从有漏荐的查询中按同一顺序选一例。对 LRURec 与 Mamba4Rec 同时列出短、长历史组的客户数、两项指标、参数量与耗时。若开展消融，优先在验证集比较 LightGBM 移除复购特征，或两种序列模型采用同一组短／长截断值。改进建议应能对应具体失误或测量。

<a id="development"></a>

## 二、代码开发计划

### 7. 工作平台与文件位置

按团队确定的网页流程实施。**四个模型的试训和正式训练均在 Google Colab。** MF 和 LightGBM 使用 CPU 运行时，LRURec 与 Mamba4Rec 使用 NVIDIA GPU 运行时。GPT Work 或 Codex 协助开发、审阅代码和分析日志。组员用 MacBook、浏览器或 Cursor 处理 GitHub 提交，并在 Colab 网页启动对应版本的 notebook。

| 位置 | 已有内容或计划用途 | 开发时的约定 |
|---|---|---|
| GitHub `resource/`、`docs/` | 团队可访问的项目题目、课程依据、研究计划、数据协议、上游适配说明与报告稿 | 在小组有权限访问的位置保留课程材料，仓库注明题目和 Lecture 7 第 34–35 页的定位。记录上游 commit 与引用。 |
| GitHub `configs/`、`src/`、`scripts/`、`tests/` | 数据协议、四份模型配置、两个序列适配器、统一评价和测试 | 核心逻辑写成 Python 模块，每次实验记录与代码对应的 Git 提交。 |
| [异常分析 notebook](technical/transaction_exception_analysis.ipynb) | 从原表生成缺失、数量价格等证据，以及双账、商品身份、同分钟匹配、可冲销余量和地区分类的对照。 | 通过 `ExcelRow` 回查每张 C 单和相配正单，对照附件基准解释差额。[异常工作簿](../results/transaction_exceptions.xlsx) |
| [D1 数据清洗 notebook](../notebooks/project2_colab.ipynb) | 在 CPU 上从工作簿重建经营账、客户账、冲销分类、全期有效购物篮和训练期快照。 | 按五项课程要求查看各条规则、证据工作表和实际影响。 |
| Drive `CS5481_Project2_OnlineRetail/data/raw/`、`data/processed/` | 原始 Excel、清洗后的交易和购物篮、训练词表、按截止点构建的序列、分区的查询／候选／标签 | 共用一份数据版本与协议哈希，记录每个时间段的覆盖情况。 |
| Drive `CS5481_Project2_OnlineRetail/runs/<model>/<run_id>/` | 运行配置、日志、状态、选定的模型、分数或 Top-10、带输出的 notebook 副本 | 每次新实验分配新目录。断点恢复显式指定原 `run_id` 并核对模型配置和数据版本。 |
| Drive `CS5481_Project2_OnlineRetail/results/` 与 GitHub `results/` | Drive 汇总审计、图、结果表和案例。提交所用的小表与图复制进 GitHub 和 Canvas 包 | README 给出可访问的具体相对路径。原始数据、密钥、大型权重留在相应数据目录或 Colab Secrets。 |
| GitHub `experiments/exp_XXX/` 与 `experiments/experiment_ledger.md` | 小型验证指标、训练曲线、切片误差摘要、按模型适用的诊断和总实验账本 | 每条结果关联模型、代码、数据版本和 Drive 的 `run_id`。测试结果在冻结后发布。 |

在仓库用锁文件或分模型环境配置记录实际可运行的 Python、PyTorch、LightGBM、RecBole、`mamba-ssm`、`causal-conv1d` 等版本。Codex 可用合成交易表检查数据逻辑。模型训练与硬件验收使用组员实际获配的 Colab 运行时。

**在步骤 D0 验证 Mamba 的运行环境。** 在实际 NVIDIA GPU 上记录 GPU 型号、驱动、CUDA 和 PyTorch 版本，安装官方依赖，用两条不同长度的合成历史完成导入、前向打分、反向更新和断点读取。记录验证通过的版本和安装命令。若失败，排查报错并重试受支持的组合，同时推进公共数据、MF、LightGBM 和 LRURec。若依赖互不兼容，可为 Mamba4Rec 单独锁环境并沿用同一数据和评价器。[Mamba4Rec 安装要求](https://github.com/chengkai-liu/Mamba4Rec)

合成样本检查通过后，两种序列模型各在 GPU 上用真实小样本跑 1 个 epoch 加一次全候选验证，记录耗时、峰值显存和截断率。显存不足时减小批量，必要时按客户或候选分块评分，并报告实际设置。正式预算按测量结果确定。

### 8. 共用数据与模型接口

| 文件或接口 | 必需字段或行为 | 防止的具体错误 |
|---|---|---|
| [数据清单](../results/data_manifest.json)、[五步基线审计](../results/cleaning_audit.csv)、[逐步清洗审计](../results/cleaning_steps.csv)、[商品编码规则](../results/merchandise_rules.csv) | 已生成。数据清单记录工作簿哈希、工作表、字段、时间范围、聚合计数及输出哈希。审计区分原始命中数与首条规则的净剔除数，商品规则给出训练期证据。 | 校验数据版本，避免重复扣除同时命中多条规则的记录。 |
| [逐行决定](../data/processed/row_decisions.csv.gz)、[已知客户账](../data/processed/signed_ledger_b.csv.gz)、[冲销分配](../results/credit_allocations.csv.gz)、[训练期快照](../data/processed/training_basket_events.csv.gz) | 通过 Excel 行号连接正单和冲销，保留原篮完成时间、有效件数、按原单计价的金额和明细数。全期产物供回顾，较早截止点按当时可见记录重放。 | 避免重复扣减、将费用当作商品，或把未来冲销写入过去的历史。 |
| `queries.parquet`、目录与标签清单 | 每条查询保存 `query_id, customer_id, split, history_end, target_start, target_end, catalog_id`。历史、标签和目录分开存储，通过 ID 关联。 | 训练快照看见自己的未来，或某个模型私下删掉难客户。 |
| `histories.parquet`、`item_vocab` | 历史含 `customer_id, history_end, history_item_ids, history_len`。由查询的客户与截止点关联。目标商品集合独立保存。变长历史按时间排序、保留复购，两种序列模型从同一行取最近 `L` 个有效 ID。 | 不同模型采用不同历史、把当月标签写入输入、或把填充当商品。 |
| `score(query_batch, candidate_item_ids)` | 四个模型按传入候选顺序返回二维分数。统一评价器按该查询目录筛选、处理并列、生成 Top-10 并汇总。 | 上游自动过滤复购商品，或不同模型使用不同候选集。 |
| `top10.parquet`、`metrics.csv` | `top10` 保存 `model, seed, split, query_id, item_id, rank, score`。逐轮日志保存轮次或树数、耗时、验证两指标、适用的训练损失。 | 结果仅以屏幕截图存在，无法重算命中或定位最佳轮次。 |
| `auc_pairs.parquet`、`auc_scores.parquet` | 每个评估分区保存固定的 `split, query_id, item_id, label` 清单，记录抽样种子、样本清单哈希。分数表保存 `model, seed, split, query_id, item_id, score`，按 `split, query_id, item_id` 关联标签，分模型、种子复算 AUC。 | 只保留前十名会丢失 AUC 样本。分清训练负采样与评估抽样，可防止不同模型使用不同评估样本。 |
| `run.json`、`status.json`、权重或模型文件 | 记录 `run_id`、代码和上游提交、数据和协议哈希、超参数、种子、软件版本、设备、耗时、早停、采样和截断决策。 | 断线后无法区分任务状态或识别生成结果的配置。 |

D2 将共用的查询、历史、候选、特征和样本表写入 `data/processed/d2/`；模型分数、检查点和实际排名表由 D3 至 D7 生成。大规模评分可以按客户或商品分块。遍历每个查询的完整候选目录时，保留 Top-10、固定 AUC 样本的原始分数和案例所需分数。对少量查询同时计算整批及分块结果，核验排名与抽样分数一致。

### 9. 开发与训练步骤及完成标准

本项目集中完成，按 D0 至 D7 的顺序推进。按表中依赖衔接各步，并逐步保存验收产物。D0 的 CPU 和 Drive 检查通过后即可开展 D1。Mamba 环境检查在 D5b 真实试训前通过。若小样本试训显示 Colab 额度不足，在正式搜索前按第 5 节统一缩减搜索预算。

| 执行步骤 | 编码任务与拟建模块 | 完成标准与交给下一步的产物 |
|---|---|---|
| **D0**　入口与 GPU 环境 | 使用 README、固定版本的 CPU 依赖、D2 数据协议和项目 notebook 复现数据准备。CPU 验证 Drive 输入，NVIDIA GPU 单独检验 Mamba 依赖和一次合成序列更新。 | 能读取原 Excel 并核对哈希。保存实际 GPU、PyTorch、CUDA、扩展版本和试训结果，开发数据协议可同步进行。 |
| **D1**　清洗与探索性分析 | 可重现的异常证据、经营和客户账、冲销分配、有效购物篮及训练期探索。[D1 运行入口](../scripts/prepare_data.py) | 核对两笔全额冲销大单、12 条同分钟匹配、部分冲销和正负金额。10 月 1 日快照有 258,532 次篮内商品事件。 |
| **D2**　查询、共享序列与评价器 | `scripts/build_queries.py` 生成快照、候选、词表与隔离的标签。`src/data/sequences.py` 导出两种序列模型共用的变长历史。[统一评价器](../src/evaluation/ranking.py) 计算 Top-10、两个主指标及第 5 节规定的抽样 ROC AUC。 | 手算指标通过。追加未来交易后训练快照不变。四个模型的查询与目录相同，两种序列模型移除填充后的商品与先后顺序逐条相同。 |
| **D3**　MF 基线 | `src/models/mf.py`、`configs/mf.yaml`，实现嵌入、训练采样和独立评分。 | 小数据上 loss 有限、参数更新。从原始数据到验证 Top-10 完整运行，保存和加载后排名一致，留下一个已工作的模型接口示例。 |
| **D4**　LightGBM | `src/features/asof.py`、`src/models/lightgbm_ranker.py`、`configs/lightgbm.yaml`，实现按历史截止点构造特征和排序组。 | 手工核对一个客户–商品对的频次和最近购买日期。检查组连续、组大小总和、正负例互斥、没有来自未来的特征，保存验证表现与特征字典。 |
| **D5a**　LRURec | `src/models/lrurec_adapter.py`、`configs/lrurec.yaml`、`docs/lrurec_adaptation.md`。接官方线性循环与共享序列，把月度损失放入共用的 `src/models/sequence_objective.py`。 | 用短、长、含复购序列检查掩码。在 Colab GPU 上用真实小样本跑 1 个 epoch，保存、重载、复算验证 Top-10。 |
| **D5b**　Mamba4Rec | `src/models/mamba4rec_adapter.py`、`configs/mamba4rec.yaml`、`docs/mamba4rec_adaptation.md`。接官方 Mamba 层与评分方法，直接调用 D2 序列、`sequence_objective.py` 和统一评价器。 | 在锁定依赖的 Colab GPU 上跑相同真实小样本，检验右侧填充、有效位置、多正例损失、参数更新、断点恢复和全候选 Top-10。 |
| **D6**　调参、选模、测试 | `scripts/train.py`、`scripts/tune.py`、`scripts/freeze_experiment.py`、`scripts/evaluate.py`。按第 5 节执行四个模型的搜索预算与冻结。 | `tuning.csv` 包括所有尝试、种子及失败原因。`selection.json` 先于测试结果生成。四个模型的两个主指标可从冻结的 Top-10 与标签复算，抽样 ROC AUC 可从同一分区的抽样清单和对应分数复算。 |
| **D7**　解释与交付 | `scripts/make_report_tables.py` 导出主表、覆盖、分组和案例。完成技术报告、README 和 5 分钟展示。 | 一名组员在新环境按 README 跑通小样本，并从冻结预测重建主表。报告数字与表格一致，组长完成 Canvas 提交和展示排练。 |

D2 通过后，用 MF 检查从训练到验证的完整流程，再依次试训 LightGBM、LRURec 和 Mamba4Rec。四个模型的小样本验证通过后进入 D6，完成正式训练与调参，用验证集选模并冻结配置，然后统一测试。D7 用冻结的预测完成分析与提交。

### 10. 已有运行入口与后续接口

**从仓库根目录在 CPU 上运行三个 D1 命令。** 异常分析和清洗各自读取经核验的原始工作簿。训练 EDA 从清单核验两份专用训练快照后再进行统计。[异常分析](../scripts/analyze_exceptions.py) · [数据清洗](../scripts/prepare_data.py) · [训练期 EDA](../scripts/analyze_training.py)

```bash
python -m scripts.analyze_exceptions --workbook "data/raw/Online Retail.xlsx" --output results/transaction_exceptions.xlsx
python -m scripts.prepare_data --project-dir . --workbook "data/raw/Online Retail.xlsx" --output results/cleaning_audit.csv
python -m scripts.analyze_training --project-dir .
```

`prepare_data` 的 `--project-dir` 指定项目输出根目录，`--output` 只指定五步基线 CSV 的位置。对 `analyze_training`，`--project-dir` 指向包含两份 D1 输入及其数据清单的项目文件夹。使用仓库默认布局时，三条命令均可省略参数。

**在 CPU 上构建 D2 数据并复核跨月冲销。** 在仓库根目录安装固定版本的依赖，随后可选 notebook 或查询构建脚本运行。默认产物目录为 `data/processed/d2/`，可通过 `--output-dir` 切换到其他目录。单独运行审计会写入 `results/d2_audit/label_delay_pretest.csv`。[CPU 依赖](../requirements.txt) · [D2 notebook](../notebooks/monthly_queries_and_evaluation.ipynb) · [查询构建脚本](../scripts/build_queries.py) · [数据协议](data_protocol.zh-CN.md)

```bash
python -m pip install -r requirements.txt
python scripts/build_queries.py --project-root .
python -m scripts.audit_label_delay
```

**以下为 D3–D7 的拟开发模型接口。** 将在对应的 Colab 运行时执行。各会话共用同一代码版本、数据协议、Drive 项目根目录和实验标识。冻结清单与模型输出使用相对于同一 Drive 项目根目录的路径。

```bash
# CPU 上运行模型试训
python -m scripts.train --model mf --config configs/mf.yaml --smoke
python -m scripts.train --model lightgbm --config configs/lightgbm.yaml --smoke
# 在各自的 GPU 环境试训
python -m scripts.train --model lrurec --config configs/lrurec.yaml --smoke
python -m scripts.train --model mamba4rec --config configs/mamba4rec.yaml --smoke
```

四项试训通过后，各环境在训练集拟合、在验证集选参。汇齐四个模型的验证记录后，执行一次配置冻结。

```bash
# CPU
python -m scripts.tune --model mf --selection-split validation
python -m scripts.tune --model lightgbm --selection-split validation
# 各自的 GPU 环境
python -m scripts.tune --model lrurec --selection-split validation
python -m scripts.tune --model mamba4rec --selection-split validation
# 汇总并冻结
python -m scripts.freeze_experiment --suite core --output results/selection.json
```

在各模型的原运行环境加载同一冻结清单。下文的 MODEL 由 notebook 设为当前模型的注册名。每个模型分别输出验证与测试结果，每次加载三个种子的选定权重。每个分区的评估抽样清单由一个会话生成，其余会话从 Drive 读取并核对哈希。四份测试输出共同构成一轮锁定测试。

```bash
python -m scripts.evaluate --model "$MODEL" --split validation --selection results/selection.json
python -m scripts.evaluate --model "$MODEL" --split test --selection results/selection.json
# 所有模型的输出齐全后汇总
python -m scripts.make_report_tables --suite core
```

`core` 注册 `mf`、`lightgbm`、`lrurec`、`mamba4rec`。`--smoke` 只用训练期小样本和相应验证查询，在指定设备上完成一次更新、评分、保存和重载。正式训练、验证与测试入口分开。测试前校验 `selection.json` 含四个模型及匹配的配置、种子和协议，测试表和日志使用独立目录。

### 11. Colab 网页训练、实验记录与复核

1. GPT Work 或 Codex 起草代码，由 Codex 按任务准备 GitHub PR，组员审阅并确定 Git 提交。组员在 Colab 网页检出该提交、选择运行时并点击运行。notebook 第一条日志写入提交 SHA、数据协议、`run_id`、软件版本和实际设备。Drive 项目根目录可配置为 `/content/drive/MyDrive/CS5481_Project2_OnlineRetail`。
2. Colab 在 Drive 写入逐轮 `metrics.csv`、`status.json` 与实际权重。MF、LRURec 和 Mamba4Rec 保存验证最佳 `best.pt` 与可续训 `last.pt`。后者包括模型、优化器、已经完成的轮次、随机数状态和实际使用的调度器与早停状态。神经模型按完整 epoch 保存断点，恢复时按轮次去重日志。LightGBM 保存最佳迭代数与原生模型，已完成的配置直接复用，中断的配置按原种子重跑。首次真实试训核对保存、重载和中断处理。
3. 每次验证实验把小型摘要放进 GitHub `experiments/exp_XXX/`。`dynamics.json` 记录收敛、验证指标、资源开销，`diagnostics_panel.png` 展示曲线，`sliced_error_analysis.md` 展示短／长历史与复购等切片。适用时为神经模型写 `layer_health.json` 记录梯度和参数诊断。冻结的 Top-10、AUC 样本分数、权重和细粒度日志留在 Drive。
4. 组员用 GPT Work 或 Codex 读取验证摘要和实验账本。小样本阶段诊断实现、确定预算，再预登记正式配置。所有根据完整验证结果比较的搜索配置均计入预算并留痕。修复会改变结果的代码时标明受影响运行，在相同协议下重跑。结束调参、冻结配置后才打开测试标签。网页提醒为选用功能，训练脚本负责早停和保存。

### 12. 会影响成绩的代码检查

| 检查 | 小样本应通过的断言 |
|---|---|
| 时间与标签隔离 | 所有特征事件时间 `< history_end`。训练标签窗口结束不晚于训练截止。追加验证或测试交易，训练特征、候选及负采样不变。改变某训练月截点之后的交易，不应改变该截点之前的历史输入。 |
| 购物篮与复购 | 打乱同篮行后，聚合事件、序列及月度标签相同。覆盖同票多时间、多行不同单价，以及历史 `[A,B,A]` 且目标含 A 的复购例。把 A 的分数设为最高时，四条评价路径均须返回 A 为首位。 |
| 客户、候选与负例 | 四个模型按窗口得到相同 `query_id`、正标签哈希、商品顺序及候选哈希。训练负采样排除当月全部正例。分别统计无购买、无候选正例和冷启动。用九月首次出现的虚拟商品检查历史编码、完整候选保留和无训练支持参数的回退分数。 |
| 共用序列与月度标签 | 用长度 1、含复购、超过 `L` 的历史核对有效 ID、最后有效位置与截断。固定权重并关闭 dropout，比较同一条历史单条打分和放入不同补齐长度的批次的候选分数（声明容差）。整月买 `A,B,B` 产生 `{A,B}`，与损失的手算值一致。填充 ID 与目录外商品永远排除。 |
| 排名与指标 | 给定真实 `{B,D}`，预测 Top-10 为 `[A,B,C,E,F,G,H,I,J,K]`。手算 Recall@10 = 0.5，NDCG@10 ≈ 0.38685。另覆盖无命中、并列、重复商品和相关商品多于 10 个。 |
| 抽样 ROC AUC | 一个查询的正例分数为 0.8，两个负例为 0.8、0.2，AUC 应为 0.75。再加入一个正例分数为 0.1、负例分数为 0.3 的查询，两者的宏平均应为 0.375。检查单一类别查询的排除计数、四个模型的抽样清单哈希，以及训练种子变化时评估清单能否复用。 |
| 训练与恢复 | MF、LRURec、Mamba4Rec 在小批次能更新参数，两种序列模型编码器与评分层的梯度有限且有非零项。加载前后的分数在数值容差内。LightGBM 读回原生模型后迭代数与排序一致。 |
| 冻结与复现 | 每次运行的配置、种子、硬件和输入能追溯。测试入口检查四模型选择记录及协议，对照运行明细核对三种子的均值、标准差。 |

若看过测试后发现实质性实现错误，保留原运行并标记作废，记录更改内容和受影响的实验，在报告中披露重新评估过程。

<a id="submission"></a>

## 三、评分与提交清单

### 13. 老师的六项评分逐项验收

课程评分项和权重依据 `Group_Project_Tutorial_Recommender_Systems.pdf` 第 1、3 页，算法数量采用组员向老师确认的口径。本组将四个模型一同纳入评分所需的实验、解释与展示。最终 README 指向以下文件的实际路径。题目三页没有另列 AI 使用或披露细则，文中的 GPT Work、Codex 分工属于小组工作安排，课程另行发布的规则以课程通知为准。

| 评分项 | 权重 | 提交前必须能定位的证据 |
|---|---:|---|
| 数据探索与预处理 | 30% | 已有[异常证据工作簿](../results/transaction_exceptions.xlsx)、[逐步清洗审计](../results/cleaning_steps.csv)、[商品规则](../results/merchandise_rules.csv)、[主 notebook 的聚合核验](../notebooks/project2_colab.ipynb)和[训练期 EDA 图表](../results/training_eda/)。[数据协议](data_protocol.zh-CN.md)与 [D2 notebook](../notebooks/monthly_queries_and_evaluation.ipynb) 给出各截止点的查询、共享序列、月末标签和指标验算示例。 |
| 模型定义 | 10% | 说明 MF、LightGBM、LRURec 和 Mamba4Rec 的输入与评分方式。列出 LightGBM 分组／标签／特征／负例，分别标明两种序列模型保留的官方模块和改动。 |
| 模型训练 | 10% | 四套独立训练记录含随机种子、Colab 环境、耗时、早停、采样和截断配置，能定位权重或原生模型。 |
| 超参数调优 | 10% | `results/tuning.csv` 含尝试配置、验证两指标、早停轮次、预算和选择标准。 |
| 模型选择 | 10% | `results/selection.json` 与报告写明四个模型的冻结配置、验证集选出的最终模型及依据，均在测试前确定。 |
| 结果解释 | 30% | `results/test_metrics.csv` 并列报告四个模型的 NDCG@10、Recall@10，并以补充列报告查询宏平均的抽样 ROC AUC。`cohort_coverage.csv`、`segment_metrics.csv`、`case_study.csv` 支撑优缺点、失误及后续改进。 |

### 14. Canvas 提交与课堂展示

报告草稿记录 D1、D2 已测结果。完成模型试训和冻结评估后，再补入相应实验结果。已有的查询、覆盖统计和标签定义可以按数据协议复现。[技术报告草稿](technical_report.zh-CN.md) · [数据协议](data_protocol.zh-CN.md)

- [ ] **小组行政事项。** 每组最多 5 人，选出组长，按题目说明将成员名单提交至 `jie.wang@cityu-dg.edu.cn`。实际截止日以课程通知为准。
- [ ] **技术报告。** 依次说明任务与数据、清洗与聚合、时间与候选协议、四个模型的输入与输出、训练与调参、验证选模、最终测试、错误分析及结论。注明数据归属、DOI、许可和两套序列模型的上游版本与团队改动。
- [ ] **源码和 README。** 在 Canvas 提交技术报告、源代码、README 以及所采用且课程允许的演示材料。README 给出**运行说明、软件要求、随机种子、数据引用、结果表或图的位置**。另附数据获取方式、Drive 路径配置和复现实例，保证阅卷人能找到小型输出文件。
- [ ] **5 分钟课堂展示。** 建议 35 秒交代任务、数据及四个模型，65 秒讲清洗与时间切分，55 秒用一图解释四种打分方法，105 秒展示共同指标、两种序列模型的差异和一个具体失误，40 秒说明结论与改进。合计 300 秒。
- [ ] **核对四模型实验。** 主结果表各有 MF、LightGBM、LRURec、Mamba4Rec 一行，四行样本与候选哈希相同。给出验证选模依据、测试 NDCG@10、Recall@10、补充的抽样 ROC AUC、各模型耗时和相同客户的推荐例子。

### 15. 从 D2 数据协议到 MF 试训

使用 `data/processed/d2/manifest.json` 识别每次实验的数据、截止点、随机种子与产物版本。二月至九月为训练目标；十月验证与之后的十一月测试共用十月一日的历史。已生成 5,430 条训练查询和 758 条合格的十月验证查询。D2 notebook 逐步演示查询、统一序列和指标验算。[运行清单](../data/processed/d2/manifest.json) · [月度覆盖](../data/processed/d2/coverage.csv) · [月度查询与评价](../notebooks/monthly_queries_and_evaluation.ipynb) · [统一排名评价器](../src/evaluation/ranking.py)

下一项模型工作为 D3 的 MF 试训。用共用训练正负样本拟合，给验证集全部候选评分，核对保存和重载后的排名，记录代码、环境、种子与耗时。在序列模型试训前继续并行核验 D0 的 GPU 兼容性。待四个模型的配置冻结后再解封测试标签及评分。

<a id="sources"></a>

## 材料与来源

- **课程项目题目**　`Group_Project_Tutorial_Recommender_Systems.pdf`，PDF 第 1 页为提交、展示与两项目评分表，第 2 页列出近期算法及作者代码，第 3 页为项目二的完整要求。`Lecture_7.pdf` 第 34–35 页给出课堂矩阵分解和嵌入点积评分。本计划对项目二使用隐式购买反馈。
- **数据**　Chen, D. (2015). *Online Retail*. UCI Machine Learning Repository，CC BY 4.0。原始数据审计基于上述 SHA-256 对应的课程上传工作簿。[UCI 数据与署名](https://archive.ics.uci.edu/dataset/352/online+retail) · [DOI](https://doi.org/10.24432/C5BW33)
- **LightGBM**　Ke 等（2017），*LightGBM: A Highly Efficient Gradient Boosting Decision Tree*。[原论文](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) · [排序 API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)
- **LRURec**　*Linear Recurrent Units for Sequential Recommendation*，WSDM 2024。复现记录应区分官方编码器和本项目的未来整月任务适配。[论文](https://doi.org/10.1145/3616855.3635760) · [作者仓库](https://github.com/yueqirex/LRURec)
- **Mamba4Rec**　*Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models*（2024）。官方 README 列出运行依赖，`mamba4rec.py` 提供有效状态读取、单目标训练与商品打分代码。[论文](https://arxiv.org/abs/2403.03900) · [作者仓库](https://github.com/chengkai-liu/Mamba4Rec) · [模型代码](https://github.com/chengkai-liu/Mamba4Rec/blob/44a0c8432b34e3b89d46035b10e8525b1bd368dc/mamba4rec.py)
- **Colab**　数据和运行输出使用挂载的 Drive 路径，README 记录实际环境及复现方式。[Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
