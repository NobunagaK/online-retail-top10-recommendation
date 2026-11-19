"""Frozen merchandise decisions for the UCI Online Retail workbook.

Review window: 2010-12-01 (inclusive) through 2011-10-01 (exclusive). The
review covered unusual stock codes and descriptions that could denote fees,
services, discounts, adjustments or vouchers. Runtime matching uses exact
StockCode values only. Unlisted codes keep their eligibility under this rule.
Quantities, prices, descriptions and code shapes are never used as proxies
for merchandise status at application time.

The two counts in each evidence record refer to the project workbook:
training_rows includes all raw rows in the window; training_basic_rows counts
lines after de-duplication on the eight original columns, requiring a customer,
a non-C invoice, positive quantity and positive unit price. These audit counts
are descriptive and do not enter recommendation features. Worksheet row 1
contains the column headers.
"""

import pandas as pd


# Policy names group reasons shared by literal codes. Their spelling has no
# special meaning when matching a StockCode in the workbook.
_POLICY = {
    "delivery": (
        "exclude", "Postage or carriage pays for delivery.",
        "邮费或运费支付的是配送服务。",
    ),
    "packing_fee": (
        "exclude", "The line bills a packing fee.",
        "该行收取包装费。",
    ),
    "financial_fee": (
        "exclude", "The description identifies a payment, platform or commission fee.",
        "描述表明该行记录银行、平台费用或佣金。",
    ),
    "bad_debt": (
        "exclude", "The line records a bad-debt adjustment.",
        "该行属于坏账调整。",
    ),
    "discount": (
        "exclude", "The line records a discount on a transaction.",
        "该行记录交易折扣。",
    ),
    "manual": (
        "exclude", "The source description only says Manual. It identifies no particular product to rank.",
        "原文描述只有 Manual，无法据此识别可供推荐的具体商品。",
    ),
    "sample": (
        "exclude", "The generic sample code does not identify a particular item to recommend.",
        "样品通用编码未指明可推荐的具体商品。",
    ),
    "voucher": (
        "exclude", "The denomination identifies prepaid store credit. This project ranks merchandise.",
        "面额对应店铺预付购物额度；本项目的推荐对象是商品。",
    ),
    "product": (
        "retain", "The training description names a physical product.",
        "训练期描述指明了具体实物商品。",
    ),
    "unresolved": (
        "retain", "Training rows contain only a missing description or a channel name. "
                  "Evidence is insufficient for a code-wide exclusion.",
        "训练期描述缺失或只记录销售渠道，没有足够证据对整个商品编码设定排除规则。",
    ),
}

# Columns: exact code, policy, literal example description, Excel row, time,
# all raw training rows, and training rows surviving the basic purchase filters.
_TRAINING_EVIDENCE = (
    ("23444", "delivery", "Next Day Carriage", 205750, "2011-05-26 16:11", 38, 36),
    ("23574", "packing_fee", "PACKING CHARGE", 252607, "2011-07-06 13:04", 9, 7),
    ("AMAZONFEE", "financial_fee", "AMAZON FEE", 14516, "2010-12-07 12:41", 27, 0),
    ("B", "bad_debt", "Adjust bad debt", 299984, "2011-08-12 14:50", 3, 0),
    ("BANK CHARGES", "financial_fee", "Bank Charges", 4408, "2010-12-02 15:08", 27, 7),
    ("C2", "delivery", "CARRIAGE", 1425, "2010-12-01 14:05", 102, 94),
    ("CRUK", "financial_fee", "CRUK Commission", 317510, "2011-08-30 10:49", 6, 0),
    ("D", "discount", "Discount", 143, "2010-12-01 09:41", 61, 0),
    ("DOT", "delivery", "DOTCOM POSTAGE", 1816, "2010-12-01 14:32", 603, 6),
    ("M", "manual", "Manual", 2241, "2010-12-01 15:35", 411, 188),
    ("m", "manual", "Manual", 40385, "2010-12-21 15:18", 1, 0),
    ("POST", "delivery", "POSTAGE", 47, "2010-12-01 08:45", 872, 770),
    ("S", "sample", "SAMPLES", 14438, "2010-12-07 12:03", 55, 0),
    ("gift_0001_10", "voucher", "Dotcomgiftshop Gift Voucher £10.00", 112444, "2011-03-07 17:14", 8, 0),
    ("gift_0001_20", "voucher", "Dotcomgiftshop Gift Voucher £20.00", 44796, "2011-01-05 14:44", 9, 0),
    ("gift_0001_30", "voucher", "Dotcomgiftshop Gift Voucher £30.00", 44727, "2011-01-05 14:44", 8, 0),
    ("gift_0001_40", "voucher", "Dotcomgiftshop Gift Voucher £40.00", 38250, "2010-12-20 10:14", 3, 0),
    ("gift_0001_50", "voucher", "Dotcomgiftshop Gift Voucher £50.00", 42059, "2010-12-23 13:26", 4, 0),
    ("21518", "product", "BANK ACCOUNT  GREETING  CARD ", 3343, "2010-12-02 10:39", 39, 17),
    ("23099", "product", "FRENCH CARRIAGE LANTERN", 191482, "2011-05-16 14:30", 42, 39),
    ("85123A", "product", "WHITE HANGING HEART T-LIGHT HOLDER", 2, "2010-12-01 08:26", 1813, 1581),
    ("PADS", "product", "PADS TO MATCH ALL CUSHIONS", 157197, "2011-04-15 09:27", 4, 3),
    ("DCGS0003", "product", "BOXED GLASS ASHTRAY", 24908, "2010-12-10 14:59", 5, 0),
    ("DCGS0004", "product", "HAYNES CAMPER SHOULDER BAG", 170785, "2011-04-27 17:23", 1, 0),
    ("DCGS0055", "unresolved", None, 74827, "2011-01-28 13:08", 1, 0),
    ("DCGS0057", "unresolved", None, 75297, "2011-01-28 15:01", 1, 0),
    ("DCGS0066P", "unresolved", None, 279259, "2011-07-26 11:52", 1, 0),
    ("DCGS0067", "unresolved", "ebay", 279258, "2011-07-26 11:52", 1, 0),
    ("DCGS0068", "unresolved", "ebay", 279257, "2011-07-26 11:52", 1, 0),
    ("DCGS0069", "product", "OOH LA LA DOGS COLLAR", 75055, "2011-01-28 14:25", 2, 0),
    ("DCGS0070", "product", "CAMOUFLAGE DOG COLLAR", 40054, "2010-12-21 13:06", 2, 0),
    ("DCGS0071", "unresolved", None, 279254, "2011-07-26 11:51", 1, 0),
    ("DCGS0072", "unresolved", None, 74840, "2011-01-28 13:08", 1, 0),
    ("DCGS0073", "unresolved", "ebay", 279253, "2011-07-26 11:51", 1, 0),
    ("DCGS0074", "unresolved", None, 74841, "2011-01-28 13:09", 1, 0),
    ("DCGS0076", "product", "SUNJAR LED NIGHT NIGHT LIGHT", 21328, "2010-12-09 14:09", 2, 0),
    ("DCGSSBOY", "product", "BOYS PARTY BAG", 84018, "2011-02-07 14:04", 10, 0),
    ("DCGSSGIRL", "product", "GIRLS PARTY BAG", 84019, "2011-02-07 14:04", 11, 0),
)

def merchandise_rules() -> pd.DataFrame:
    """Return the frozen exclusion decisions and reviewed retained codes with evidence."""
    return pd.DataFrame.from_records(
        [
            (code, *_POLICY[policy], *evidence)
            for code, policy, *evidence in _TRAINING_EVIDENCE
        ],
        columns=[
            "StockCode", "action", "reason_en", "reason_zh", "training_description",
            "training_example_excel_row", "training_example_date", "training_rows",
            "training_basic_rows",
        ],
    )
