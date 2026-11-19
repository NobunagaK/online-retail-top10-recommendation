import pandas as pd
import pytest

from src.data.history_features import history_features
from src.data.sequences import build_histories, build_item_vocab, prepare_sequence


def event(customer, invoice, sku, time, quantity=1, amount=2.5):
    return dict(CustomerID=customer, InvoiceNo=invoice, StockCode=sku,
                basket_time=pd.Timestamp(time), quantity=quantity, amount_gbp=amount,
                source_lines=1)


def test_vocab_is_append_only_across_snapshot_replays_and_never_uses_evaluation_labels():
    feb, mar, oct = (pd.Timestamp(d) for d in ("2011-02-01", "2011-03-01", "2011-10-01"))
    # An earlier retained purchase of 00B disappears in the later replay.
    a = event("1", "12", "00B", "2010-12-15")
    b = event("1", "11", "00A", "2010-12-15")
    c = event("3", "22", "b", "2011-02-01")
    d = event("3", "23", "a", "2011-02-01")
    e = event("3", "25", "eval-only", "2011-10-03")
    snapshots = {feb: pd.DataFrame([a, b, c]), mar: pd.DataFrame([b, c, d]), oct: pd.DataFrame([b, c, d])}
    vocab = build_item_vocab(dict(reversed(list(snapshots.items()))))
    assert list(vocab[["stock_code", "item_id"]].itertuples(index=False, name=None)) == [
        ("00A", 1), ("00B", 2), ("a", 3), ("b", 4),
    ]
    assert vocab.loc[vocab.stock_code.eq("00B"), "first_history_end"].item() == feb
    history = build_histories(snapshots[feb], feb, vocab).iloc[0]
    assert history.history_item_ids == [1, 2]
    later = dict(snapshots, **{"2011-11-01": pd.DataFrame([e])})
    pd.testing.assert_frame_equal(vocab, build_item_vocab(later))
    pd.testing.assert_frame_equal(vocab.iloc[:2], build_item_vocab({feb: snapshots[feb]}))


def test_history_uses_chronological_invoice_and_sku_order_keeps_repeats_and_basket_boundaries():
    end = pd.Timestamp("2011-10-01")
    events = pd.DataFrame([
        event("c", "02", "A", "2011-09-18"), event("c", "01", "B", "2011-09-18"),
        event("c", "01", "A", "2011-09-18"), event("c", "03", "b", "2011-09-19"),
        event("other", "01", "A", "2011-09-18"), event("c", "04", "X", "2011-10-01"),
    ])
    vocab = build_item_vocab({end: events})
    shuffled = events.sample(frac=1, random_state=25)
    history = build_histories(shuffled, end, vocab, max_length=3, customer_ids=["absent"]).set_index("customer_id")
    assert history.loc["c", "history_item_ids"] == [1, 2, 1, 3]
    assert history.loc["c", "history_basket_sizes"] == [2, 1, 1]
    assert history.loc["c", "history_len"] == 4 and history.loc["c", "seq_len"] == 3
    assert history.loc["c", "is_truncated"] and history.loc["c", "cuts_basket"]
    assert history.loc["absent", "history_item_ids"] == [] and history.loc["absent", "history_len"] == 0
    assert not build_histories(shuffled, end, vocab, max_length=2).iloc[0].cuts_basket
    pd.testing.assert_frame_equal(history, build_histories(events, end, vocab, 3, customer_ids=["absent"]).set_index("customer_id"))


@pytest.mark.parametrize("ids,L,expected", [
    ([5], 4, [5]), ([8, 7, 8], 4, [8, 7, 8]), ([6, 7, 8, 7], 3, [7, 8, 7]), ([], 2, []),
])
def test_models_keep_same_real_ids_and_read_the_same_last_valid_item(ids, L, expected):
    left, left_len, left_last = prepare_sequence(ids, L, padding="left")
    right, right_len, right_last = prepare_sequence(ids, L, padding="right")
    assert left == [0] * (L - len(expected)) + expected
    assert right == expected + [0] * (L - len(expected))
    assert left_len == right_len == len(expected)
    if expected:
        assert left[left_last] == right[right_last] == expected[-1]
    else:
        assert left_last is None and right_last is None


def test_reserved_zero_and_invalid_model_sequence_inputs():
    with pytest.raises(ValueError):
        prepare_sequence([3, 0, 2])
    with pytest.raises(ValueError):
        prepare_sequence([1], max_length=0)
    with pytest.raises(ValueError):
        prepare_sequence([1], padding="both")


def test_recency_windows_and_all_historical_features_use_the_fixed_cutoff():
    end = pd.Timestamp("2011-10-01")
    rows = [
        event("1", "01", "A", end - pd.Timedelta(days=90, seconds=1)),
        event("1", "02", "A", end - pd.Timedelta(days=90)),
        event("1", "03", "A", end - pd.Timedelta(days=30)),
        event("1", "04", "A", end - pd.Timedelta(days=7)),
        event("1", "04", "B", end - pd.Timedelta(days=7)),
        event("2", "04", "A", end - pd.Timedelta(hours=1)),
        event("1", "05", "A", end - pd.Timedelta(hours=12), quantity=3, amount=10),
    ]
    events = pd.DataFrame(rows)
    # The November query uses the exact same historical cutoff as the October
    # query, even after target-period transactions and credits have been observed.
    future = pd.concat([events, pd.DataFrame([event("1", "06", "A", end),
                                               event("1", "07", "X", "2011-10-31")])])
    october = history_features(events, end, customer_ids=["empty"])
    november = history_features(future, end, customer_ids=["empty"])
    for key in october:
        pd.testing.assert_frame_equal(october[key], november[key])
    cust = october["customer_features"].set_index("customer_id")
    pair = october["customer_item_features"].set_index(["customer_id", "stock_code"])
    for row in [cust.loc["1"], pair.loc[("1", "A")]]:
        assert row[["invoices_7d", "invoices_30d", "invoices_90d"]].tolist() == [2, 3, 4]
        assert row.days_since_last_purchase == 0.5
    assert cust.loc["1", "event_count"] == 6 and cust.loc["1", "invoice_count"] == 5
    assert pair.loc[("1", "A"), "quantity"] == 7 and pair.loc[("1", "A"), "amount_gbp"] == 20
    sku = october["item_features"].set_index("stock_code").loc["A"]
    assert sku.invoice_count == 6 and sku.distinct_customers == 2  # two customers share invoice 04
    assert sku.days_since_last_purchase == pytest.approx(1/24)
    assert cust.loc["empty", "invoice_count"] == 0
    assert pd.isna(cust.loc["empty", "days_since_last_purchase"])


def test_no_prior_events_produce_empty_tables_with_a_reusable_empty_history():
    future = pd.DataFrame([event("c", "01", "A", "2011-11-01")])
    end = "2011-10-01"
    vocab = build_item_vocab({end: future})
    assert vocab.empty
    history = build_histories(future, end, vocab, customer_ids=["01"])
    assert history.iloc[0].history_len == 0
    feats = history_features(future, end, customer_ids=["01"])
    assert feats["item_features"].empty and feats["customer_item_features"].empty
    assert feats["customer_features"].iloc[0].customer_id == "01"
