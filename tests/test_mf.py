from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.mf import make_model


@pytest.fixture
def data():
    queries = pd.DataFrame({"query_id": ["q1", "q2"], "customer_id": ["A", "B"]})
    pairs = pd.DataFrame([
        ("q1", 1, 1),
        ("q2", 2, 1), ("q2", 3, 0), ("q2", 4, 0), ("q2", 5, 0),
    ], columns=["query_id", "item_id", "label"])
    vocab = pd.DataFrame({"stock_code": list("abcdefg"), "item_id": range(1, 8)})
    return SimpleNamespace(train_queries=queries, train_pairs=pairs, vocab=vocab)


@pytest.mark.parametrize("l2", [0, 0.16])
def test_loss_is_mean_per_query_including_sampled_l2(data, l2):
    model = make_model(data, {"embedding_dim": 2, "l2": l2}, "cpu")
    with torch.no_grad():
        for parameter in model.factors.parameters():
            parameter.zero_()
        model.factors.user_bias.weight[:, 0] = torch.tensor([2.0, -1.0])
    # q1 has one positive, q2 has one positive and three negatives.
    loss_q1 = np.logaddexp(0, -2.0) + 0.5 * l2 * 4
    loss_q2 = (np.logaddexp(0, 1.0) + 3 * np.logaddexp(0, -1.0)) / 4 + 0.5 * l2
    assert model.train_epoch(data) == pytest.approx((loss_q1 + loss_q2) / 2)


def test_unsupported_embeddings_and_biases_use_only_supported_rows(data):
    model = make_model(data, {"embedding_dim": 2}, "cpu")
    assert model.supported_item_ids == set()
    model.train_epoch(data)
    assert model.supported_item_ids == {1, 2, 3, 4, 5}
    with torch.no_grad():
        # Corrupt PAD and the two validation-catalog-only rows to expose accidental use.
        factors = model.factors
        factors.user.weight[0] = torch.tensor([2.0, -1.0])
        factors.user_bias.weight[0] = 0.5
        factors.item.weight.fill_(999)
        factors.item_bias.weight.fill_(-500)
        factors.item.weight[1:6] = torch.tensor([[1, 2], [2, 0], [-1, 3], [0, 2], [4, 3]])
        factors.item_bias.weight[1:6, 0] = torch.tensor([2, 0, -3, 5, 1])
    query = SimpleNamespace(query_id="validation_A", customer_id="A")
    # Item 1 was a purchase and stays eligible. Item 4 was negative-only and learned a row.
    supported = np.array([2.5, 4.5, -7.5, 3.5, 6.5])
    wanted = [supported.mean(), supported[3], supported[0], supported.mean(), supported[0]]
    actual = model.score(query, [7, 4, 1, 6, 1])
    np.testing.assert_allclose(actual, wanted, atol=1e-6)
    assert actual[0] == actual[3]
    with pytest.raises(ValueError, match="positive D2 item IDs"):
        model.score(query, [0, 1])


def test_updates_and_resume_reproduce_next_training_epoch(data, tmp_path):
    config = {"seed": 2026, "embedding_dim": 4, "batch_size": 2, "learning_rate": 0.01}
    query = data.train_queries.iloc[0]
    catalog = list(range(1, 8))
    model = make_model(data, config, "cpu")
    original = model.factors.item.weight.clone()
    first_loss = model.train_epoch(data)
    assert np.isfinite(first_loss) and model.completed_epochs == 1
    assert not torch.equal(original[1:6], model.factors.item.weight[1:6])
    assert torch.equal(original[6:], model.factors.item.weight[6:])
    first_scores = model.score(query, catalog)
    path = tmp_path / "run" / "last.pt"
    model.save(path)

    fresh = make_model(data, config, "cpu")
    assert fresh.train_epoch(data) == first_loss
    np.testing.assert_array_equal(fresh.score(query, catalog), first_scores)
    resumed = make_model(data, dict(config, seed=3407), "cpu")
    resumed.restore(path)
    assert resumed.completed_epochs == 1
    assert resumed.supported_item_ids == model.supported_item_ids
    np.testing.assert_array_equal(resumed.score(query, catalog), first_scores)
    assert resumed.train_epoch(data) == model.train_epoch(data)
    np.testing.assert_array_equal(resumed.score(query, catalog), model.score(query, catalog))
    assert resumed.completed_epochs == 2
    assert all(torch.equal(p, q) for p, q in zip(resumed.factors.parameters(), model.factors.parameters()))


def test_resume_detects_changed_sampled_negatives(data, tmp_path):
    original = make_model(data, {}, "cpu")
    path = tmp_path / "last.pt"
    original.save(path)
    data.train_pairs.loc[data.train_pairs["item_id"] == 5, "item_id"] = 7
    changed = make_model(data, {}, "cpu")
    with pytest.raises(ValueError, match="selected D2 training pairs"):
        changed.restore(path)
