"""Compare October candidate chunks to the complete pinned LRURec forward with the same weights."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import digest, write_json
from src.data.monthly_queries import ordered_hash
from src.data.sequences import prepare_sequence
from src.models.lrurec_adapter import UPSTREAM_COMMIT, UPSTREAM_SHA256, make_model
from src.models.pilot_data import PilotData


@torch.no_grad()
def check_scores(data_dir, run_dir, chunk_size):
    run = json.loads((run_dir / "run.json").read_text())
    checkpoint = run_dir / "best.pt"
    if run["model"] != "lrurec" or run["status"] != "complete" or run["split"] != "validation":
        raise ValueError("Supply a completed LRURec October validation run.")
    if (run["upstream_model"]["commit"] != UPSTREAM_COMMIT or run["upstream_model"]["source_sha256"] != UPSTREAM_SHA256
            or digest(checkpoint) != run["output_sha256"]["best.pt"]):
        raise ValueError("The pinned implementation or checkpoint differs from the training run.")
    data = PilotData(data_dir, smoke=run["smoke"])
    if (digest(data_dir / "manifest.json") != run["manifest_sha256"]
            or ordered_hash(data.train_queries.query_id) != run["training_queries_sha256"]
            or ordered_hash(data.val_queries.query_id) != run["validation_queries_sha256"]):
        raise ValueError("D2 data and queries must match the completed run.")

    torch.set_num_threads(run["threads"])
    runner = make_model(data, run["config"], device="cpu").restore(checkpoint)
    reference = deepcopy(runner.network.upstream).eval()
    # Independently reconstruct rows actually seen in the truncated training inputs and full-catalog losses.
    input_ids = {int(item) for q in data.train_queries.query_id for item in data.histories_by_query[q][-runner.max_length:]}
    loss_ids = {item for c in data.train_queries.catalog_id.unique() for item in data.catalog_ids[c]}
    if (input_ids != set(runner.input_support.nonzero().flatten().tolist())
            or loss_ids != set(runner.catalog_support.nonzero().flatten().tolist())):
        raise ValueError("D2 training support differs from the checkpoint.")
    real_ids = set(data.vocab.item_id)
    embedding_fallback = real_ids - (input_ids | loss_ids)
    bias_fallback = real_ids - loss_ids
    weights, bias = reference.embedding.token.weight, reference.model.bias
    weights[sorted(embedding_fallback)] = weights[sorted(input_ids | loss_ids)].mean(0)
    bias[sorted(bias_fallback)] = bias[sorted(loss_ids)].mean()

    # All October queries share one catalog sorted by exact StockCode, which breaks score ties.
    catalog_key, = data.val_queries.catalog_id.unique()
    candidates = data.catalog_ids[catalog_key]
    encoder_calls = 0

    def count_forward(_module, _inputs, _output):
        nonlocal encoder_calls
        encoder_calls += 1

    max_error = 0.0
    identical_rankings = 0
    atol, rtol = 2e-6, 2e-5
    with runner.network.upstream.embedding.register_forward_hook(count_forward):
        for query in data.val_queries.itertuples(index=False):
            padded, length, _ = prepare_sequence(data.histories_by_query[query.query_id], runner.max_length, "left")
            if length == 0:
                raise ValueError("Expected an eligible October query with purchase history.")
            # This calls the authors' complete prediction path over every sequence position and vocabulary row.
            full_scores, _ = reference(torch.tensor([padded]))
            expected = full_scores[0, -1, candidates].numpy()
            before = encoder_calls
            chunks = [runner.score(query, candidates[i:i + chunk_size]) for i in range(0, len(candidates), chunk_size)]
            if encoder_calls - before != 1:
                raise AssertionError(f"{query.query_id}: each history must be encoded exactly once.")
            actual = np.concatenate(chunks)
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=query.query_id)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            identical_rankings += int(np.array_equal(
                np.argsort(-actual, kind="stable")[:10], np.argsort(-expected, kind="stable")[:10]))

    n_queries = len(data.val_queries)
    return {
        "status": "passed", "comparison": "candidate chunks vs authors' complete LRU.forward at final position",
        "model_commit": run["git_commit"], "upstream_model": run["upstream_model"],
        "checkpoint_sha256": digest(checkpoint), "manifest_sha256": run["manifest_sha256"],
        "validation_queries_sha256": run["validation_queries_sha256"], "ordered_catalog_ids_sha256": ordered_hash(candidates),
        "device": "cpu", "torch": str(torch.__version__), "training_queries": len(data.train_queries),
        "queries": n_queries, "candidates_per_query": len(candidates), "scores_compared": n_queries * len(candidates),
        "chunk_size": chunk_size, "encoder_calls": encoder_calls, "encoder_calls_per_query": 1,
        "training_history_items": len(input_ids), "training_loss_items": len(loss_ids),
        "candidate_embedding_fallback_rows": len(embedding_fallback.intersection(candidates)),
        "candidate_bias_fallback_rows": len(bias_fallback.intersection(candidates)),
        "atol": atol, "rtol": rtol, "max_absolute_score_error": max_error,
        "identical_ordered_top10_queries": identical_rankings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/d2"))
    parser.add_argument("--run-dir", type=Path, required=True, help="Pilot directory containing run.json and best.pt.")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON evidence.")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    result = check_scores(args.data_dir, args.run_dir, args.chunk_size)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
