#!/usr/bin/env python3
"""Deduplicated full-embedding reversion inference for the trained model."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.serum_gate.train_serum_mutation_set import (  # noqa: E402
    align_embedding_to_sequence,
    load_ha_distance_matrix,
    normalize_passage,
)
from src.fluprofiler.models.serum_mutation_set_model import (  # noqa: E402
    SerumMutationSetBatch,
    SerumMutationSetMinusModel,
)


class EmbeddingCache:
    def __init__(self, directory: Path, max_items: int) -> None:
        self.directory = directory
        self.max_items = max_items
        self.values: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, sequence_id: str) -> torch.Tensor:
        value = self.values.pop(sequence_id, None)
        if value is None:
            value = torch.as_tensor(
                torch.load(self.directory / f"matrix_{sequence_id}.pt", map_location="cpu", weights_only=False)
            ).float()
        self.values[sequence_id] = value
        if len(self.values) > self.max_items:
            self.values.popitem(last=False)
        return value


def reversion_embedding_id(sequence: str) -> str:
    digest = hashlib.sha256(str(sequence).replace("-", "").encode("utf-8")).hexdigest()
    return f"REV_HA_{digest}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--original-pairs-csv", type=Path, required=True)
    parser.add_argument("--reverted-counterfactuals-csv", type=Path, required=True)
    parser.add_argument("--original-embedding-dir", type=Path, required=True)
    parser.add_argument("--reversion-embedding-dir", type=Path, required=True)
    parser.add_argument("--distance-matrix", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> tuple[SerumMutationSetMinusModel, dict[str, Any]]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SerumMutationSetMinusModel(
        checkpoint["model_config"], load_ha_distance_matrix(args.distance_matrix)
    ).to(args.device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def make_batch(
    frame: pd.DataFrame,
    model: SerumMutationSetMinusModel,
    passage_to_id: dict[str, int],
    original_cache: EmbeddingCache,
    reversion_cache: EmbeddingCache | None,
    device: str,
) -> SerumMutationSetBatch:
    references = [
        align_embedding_to_sequence(original_cache.get(str(row.seq_id_a)), str(row.reference_sequence))
        for row in frame.itertuples(index=False)
    ]
    if reversion_cache is None:
        queries = [
            align_embedding_to_sequence(original_cache.get(str(row.seq_id_c)), str(row.query_sequence))
            for row in frame.itertuples(index=False)
        ]
    else:
        queries = [
            align_embedding_to_sequence(
                reversion_cache.get(reversion_embedding_id(str(row.reverted_query_sequence))),
                str(row.reverted_query_sequence),
            )
            for row in frame.itertuples(index=False)
        ]
    serum_passage = torch.tensor(
        [passage_to_id[normalize_passage(value)] for value in frame.serumPassCat], device=device
    )
    query_passage = torch.tensor(
        [passage_to_id[normalize_passage(value)] for value in frame.virusPassCat], device=device
    ).view(-1, 1)
    return SerumMutationSetBatch(
        reference_ha=torch.stack([item[0] for item in references]).to(device),
        query_ha=torch.stack([item[0] for item in queries]).to(device)[:, None],
        reference_aa=torch.stack([item[3] for item in references]).to(device),
        query_aa=torch.stack([item[3] for item in queries]).to(device)[:, None],
        reference_aligned_mask=torch.stack([item[1] for item in references]).to(device),
        query_aligned_mask=torch.stack([item[1] for item in queries]).to(device)[:, None],
        reference_embedding_mask=torch.stack([item[2] for item in references]).to(device),
        query_embedding_mask=torch.stack([item[2] for item in queries]).to(device)[:, None],
        serum_passage=serum_passage,
        query_passage=query_passage,
        passage_pair=serum_passage[:, None] * model.config.passage_vocab_size + query_passage,
        subtype=torch.zeros(len(frame), dtype=torch.long, device=device),
        query_mask=torch.ones(len(frame), 1, device=device),
    )


def predict_contexts(
    contexts: pd.DataFrame,
    context_id: str,
    model: SerumMutationSetMinusModel,
    passage_to_id: dict[str, int],
    original_cache: EmbeddingCache,
    reversion_cache: EmbeddingCache | None,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    with torch.inference_mode():
        for start in range(0, len(contexts), args.batch_size):
            frame = contexts.iloc[start : start + args.batch_size]
            output = model(make_batch(frame, model, passage_to_id, original_cache, reversion_cache, args.device))
            rows.append(pd.DataFrame({
                context_id: frame[context_id].to_numpy(),
                "mean": output["mean"][:, 0].cpu().numpy(),
                "self_score": output["self_score"][:, 0].cpu().numpy(),
                "query_score": output["query_score"][:, 0].cpu().numpy(),
            }))
            print(f"{context_id}: {min(start + args.batch_size, len(contexts)):,} / {len(contexts):,}", flush=True)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    model, checkpoint = load_model(args)
    passage_to_id = {normalize_passage(key): int(value) for key, value in checkpoint["passage_to_id"].items()}
    pairs = pd.read_csv(args.original_pairs_csv)
    counterfactuals = pd.read_csv(args.reverted_counterfactuals_csv)
    original_cache = EmbeddingCache(args.original_embedding_dir, max_items=256)
    reversion_cache = EmbeddingCache(args.reversion_embedding_dir, max_items=64)

    original_keys = ["seq_id_a", "seq_id_c", "reference_sequence", "query_sequence", "serumPassCat", "virusPassCat"]
    original_contexts = pairs[original_keys].drop_duplicates().sort_values(["seq_id_a", "seq_id_c"], kind="stable").reset_index(drop=True)
    original_contexts.insert(0, "original_context_id", range(len(original_contexts)))
    pair_contexts = pairs[["sample_index", *original_keys]].merge(original_contexts, on=original_keys, how="left")
    original_predictions = predict_contexts(
        original_contexts, "original_context_id", model, passage_to_id, original_cache, None, args
    ).rename(columns={
        "mean": "original_mean", "self_score": "original_self_score", "query_score": "original_query_score"
    })

    counterfactuals = counterfactuals.merge(pair_contexts[["sample_index", "original_context_id", "seq_id_a", "serumPassCat", "virusPassCat", "reference_sequence"]], on="sample_index", how="left")
    reversion_keys = ["seq_id_a", "reference_sequence", "reverted_query_sequence", "serumPassCat", "virusPassCat"]
    reversion_contexts = counterfactuals[reversion_keys].drop_duplicates().sort_values("reverted_query_sequence", kind="stable").reset_index(drop=True)
    reversion_contexts.insert(0, "reversion_context_id", range(len(reversion_contexts)))
    counterfactuals = counterfactuals.merge(reversion_contexts, on=reversion_keys, how="left")
    reversion_predictions = predict_contexts(
        reversion_contexts, "reversion_context_id", model, passage_to_id, original_cache, reversion_cache, args
    ).rename(columns={
        "mean": "reverted_mean", "self_score": "reverted_self_score", "query_score": "reverted_query_score"
    })

    result = (
        counterfactuals
        .merge(original_predictions, on="original_context_id", how="left")
        .merge(reversion_predictions, on="reversion_context_id", how="left")
        .assign(
            delta_mean=lambda table: table["original_mean"] - table["reverted_mean"],
            delta_query_score=lambda table: table["original_query_score"] - table["reverted_query_score"],
            reverted_embedding_id=lambda table: table["reverted_query_sequence"].map(reversion_embedding_id),
        )
        .sort_values("counterfactual_id", kind="stable")
    )
    fields = [
        "counterfactual_id", "sample_index", "aa_position", "reference_aa", "query_aa", "substitution", "mutation_count",
        "original_mean", "reverted_mean", "delta_mean", "original_self_score", "reverted_self_score",
        "original_query_score", "reverted_query_score", "delta_query_score", "reverted_embedding_id",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result[fields].to_csv(args.output_csv, index=False)
    print({
        "original_contexts": len(original_contexts),
        "reversion_contexts": len(reversion_contexts),
        "counterfactual_rows": len(result),
        "output_csv": str(args.output_csv),
    })


if __name__ == "__main__":
    main()
