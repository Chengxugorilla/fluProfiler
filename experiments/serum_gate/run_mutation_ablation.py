#!/usr/bin/env python3
"""Export per-mutation counterfactual predictions for SerumMutationSetMinusModel.

For every model-defined mutation token, the counterfactual changes only the
query amino-acid ID at that position to the reference amino-acid ID. The model
therefore removes that token while retaining the reference background and every
other mutation in the pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import fields
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
    def __init__(self, folder: Path, max_items: int = 2048) -> None:
        self.folder = folder
        self.max_items = max_items
        self.values: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, sequence_id: str) -> torch.Tensor:
        key = f"matrix_{sequence_id}"
        value = self.values.pop(key, None)
        if value is None:
            value = torch.as_tensor(
                torch.load(self.folder / f"{key}.pt", map_location="cpu", weights_only=False)
            ).float()
        self.values[key] = value
        if len(self.values) > self.max_items:
            self.values.popitem(last=False)
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--distance-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ablation-batch-size", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> tuple[SerumMutationSetMinusModel, dict[str, Any]]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SerumMutationSetMinusModel(
        checkpoint["model_config"],
        load_ha_distance_matrix(args.distance_matrix),
    ).to(args.device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def make_batch_builder(
    model: SerumMutationSetMinusModel,
    checkpoint: dict[str, Any],
    embedding_dir: Path,
    device: str,
):
    cache = EmbeddingCache(embedding_dir)
    passage_to_id = {
        normalize_passage(key): int(value)
        for key, value in checkpoint["passage_to_id"].items()
    }

    def aligned_input(sequence_id: str, aligned_sequence: str):
        return align_embedding_to_sequence(cache.get(str(sequence_id)), str(aligned_sequence))

    def make_batch(frame: pd.DataFrame) -> SerumMutationSetBatch:
        reference = [aligned_input(row.seq_id_a, row.serumHA) for row in frame.itertuples(index=False)]
        query = [aligned_input(row.seq_id_c, row.virusHA) for row in frame.itertuples(index=False)]
        serum_passage = torch.tensor(
            [passage_to_id[normalize_passage(value)] for value in frame.serumPassCat],
            device=device,
        )
        query_passage = torch.tensor(
            [passage_to_id[normalize_passage(value)] for value in frame.virusPassCat],
            device=device,
        ).view(-1, 1)
        return SerumMutationSetBatch(
            reference_ha=torch.stack([item[0] for item in reference]).to(device),
            query_ha=torch.stack([item[0] for item in query]).to(device)[:, None],
            reference_aa=torch.stack([item[3] for item in reference]).to(device),
            query_aa=torch.stack([item[3] for item in query]).to(device)[:, None],
            reference_aligned_mask=torch.stack([item[1] for item in reference]).to(device),
            query_aligned_mask=torch.stack([item[1] for item in query]).to(device)[:, None],
            reference_embedding_mask=torch.stack([item[2] for item in reference]).to(device),
            query_embedding_mask=torch.stack([item[2] for item in query]).to(device)[:, None],
            serum_passage=serum_passage,
            query_passage=query_passage,
            passage_pair=serum_passage[:, None] * model.config.passage_vocab_size + query_passage,
            subtype=torch.zeros(len(frame), dtype=torch.long, device=device),
            query_mask=torch.ones(len(frame), 1, device=device),
        )

    return make_batch


def select_batch(batch: SerumMutationSetBatch, indices: torch.Tensor) -> SerumMutationSetBatch:
    values: dict[str, torch.Tensor | None] = {}
    for field in fields(SerumMutationSetBatch):
        value = getattr(batch, field.name)
        values[field.name] = None if value is None else value.index_select(0, indices)
    return SerumMutationSetBatch(**values)


def reverted_batch(
    batch: SerumMutationSetBatch,
    parent_indices: torch.Tensor,
    positions: torch.Tensor,
) -> SerumMutationSetBatch:
    reverted = select_batch(batch, parent_indices)
    reverted.query_aa = reverted.query_aa.clone()
    local_indices = torch.arange(len(parent_indices), device=parent_indices.device)
    reverted.query_aa[local_indices, 0, positions] = reverted.reference_aa[local_indices, positions]
    return reverted


def main() -> None:
    args = parse_args()
    model, checkpoint = load_model(args)
    records = pd.read_csv(args.records_csv)
    if args.max_records > 0:
        records = records.iloc[: args.max_records].copy()
    make_batch = make_batch_builder(model, checkpoint, args.embedding_dir, args.device)

    pair_rows: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            frame = records.iloc[start : start + args.batch_size].reset_index(drop=True)
            batch = make_batch(frame)
            original = model(batch)
            positions = original["mutation_positions"][:, 0].detach().cpu()
            counts = original["mutation_count"][:, 0].detach().cpu().to(torch.long)
            attention = original["mutation_attention"][:, 0].detach().cpu()
            original_mean = original["mean"][:, 0].detach().cpu()
            original_self = original["self_score"][:, 0].detach().cpu()
            original_query = original["query_score"][:, 0].detach().cpu()

            for index, record in enumerate(frame.itertuples(index=False)):
                pair_rows.append({
                    "sample_index": int(record.sample_index),
                    "seq_id_a": record.seq_id_a,
                    "seq_id_c": record.seq_id_c,
                    "serumName": record.serumName,
                    "virusName": record.virusName,
                    "label": float(record.label),
                    "mutation_count": int(counts[index]),
                    "original_mean": float(original_mean[index]),
                    "original_self_score": float(original_self[index]),
                    "original_query_score": float(original_query[index]),
                })

            parent_list: list[int] = []
            position_list: list[int] = []
            token_list: list[int] = []
            for parent_index, count in enumerate(counts.tolist()):
                for token_index in range(count):
                    parent_list.append(parent_index)
                    position_list.append(int(positions[parent_index, token_index]))
                    token_list.append(token_index)

            for chunk_start in range(0, len(parent_list), args.ablation_batch_size):
                chunk_end = chunk_start + args.ablation_batch_size
                parent_indices = torch.tensor(parent_list[chunk_start:chunk_end], device=args.device)
                ablation_positions = torch.tensor(position_list[chunk_start:chunk_end], device=args.device)
                reverted = model(reverted_batch(batch, parent_indices, ablation_positions))
                reverted_mean = reverted["mean"][:, 0].detach().cpu()
                reverted_self = reverted["self_score"][:, 0].detach().cpu()
                reverted_query = reverted["query_score"][:, 0].detach().cpu()

                for local_index, parent_index in enumerate(parent_list[chunk_start:chunk_end]):
                    position = position_list[chunk_start + local_index]
                    token_index = token_list[chunk_start + local_index]
                    record = frame.iloc[parent_index]
                    mutation_rows.append({
                        "sample_index": int(record.sample_index),
                        "seq_id_a": record.seq_id_a,
                        "seq_id_c": record.seq_id_c,
                        "serumName": record.serumName,
                        "virusName": record.virusName,
                        "serumPassCat": record.serumPassCat,
                        "virusPassCat": record.virusPassCat,
                        "serumDate": record.serumDate,
                        "virusDate": record.virusDate,
                        "label": float(record.label),
                        "aa_position": position + 1,
                        "reference_aa": str(record.serumHA)[position],
                        "query_aa": str(record.virusHA)[position],
                        "substitution": str(record.serumHA)[position] + "→" + str(record.virusHA)[position],
                        "mutation_count": int(counts[parent_index]),
                        "mutation_attention": float(attention[parent_index, token_index]),
                        "attention_enrichment": float(attention[parent_index, token_index] * counts[parent_index]),
                        "original_mean": float(original_mean[parent_index]),
                        "reverted_mean": float(reverted_mean[local_index]),
                        "delta_mean": float(original_mean[parent_index] - reverted_mean[local_index]),
                        "original_self_score": float(original_self[parent_index]),
                        "reverted_self_score": float(reverted_self[local_index]),
                        "original_query_score": float(original_query[parent_index]),
                        "reverted_query_score": float(reverted_query[local_index]),
                        "delta_query_score": float(original_query[parent_index] - reverted_query[local_index]),
                    })
            print(f"{start:,} / {len(records):,}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pair_rows).to_csv(args.output_dir / "pair_predictions.csv", index=False)
    pd.DataFrame(mutation_rows).to_csv(args.output_dir / "mutation_ablation_long.csv", index=False)
    with (args.output_dir / "analysis_config.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "records_csv": str(args.records_csv),
            "counterfactual": "set query amino-acid ID to reference amino-acid ID at one model-defined mutation token",
            "delta_mean": "original_mean - reverted_mean",
            "device": args.device,
            "batch_size": args.batch_size,
            "ablation_batch_size": args.ablation_batch_size,
            "max_records": args.max_records,
        }, handle, indent=2)


if __name__ == "__main__":
    main()
