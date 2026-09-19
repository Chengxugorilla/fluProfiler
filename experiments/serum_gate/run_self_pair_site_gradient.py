#!/usr/bin/env python3
"""Compute full-HA1 self-pair background gradients for SerumMutationSetMinusModel.

Each unique test virus is used as both the reference and query sequence.  This
leaves the mutation set empty while retaining the model's complete reference
background path.  For every HA1 position, the script exports the L2 norm of
the query-score gradient with respect to the reference PLM embedding. The
self-pair mean output is unsuitable here: with an empty mutation set its
self_score - query_score terms cancel and its gradient is zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-viruses", type=int, default=0)
    parser.add_argument(
        "--target",
        choices=("mean", "self_score", "query_score"),
        default="query_score",
        help="Model output differentiated with respect to reference_ha.",
    )
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

    def make_self_pair_batch(frame: pd.DataFrame) -> SerumMutationSetBatch:
        aligned = [aligned_input(row.seq_id_c, row.virusHA) for row in frame.itertuples(index=False)]
        reference_ha = torch.stack([item[0] for item in aligned]).to(device).detach()
        reference_ha.requires_grad_(True)
        passage = torch.tensor(
            [passage_to_id[normalize_passage(value)] for value in frame.virusPassCat],
            device=device,
        )
        query_passage = passage[:, None]
        reference_aa = torch.stack([item[3] for item in aligned]).to(device)
        reference_mask = torch.stack([item[1] for item in aligned]).to(device)
        embedding_mask = torch.stack([item[2] for item in aligned]).to(device)
        return SerumMutationSetBatch(
            reference_ha=reference_ha,
            query_ha=reference_ha.detach()[:, None],
            reference_aa=reference_aa,
            query_aa=reference_aa[:, None],
            reference_aligned_mask=reference_mask,
            query_aligned_mask=reference_mask[:, None],
            reference_embedding_mask=embedding_mask,
            query_embedding_mask=embedding_mask[:, None],
            serum_passage=passage,
            query_passage=query_passage,
            passage_pair=passage[:, None] * model.config.passage_vocab_size + query_passage,
            subtype=torch.zeros(len(frame), dtype=torch.long, device=device),
            query_mask=torch.ones(len(frame), 1, device=device),
        )

    return make_self_pair_batch


def main() -> None:
    args = parse_args()
    model, checkpoint = load_model(args)
    records = pd.read_csv(args.records_csv)
    viruses = records.drop_duplicates("seq_id_c", keep="first").reset_index(drop=True)
    if args.max_viruses > 0:
        viruses = viruses.iloc[: args.max_viruses].copy()
    make_batch = make_batch_builder(model, checkpoint, args.embedding_dir, args.device)

    raw_rows: list[np.ndarray] = []
    normalized_rows: list[np.ndarray] = []
    valid_mask_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []

    for start in range(0, len(viruses), args.batch_size):
        frame = viruses.iloc[start : start + args.batch_size].reset_index(drop=True)
        batch = make_batch(frame)
        output = model(batch)
        target = output[args.target][:, 0].sum()
        gradients = torch.autograd.grad(target, batch.reference_ha)[0]
        raw_scores = gradients.norm(p=2, dim=-1)
        valid_mask = batch.reference_embedding_mask > 0
        raw_scores = raw_scores * valid_mask.to(raw_scores.dtype)
        normalized_scores = raw_scores / raw_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)

        raw_rows.append(raw_scores.detach().cpu().numpy().astype(np.float32))
        normalized_rows.append(normalized_scores.detach().cpu().numpy().astype(np.float32))
        valid_mask_rows.append(valid_mask.detach().cpu().numpy().astype(bool))
        mutation_counts = output["mutation_count"][:, 0].detach().cpu().to(torch.long)

        for index, row in enumerate(frame.itertuples(index=False)):
            metadata_rows.append({
                "seq_id_c": row.seq_id_c,
                "virusName": row.virusName,
                "virusPassCat": row.virusPassCat,
                "virusDate": row.virusDate,
                "self_pair_mean": float(output["mean"][index, 0].detach().cpu()),
                "self_pair_self_score": float(output["self_score"][index, 0].detach().cpu()),
                "self_pair_query_score": float(output["query_score"][index, 0].detach().cpu()),
                "mutation_count": int(mutation_counts[index]),
                "raw_gradient_l2_sum": float(raw_scores[index].sum().detach().cpu()),
            })
        print(f"{start:,} / {len(viruses):,}", flush=True)

    raw_scores = np.concatenate(raw_rows, axis=0)
    normalized_scores = np.concatenate(normalized_rows, axis=0)
    valid_masks = np.concatenate(valid_mask_rows, axis=0)
    site_rows: list[dict[str, Any]] = []
    for position in range(raw_scores.shape[1]):
        values = normalized_scores[valid_masks[:, position], position]
        raw_values = raw_scores[valid_masks[:, position], position]
        site_rows.append({
            "aa_position": position + 1,
            "n_viruses": int(valid_masks[:, position].sum()),
            "mean_normalized_gradient_l2": float(values.mean()),
            "median_normalized_gradient_l2": float(np.median(values)),
            "std_normalized_gradient_l2": float(values.std(ddof=0)),
            "mean_raw_gradient_l2": float(raw_values.mean()),
            "median_raw_gradient_l2": float(np.median(raw_values)),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "self_pair_site_gradients.npz",
        raw_gradient_l2=raw_scores,
        normalized_gradient_l2=normalized_scores,
        valid_position_mask=valid_masks,
    )
    pd.DataFrame(metadata_rows).to_csv(args.output_dir / "self_pair_virus_metadata.csv", index=False)
    pd.DataFrame(site_rows).sort_values(
        "mean_normalized_gradient_l2", ascending=False, kind="stable"
    ).to_csv(args.output_dir / "global_site_gradient_summary.csv", index=False)
    with (args.output_dir / "analysis_config.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "records_csv": str(args.records_csv),
            "pair_definition": "reference_ha and query_ha are the same test-virus sequence; mutation set is empty",
            "gradient_target": args.target,
            "gradient_input": "reference_ha",
            "site_score": "L2 norm of d(target) / d(reference_ha[position])",
            "normalization": "within-virus L1 normalization of valid-position L2 scores",
            "device": args.device,
            "batch_size": args.batch_size,
            "max_viruses": args.max_viruses,
            "n_viruses": int(len(viruses)),
        }, handle, indent=2)


if __name__ == "__main__":
    main()
