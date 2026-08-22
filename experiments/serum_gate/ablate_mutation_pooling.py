"""Inference-time ablation for SerumMutationSet-Minus mutation pooling.

This preserves the trained pooling projection and replaces the missing branch
with a copy of the retained statistic.  It measures dependence of a checkpoint
on the two pooling statistics; it is not a substitute for retraining a smaller
architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_serum_mutation_set as trainer  # noqa: E402
from fluprofiler.models.serum_mutation_set_model import (  # noqa: E402
    SerumMutationSetConfig,
    SerumMutationSetMinusModel,
    _masked_mean,
)


POOLING_MODES = ("baseline", "mean_only", "attention_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate mean/attention pooling ablations on one checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Defaults to <checkpoint>/../../run_config.json.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-queries-per-task",
        type=int,
        default=128,
        help="Must match training unless the original run used another value.",
    )
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional path for the resulting metrics JSON.")
    parser.add_argument("--attention-csv", type=Path, default=None,
                        help="Optional CSV with one row per mutation token and its pooling weight.")
    return parser.parse_args()


def load_task_embeddings(
    item: dict[str, Any],
    embedding_dir: Path,
) -> dict[str, torch.Tensor]:
    """Load only the embeddings needed for the current serum task."""
    keys = [item["reference_ha_key"], *item["query_ha_keys"]]
    store: dict[str, torch.Tensor] = {}
    for key in dict.fromkeys(keys):
        value = torch.load(
            embedding_dir / f"{key}.pt",
            map_location="cpu",
            weights_only=False,
        )
        store[key] = torch.as_tensor(value).float()
    return store


def install_pooling_mode(
    model: SerumMutationSetMinusModel,
    mode: str,
) -> None:
    if mode == "baseline":
        return
    if mode not in {"mean_only", "attention_only"}:
        raise ValueError(f"Unsupported pooling mode: {mode}")

    def forward(
        self: torch.nn.Module,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        mutation_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = mask > 0
        mean_pooled = _masked_mean(tokens, mask)
        logits = self.score(torch.tanh(tokens)).squeeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1) * valid.to(tokens.dtype)
        attention_pooled = (tokens * weights.unsqueeze(-1)).sum(dim=-2)

        # Keep the trained 257 -> 128 projection shape intact.  The retained
        # statistic occupies both former pooling slots.
        if mode == "mean_only":
            attention_pooled = mean_pooled
        else:
            mean_pooled = attention_pooled

        count_feature = torch.log1p(mutation_count.to(tokens.dtype)).unsqueeze(-1)
        pooled = self.projection(
            torch.cat([mean_pooled, attention_pooled, count_feature], dim=-1)
        )
        return pooled, weights

    model.mutation_pool.forward = types.MethodType(forward, model.mutation_pool)


def load_checkpoint_model(
    checkpoint_path: Path,
    distance_matrix: torch.Tensor,
    device: torch.device,
) -> tuple[SerumMutationSetMinusModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SerumMutationSetMinusModel(
        SerumMutationSetConfig(**checkpoint["model_config"]),
        distance_matrix,
    )
    state = dict(checkpoint["model_state_dict"])
    weight_key = "predictor.0.weight"
    current_width = model.state_dict()[weight_key].shape[1]
    checkpoint_width = state[weight_key].shape[1]

    # Old checkpoints include s_nagly immediately before log1p(mutation_count).
    if checkpoint_width == current_width + 1:
        config = model.config
        old_nagly_column = (
            config.mutation_dim
            + config.passage_dim
            + config.passage_dim
            + config.subtype_dim
        )
        old_weight = state[weight_key]
        state[weight_key] = torch.cat(
            [old_weight[:, :old_nagly_column], old_weight[:, old_nagly_column + 1 :]],
            dim=1,
        )
    elif checkpoint_width != current_width:
        raise ValueError(
            f"Unexpected predictor input width: checkpoint={checkpoint_width}, "
            f"model={current_width}"
        )

    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint


def evaluate_mode(
    model: SerumMutationSetMinusModel,
    dataset: trainer.SerumMutationSetTaskDataset,
    embedding_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            embeddings = load_task_embeddings(item, embedding_dir)
            batch, _ = trainer.collate_mutation_set_tasks([item], embeddings)
            out = model(trainer.move_mutation_batch(batch, device))
            labels = batch.labels.reshape(-1).tolist()
            means = out["mean"].detach().cpu().reshape(-1).tolist()
            for label, mean in zip(labels, means):
                rows.append(
                    {
                        "task_key": item["task_key"],
                        "label": label,
                        "mean": mean,
                    }
                )
    return trainer.serum_regression_metrics(pd.DataFrame(rows))


def export_attention(
    model: SerumMutationSetMinusModel,
    dataset: trainer.SerumMutationSetTaskDataset,
    embedding_dir: Path,
    device: torch.device,
    output_path: Path,
) -> dict[str, float]:
    rows: list[dict[str, Any]] = []
    query_stats: list[dict[str, float]] = []
    with torch.no_grad():
        for item_index in range(len(dataset)):
            item = dataset[item_index]
            embeddings = load_task_embeddings(item, embedding_dir)
            batch, _ = trainer.collate_mutation_set_tasks([item], embeddings)
            out = model(trainer.move_mutation_batch(batch, device))
            weights = out["mutation_attention"][0].detach().cpu()
            positions = out["mutation_positions"][0].detach().cpu()
            counts = out["mutation_count"][0].detach().cpu().long()
            for query_index, count in enumerate(counts.tolist()):
                if count == 0:
                    continue
                reference = item["reference_ha_aligned"]
                query = item["query_ha_aligned"][query_index]
                token_weights = weights[query_index, :count]
                entropy = 0.0 if count == 1 else float(
                    -(token_weights * token_weights.clamp_min(1e-12).log()).sum()
                    / torch.log(torch.tensor(float(count)))
                )
                query_stats.append({"top_weight": float(token_weights.max()), "normalized_entropy": entropy})
                for token_index in range(count):
                    position = int(positions[query_index, token_index])
                    weight = float(token_weights[token_index])
                    rows.append({
                        "task_key": item["task_key"],
                        "query_index": query_index,
                        "mutation_count": count,
                        "position_1based": position + 1,
                        "mutation": f"{reference[position]}{position + 1}{query[position]}",
                        "attention_weight": weight,
                        "uniform_weight": 1.0 / count,
                        "relative_to_uniform": weight * count,
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path.expanduser(), index=False)
    stats = pd.DataFrame(query_stats)
    return {
        "mutation_token_rows": float(len(frame)),
        "queries_with_mutations": float(len(stats)),
        "mean_top_weight": float(stats["top_weight"].mean()),
        "mean_normalized_entropy": float(stats["normalized_entropy"].mean()),
    }
def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = (
        args.run_config.expanduser().resolve()
        if args.run_config is not None
        else checkpoint_path.parent.parent / "run_config.json"
    )
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable but --device={args.device!r} was requested")

    frames = trainer.load_fixed_split_frames(
        Path(run_config["data_dir"]),
        refit_train_valid=bool(run_config["refit_train_valid"]),
        type_filter=run_config["type_filter"],
    )
    vocabs = trainer.build_vocabs(frames, use_subtype_feature=False)
    dataset = trainer.SerumMutationSetTaskDataset(
        frames["test"],
        vocabs,
        task_cols=run_config["task_cols"],
        max_queries_per_task=args.max_queries_per_task,
    )
    distance_matrix = trainer.load_ha_distance_matrix(
        Path(run_config["ha_distance_matrix"])
    )
    embedding_dir = Path(run_config["embedding_dir"])

    results: dict[str, dict[str, float]] = {}
    for mode in POOLING_MODES:
        model, _ = load_checkpoint_model(checkpoint_path, distance_matrix, device)
        install_pooling_mode(model, mode)
        results[mode] = evaluate_mode(model, dataset, embedding_dir, device)

    if args.attention_csv is not None:
        model, _ = load_checkpoint_model(checkpoint_path, distance_matrix, device)
        results["pooling_attention_summary"] = export_attention(
            model, dataset, embedding_dir, device, args.attention_csv
        )
    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.expanduser().write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
