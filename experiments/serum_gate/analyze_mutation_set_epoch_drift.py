#!/usr/bin/env python3
"""Track season-split generalization and internal drift across epoch checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_serum_mutation_set as trainer  # noqa: E402
from fluprofiler.models.serum_mutation_set_model import (  # noqa: E402
    SerumMutationSetConfig,
    SerumMutationSetMinusModel,
)

MODULE_PREFIXES = {
    "site_projection": ("site_projection.",),
    "background_encoder": ("background_encoder.",),
    "mutation_tokens": (
        "position_embedding.", "amino_acid_embedding.", "presence_embedding.",
        "mutation_token_projection.", "background_to_mutation.", "null_mutation",
    ),
    "mutation_transformer": ("mutation_blocks.", "mutation_pool."),
    "conditioning": (
        "passage_embedding.", "passage_pair_embedding.", "subtype_embedding.",
        "serum_condition_encoder.", "film.",
    ),
    "predictor": ("predictor.",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze prediction and representation drift across epoch checkpoints."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-queries-per-task", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-predictions", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def checkpoint_paths(run_dir: Path) -> list[Path]:
    paths = sorted((run_dir / "checkpoints").glob("epoch_*.pth"))
    if not paths:
        raise FileNotFoundError(
            "No epoch_XXXX.pth checkpoints found. Re-run training with --save-every-epoch."
        )
    return paths


def relative_parameter_drift(
    current: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], prefixes: tuple[str, ...]
) -> float:
    numerator = 0.0
    denominator = 0.0
    for name, value in current.items():
        if not name.startswith(prefixes) or not torch.is_floating_point(value):
            continue
        delta = value.float() - reference[name].float()
        numerator += float(delta.square().sum())
        denominator += float(reference[name].float().square().sum())
    return float(np.sqrt(numerator) / max(np.sqrt(denominator), 1e-12))


def summarize_batch(out: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, float]:
    valid = mask > 0
    z_mut = out["z_mutation"][valid]
    theta = out["theta_serum"]
    background = out["z_background"]
    attention = out["mutation_attention"]
    token_mask = out["mutation_token_mask"] > 0
    token_count = token_mask.sum(dim=-1).clamp_min(1)
    masked_attention = attention.masked_fill(~token_mask, 0.0)
    entropy = -(masked_attention.clamp_min(1e-12).log() * masked_attention).sum(dim=-1)
    entropy = entropy / token_count.float().log().clamp_min(1.0)
    gamma, beta = out["theta_serum"].new_zeros((1, 1)), out["theta_serum"].new_zeros((1, 1))
    # gamma/beta are recomputed by the caller because they are model parameters.
    return {
        "z_background_l2": float(background.norm(dim=-1).mean()),
        "z_mutation_l2": float(z_mut.norm(dim=-1).mean()),
        "theta_l2": float(theta.norm(dim=-1).mean()),
        "pool_attention_top1": float(masked_attention.max(dim=-1).values[valid].mean()),
        "pool_attention_entropy": float(entropy[valid].mean()),
        "mean_mutation_count": float(out["mutation_count"][valid].float().mean()),
    }


def evaluate(
    model: SerumMutationSetMinusModel, loader: Any, device: torch.device, write_predictions: bool
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    aggregates: dict[str, list[float]] = {}
    model.eval()
    with torch.inference_mode():
        for batch, items in loader:
            moved = trainer.move_mutation_batch(batch, device)
            out = model(moved)
            stats = summarize_batch(out, moved.query_mask)
            gamma, beta = model.film(out["theta_serum"]).chunk(2, dim=-1)
            stats["film_gamma_abs_mean"] = float(gamma.abs().mean())
            stats["film_beta_abs_mean"] = float(beta.abs().mean())
            for name, value in stats.items():
                aggregates.setdefault(name, []).append(value)
            if write_predictions:
                metadata = items[0]["query_meta"]
                values = {
                    name: out[name].detach().cpu().reshape(-1).numpy()
                    for name in ("mean", "self_score", "query_score", "mutation_count")
                }
                labels = moved.labels.detach().cpu().reshape(-1).numpy()
                for index, label in enumerate(labels):
                    row = metadata.iloc[index].to_dict()
                    row.update({"task_key": items[0]["task_key"], "label": float(label), **{k: float(v[index]) for k, v in values.items()}})
                    rows.append(row)
    prediction = pd.DataFrame(rows)
    metrics = trainer.serum_regression_metrics(prediction) if write_predictions else {}
    metrics.update({name: float(np.mean(values)) for name, values in aggregates.items()})
    return metrics, rows


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    paths = checkpoint_paths(run_dir)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = trainer.load_fixed_split_frames(
        Path(config["data_dir"]), refit_train_valid=bool(config["refit_train_valid"]),
        type_filter=str(config.get("type_filter") or ""),
    )
    frame = frames[args.split]
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    vocabs = trainer.MutationSetVocabs(first["passage_to_id"], first["subtype_to_id"])
    embedding_dir = Path(config["embedding_dir"])
    embeddings = trainer.load_embeddings(
        embedding_dir, trainer.required_ha_embedding_files({args.split: frame}), show_progress=True
    )
    loader = trainer.build_loader(
        frame, vocabs, embeddings, batch_size=1, shuffle=False,
        max_queries_per_task=args.max_queries_per_task, task_cols=first["task_cols"],
    )
    distance = trainer.load_ha_distance_matrix(Path(config["ha_distance_matrix"]))
    reference_state = first["model_state_dict"]
    summary: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = SerumMutationSetMinusModel(
            SerumMutationSetConfig(**checkpoint["model_config"]), distance
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        metrics, rows = evaluate(model, loader, device, args.write_predictions)
        result: dict[str, Any] = {"epoch": int(checkpoint["epoch"]), "checkpoint": path.name, **metrics}
        result.update({
            f"drift_{group}": relative_parameter_drift(
                checkpoint["model_state_dict"], reference_state, prefixes
            ) for group, prefixes in MODULE_PREFIXES.items()
        })
        summary.append(result)
        if args.write_predictions:
            prediction = pd.DataFrame(rows)
            prediction.insert(0, "epoch", int(checkpoint["epoch"]))
            all_predictions.append(prediction)
        print(json.dumps(result, ensure_ascii=False))
    pd.DataFrame(summary).sort_values("epoch").to_csv(output_dir / "epoch_drift_summary.csv", index=False)
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(output_dir / "epoch_predictions.csv", index=False)


if __name__ == "__main__":
    main()
