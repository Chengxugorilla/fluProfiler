#!/usr/bin/env python3
"""
Evaluate a direct HA embedding-distance baseline on fixed split CSVs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_REPO_ROOT / "src"))
sys.path.append(str(_THIS_FILE.parent))

from fluprofiler.data.loaders import load_embedding  # noqa: E402
from train_metric_ha import (  # noqa: E402
    load_fixed_split_frames,
    regression_metrics,
    required_embedding_files,
    validate_embedding_files,
)


def mean_pool_embedding(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {tuple(matrix.shape)}")
    return matrix.float().mean(dim=0)


def embedding_distance(serum_matrix: torch.Tensor, virus_matrix: torch.Tensor, metric: str = "euclidean") -> float:
    serum = mean_pool_embedding(serum_matrix)
    virus = mean_pool_embedding(virus_matrix)

    if metric == "euclidean":
        return float(torch.linalg.vector_norm(serum - virus).item())
    if metric == "cosine":
        denom = torch.linalg.vector_norm(serum) * torch.linalg.vector_norm(virus)
        if float(denom.item()) == 0.0:
            return 0.0
        similarity = torch.dot(serum, virus) / denom
        return float((1.0 - similarity).item())
    raise ValueError(f"Unsupported distance metric: {metric}")


def fit_linear_calibration(distances: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    distances = np.asarray(distances, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if distances.shape != labels.shape:
        raise ValueError("distances and labels must have the same shape")
    if distances.ndim != 1:
        raise ValueError("distances and labels must be 1D arrays")
    if len(distances) == 0:
        raise ValueError("Cannot fit calibration without training samples")

    slope, intercept = np.polyfit(distances, labels, deg=1)
    return {"slope": float(slope), "intercept": float(intercept)}


def apply_linear_calibration(distances: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    return calibration["slope"] * distances + calibration["intercept"]


def precompute_pooled_embeddings(embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: mean_pool_embedding(value).cpu() for key, value in embeddings.items()}


def compute_split_distances(
    frame,
    pooled_embeddings: dict[str, torch.Tensor],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    serum_vectors = torch.stack([pooled_embeddings[f"matrix_{seq_id}"] for seq_id in frame["seq_id_a"].tolist()])
    virus_vectors = torch.stack([pooled_embeddings[f"matrix_{seq_id}"] for seq_id in frame["seq_id_c"].tolist()])

    if metric == "euclidean":
        distances = torch.linalg.vector_norm(serum_vectors - virus_vectors, dim=1)
    elif metric == "cosine":
        similarity = torch.nn.functional.cosine_similarity(serum_vectors, virus_vectors, dim=1)
        distances = 1.0 - similarity
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")

    labels = frame["label"].to_numpy(dtype=float)
    return distances.numpy().astype(float), labels


def evaluate_split(distances: np.ndarray, labels: np.ndarray, calibration: dict[str, float]) -> dict[str, float]:
    predictions = apply_linear_calibration(distances, calibration)
    metrics = regression_metrics(labels.tolist(), predictions.tolist())
    metrics["distance_mean"] = float(np.mean(distances)) if len(distances) else 0.0
    metrics["distance_std"] = float(np.std(distances)) if len(distances) else 0.0
    metrics["pred_mean"] = float(np.mean(predictions)) if len(predictions) else 0.0
    metrics["label_mean"] = float(np.mean(labels)) if len(labels) else 0.0
    return metrics


def prepare_output_dir(output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": output_dir,
        "config": output_dir / "run_config.json",
        "metrics": output_dir / "metrics.json",
        "log": output_dir / "log.txt",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direct HA embedding-distance baseline.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", choices=("euclidean", "cosine"), default="euclidean")
    parser.add_argument("--sample-limit", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_limit == 0 or args.sample_limit < -1:
        raise ValueError("--sample-limit must be -1 (all rows) or a positive integer")

    sample_limit = None if args.sample_limit < 0 else int(args.sample_limit)
    frames = load_fixed_split_frames(args.data_dir, sample_limit=sample_limit)
    embedding_files = required_embedding_files(frames)
    embedding_dir = validate_embedding_files(args.embedding_dir, embedding_files)
    embeddings = load_embedding(str(embedding_dir), files=embedding_files)
    pooled_embeddings = precompute_pooled_embeddings(embeddings)
    paths = prepare_output_dir(args.output_dir)

    split_distances: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: compute_split_distances(frame, pooled_embeddings, args.metric)
        for name, frame in frames.items()
    }
    calibration = fit_linear_calibration(*split_distances["train"])
    metrics = {
        name: evaluate_split(distances, labels, calibration)
        for name, (distances, labels) in split_distances.items()
    }

    run_config: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": "embedding_distance_linear_calibration",
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "embedding_dir": str(embedding_dir),
        "output_dir": str(paths["root"]),
        "row_counts": {name: len(frame) for name, frame in frames.items()},
        "embedding_file_count": len(embedding_files),
        "metric": args.metric,
        "sample_limit": sample_limit,
        "calibration": calibration,
    }
    result = {"config": run_config, "metrics": metrics}

    paths["config"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["metrics"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with paths["log"].open("w", encoding="utf-8") as log_file:
        log_file.write("===== RUN CONFIG START =====\n")
        json.dump(run_config, log_file, indent=2, ensure_ascii=False)
        log_file.write("\n===== RUN CONFIG END =====\n\n")
        json.dump(metrics, log_file, ensure_ascii=False)
        log_file.write("\n")


if __name__ == "__main__":
    main()
