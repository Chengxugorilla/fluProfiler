#!/usr/bin/env python3
"""
Plot true-vs-predicted labels for a Metric HA checkpoint on fixed splits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_REPO_ROOT / "src"))
sys.path.append(str(_REPO_ROOT / "src" / "fluprofiler"))
sys.path.append(str(_THIS_FILE.parent))
sys.path.append(str(_REPO_ROOT / "experiments" / "reverse_tests"))

from experiment_tools import GpuEmbeddingCache  # noqa: E402
from fluprofiler.data.loaders import load_embedding  # noqa: E402
from fluprofiler.models.metric_antigenic_model import MetricHAAntigenicModel  # noqa: E402
from train_metric_ha import (  # noqa: E402
    MetricHADataset,
    batch_to_model_input,
    build_feature_vocabs,
    load_fixed_split_frames,
    regression_metrics,
    required_embedding_files,
    validate_embedding_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Metric HA true-vs-predicted labels.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=["train", "valid"])
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu-cache-gb", type=float, default=20.0)
    parser.add_argument("--max-plot-points", type=int, default=0, help="0 means plot all points.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return (_REPO_ROOT / path).resolve()


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but torch.cuda.is_available() is false")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.get_device_properties(device.index)
    return device


def infer_split(
    split_name: str,
    frame,
    vocabs,
    model: MetricHAAntigenicModel,
    device: torch.device,
    cache: GpuEmbeddingCache,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    dataset = MetricHADataset(frame, vocabs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true: list[float] = []
    y_pred: list[float] = []
    with torch.no_grad():
        for batch in loader:
            model_batch = batch_to_model_input(batch, device, cache)
            out = model(model_batch)
            y_true.extend(batch["label"].cpu().view(-1).tolist())
            y_pred.extend(out["pred"].detach().cpu().view(-1).tolist())
    metrics = regression_metrics(y_true, y_pred)
    return np.asarray(y_true), np.asarray(y_pred), metrics


def plot_split(
    output_dir: Path,
    split_name: str,
    epoch: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    max_plot_points: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"{split_name}_true_vs_pred.png"
    out_csv = output_dir / f"{split_name}_true_vs_pred.csv"
    np.savetxt(
        out_csv,
        np.column_stack([y_true, y_pred]),
        delimiter=",",
        header="true,pred",
        comments="",
    )

    plot_true = y_true
    plot_pred = y_pred
    if max_plot_points > 0 and len(y_true) > max_plot_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_true), size=max_plot_points, replace=False)
        plot_true = y_true[idx]
        plot_pred = y_pred[idx]

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=180)
    point_size = 5 if len(plot_true) > 20000 else 9
    alpha = 0.13 if len(plot_true) > 20000 else 0.25
    ax.scatter(plot_true, plot_pred, s=point_size, alpha=alpha, linewidths=0, color="#2563eb")
    ax.plot([lo, hi], [lo, hi], color="#dc2626", linewidth=1.4, linestyle="--", label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")
    ax.set_title(f"Metric HA Serum {split_name.title()}: True vs Predicted (checkpoint epoch {epoch})")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", frameon=False)
    text = (
        f"n = {len(y_true)}\n"
        f"MAE = {metrics['mae']:.3f}\n"
        f"MSE = {metrics['mse']:.3f}\n"
        f"Pearson = {metrics['pearson']:.3f}\n"
        f"R2 = {metrics['r2']:.3f}"
    )
    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9, "boxstyle": "round,pad=0.35"},
    )
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_csv}")
    print(split_name, metrics)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_dir = resolve_repo_path(args.data_dir)
    embedding_dir_arg = resolve_repo_path(args.embedding_dir)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    output_dir = resolve_repo_path(args.output_dir)

    frames = load_fixed_split_frames(data_dir)
    selected_frames = {split: frames[split] for split in args.splits}
    vocabs = build_feature_vocabs(frames)
    embedding_files = required_embedding_files(selected_frames)
    embedding_dir = validate_embedding_files(embedding_dir_arg, embedding_files)
    embeddings = load_embedding(str(embedding_dir), files=embedding_files)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = MetricHAAntigenicModel(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    max_cache_bytes = int(float(args.gpu_cache_gb) * 1024**3) if device.type == "cuda" else 0
    cache = GpuEmbeddingCache(cpu_store=embeddings, device=device, max_bytes=max_cache_bytes)
    for split_name in args.splits:
        y_true, y_pred, metrics = infer_split(
            split_name,
            frames[split_name],
            vocabs,
            model,
            device,
            cache,
            args.batch_size,
        )
        plot_split(
            output_dir,
            split_name,
            int(checkpoint["epoch"]),
            y_true,
            y_pred,
            metrics,
            args.max_plot_points,
            args.seed,
        )


if __name__ == "__main__":
    main()
