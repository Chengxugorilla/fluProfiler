#!/usr/bin/env python3
"""
Evaluate a saved SerumGate checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_REPO_ROOT / "src"))
sys.path.append(str(_THIS_FILE.parent))

from fluprofiler.models.serum_gate_model import SerumGateModel  # noqa: E402
from train_zero_shot import (  # noqa: E402
    GpuEmbeddingCache,
    SerumGateVocabs,
    build_loader,
    evaluate_model,
    load_embeddings,
    load_fixed_split_frames,
    parse_column_list,
    required_embedding_files,
    validate_embedding_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SerumGate checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--serum-task-cols", type=str, default="seq_id_a,seq_id_b,serumPassCat")
    parser.add_argument("--sample-limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-queries-per-task", type=int, default=128)
    parser.add_argument("--gpu-cache-gb", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    if args.max_queries_per_task <= 0:
        args.max_queries_per_task = None
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    task_cols = parse_column_list(args.serum_task_cols)
    sample_limit = None if args.sample_limit < 0 else int(args.sample_limit)
    frames = load_fixed_split_frames(
        args.data_dir,
        sample_limit=sample_limit,
    )
    embedding_files = required_embedding_files(frames)
    embedding_dir = validate_embedding_files(args.embedding_dir, embedding_files)
    embeddings = load_embeddings(embedding_dir, embedding_files)
    vocabs = SerumGateVocabs(
        passage_to_id=checkpoint["passage_to_id"],
        subtype_to_id=checkpoint["subtype_to_id"],
    )
    device = torch.device(args.device)
    gpu_cache = None
    if args.gpu_cache_gb > 0:
        if device.type != "cuda":
            raise ValueError("--gpu-cache-gb requires a CUDA device")
        gpu_cache = GpuEmbeddingCache(
            cpu_store=embeddings,
            device=device,
            max_bytes=int(float(args.gpu_cache_gb) * 1024**3),
        )
    loader = build_loader(
        frames[args.split],
        vocabs,
        embeddings,
        args.batch_size,
        shuffle=False,
        max_queries_per_task=args.max_queries_per_task,
        gpu_cache=gpu_cache,
        device=device,
        task_cols=task_cols,
    )
    model = SerumGateModel(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, predictions = evaluate_model(model.to(device), loader, device)

    result = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "split": args.split,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    predictions.to_csv(output_dir / f"predictions_{args.split}.csv", index=False)


if __name__ == "__main__":
    main()
