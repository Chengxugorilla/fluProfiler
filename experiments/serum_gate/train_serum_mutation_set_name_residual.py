#!/usr/bin/env python3
"""Train an isolated MutationSet-Minus variant with Name scalar residuals."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_THIS_FILE.parent))
sys.path.append(str(_REPO_ROOT / "src"))

import train_serum_mutation_set as base  # noqa: E402
from fluprofiler.models.serum_mutation_set_name_residual_model import (  # noqa: E402
    SerumMutationSetMinusNameResidualModel,
)


UNK_NAME = "<UNK>"


@dataclass(frozen=True)
class NameResidualVocabs:
    passage_to_id: dict[str, int]
    subtype_to_id: dict[str, int]
    serum_name_to_id: dict[str, int]
    virus_name_to_id: dict[str, int]


def normalize_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).strip().upper().split())


def build_name_vocabs(train_frame: pd.DataFrame) -> NameResidualVocabs:
    missing = {"serumName", "virusName"} - set(train_frame.columns)
    if missing:
        raise ValueError("training frame is missing name column(s): " + ", ".join(sorted(missing)))

    def make_map(column: str) -> dict[str, int]:
        names = sorted({normalize_name(value) for value in train_frame[column] if normalize_name(value)})
        return {UNK_NAME: 0, **{name: index + 1 for index, name in enumerate(names)}}

    return NameResidualVocabs({}, {}, make_map("serumName"), make_map("virusName"))


def name_id(value: Any, vocabulary: dict[str, int]) -> int:
    normalized = normalize_name(value)
    return vocabulary.get(normalized, 0) if normalized else 0


class NameResidualTaskDataset(base.SerumMutationSetTaskDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        metadata = item["query_meta"]
        serum_names = {normalize_name(value) for value in metadata["serumName"] if normalize_name(value)}
        if len(serum_names) > 1:
            raise ValueError(f"task {item['task_key']!r} has more than one serumName")
        serum_name = next(iter(serum_names), "")
        item["serum_name_id"] = torch.tensor(
            name_id(serum_name, self.vocabs.serum_name_to_id), dtype=torch.long,
        )
        item["virus_name_id"] = torch.tensor(
            [name_id(value, self.vocabs.virus_name_to_id) for value in metadata["virusName"]],
            dtype=torch.long,
        )
        return item


def collate_name_residual_tasks(
    items: list[dict[str, Any]], embeddings: Any, aligned_cache: base.AlignedEmbeddingCache | None = None,
) -> tuple[base.SerumMutationSetBatch, list[dict[str, Any]]]:
    batch, source_items = base.collate_mutation_set_tasks(items, embeddings, aligned_cache)
    batch.serum_name_id = items[0]["serum_name_id"].view(1)
    batch.virus_name_id = items[0]["virus_name_id"].view(1, -1)
    return batch, source_items


def build_loader(
    frame: pd.DataFrame, vocabs: NameResidualVocabs, embeddings: Any, batch_size: int,
    shuffle: bool, max_queries_per_task: int | None, task_cols: list[str],
    aligned_cache: base.AlignedEmbeddingCache | None = None,
) -> DataLoader:
    if batch_size != 1:
        raise ValueError("Use --batch-size 1 for variable-size serum tasks")
    dataset = NameResidualTaskDataset(frame, vocabs, task_cols, max_queries_per_task)
    return DataLoader(
        dataset, batch_size=1, shuffle=shuffle,
        collate_fn=partial(
            collate_name_residual_tasks, embeddings=embeddings,
            aligned_cache=aligned_cache or base.AlignedEmbeddingCache(),
        ),
    )


def move_name_residual_batch(
    batch: base.SerumMutationSetBatch, device: torch.device,
) -> base.SerumMutationSetBatch:
    values = {
        field.name: getattr(batch, field.name).to(device)
        if isinstance(getattr(batch, field.name), torch.Tensor) else getattr(batch, field.name)
        for field in fields(base.SerumMutationSetBatch)
    }
    moved = base.SerumMutationSetBatch(**values)
    moved.serum_name_id = batch.serum_name_id.to(device)
    moved.virus_name_id = batch.virus_name_id.to(device)
    return moved


def evaluate_name_residual_model(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, progress_bar: Any = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch, items in loader:
            moved = move_name_residual_batch(batch, device)
            out = model(moved)
            losses.append(float(out["huber_loss"].item()))
            recorded = {
                name: out[name].detach().cpu().reshape(-1).numpy()
                for name in (
                    "mean", "sequence_mean", "serum_name_effect", "virus_name_effect",
                    "self_score", "query_score", "mutation_count",
                )
            }
            labels = moved.labels.detach().cpu().reshape(-1).numpy()
            for index, label in enumerate(labels):
                row = items[0]["query_meta"].iloc[index].to_dict()
                row.update({
                    "task_key": items[0]["task_key"], "label": float(label),
                    **{name: float(values[index]) for name, values in recorded.items()},
                })
                rows.append(row)
            if progress_bar is not None:
                progress_bar.update(1)
    predictions = pd.DataFrame(rows)
    metrics = base.serum_regression_metrics(predictions)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, predictions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    name_parser = argparse.ArgumentParser(add_help=False)
    name_parser.add_argument("--serum-name-l2", type=float, default=1e-3)
    name_parser.add_argument("--virus-name-l2", type=float, default=1e-2)
    name_args, remaining = name_parser.parse_known_args(argv)
    args = base.parse_args(remaining)
    if name_args.serum_name_l2 < 0 or name_args.virus_name_l2 < 0:
        name_parser.error("name L2 coefficients must be non-negative")
    args.serum_name_l2 = name_args.serum_name_l2
    args.virus_name_l2 = name_args.virus_name_l2
    return args


_original_build_vocabs = base.build_vocabs
_original_build_model_config = base._build_model_config
_original_load_frames = base.load_fixed_split_frames
_active_vocabs: NameResidualVocabs | None = None


def _build_vocabs(frames: dict[str, pd.DataFrame], use_subtype_feature: bool = True) -> NameResidualVocabs:
    global _active_vocabs
    base_vocabs = _original_build_vocabs(frames, use_subtype_feature)
    name_vocabs = build_name_vocabs(frames["train"])
    _active_vocabs = NameResidualVocabs(
        base_vocabs.passage_to_id, base_vocabs.subtype_to_id,
        name_vocabs.serum_name_to_id, name_vocabs.virus_name_to_id,
    )
    return _active_vocabs


def _load_frames(*args: Any, **kwargs: Any) -> dict[str, pd.DataFrame]:
    frames = _original_load_frames(*args, **kwargs)
    for split, frame in frames.items():
        missing = {"serumName", "virusName"} - set(frame.columns)
        if missing:
            raise ValueError(f"{split} frame is missing name column(s): {', '.join(sorted(missing))}")
    return frames


def _build_model_config(*args: Any, **kwargs: Any) -> base.SerumMutationSetConfig:
    config = _original_build_model_config(*args, **kwargs)
    if _active_vocabs is None:
        raise RuntimeError("name vocabularies must be built before the model config")
    parsed_args = args[0]
    config.serum_name_vocab_size = len(_active_vocabs.serum_name_to_id)
    config.virus_name_vocab_size = len(_active_vocabs.virus_name_to_id)
    config.serum_name_l2 = float(parsed_args.serum_name_l2)
    config.virus_name_l2 = float(parsed_args.virus_name_l2)
    return config


class _ConfiguredNameResidualModel(SerumMutationSetMinusNameResidualModel):
    def __init__(self, config: base.SerumMutationSetConfig, ha_distance_matrix: torch.Tensor):
        super().__init__(
            config, ha_distance_matrix,
            serum_name_vocab_size=config.serum_name_vocab_size,
            virus_name_vocab_size=config.virus_name_vocab_size,
            serum_name_l2=config.serum_name_l2,
            virus_name_l2=config.virus_name_l2,
        )


def _augment_saved_metadata(result: dict[str, Any]) -> None:
    if _active_vocabs is None:
        raise RuntimeError("name vocabularies were not created")
    root = Path(result["paths"]["root"])
    metadata = {
        "model": "SerumMutationSet-Minus-NameResidual",
        "name_normalization": "strip, uppercase, collapse whitespace; missing/unseen -> <UNK>=0",
        "serum_name_to_id": _active_vocabs.serum_name_to_id,
        "virus_name_to_id": _active_vocabs.virus_name_to_id,
    }
    config_path = root / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(metadata)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    for checkpoint_path in (root / "checkpoints").glob("*.pth"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint.update(metadata)
        torch.save(checkpoint, checkpoint_path)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    base.build_vocabs = _build_vocabs
    base.build_loader = build_loader
    base.move_mutation_batch = move_name_residual_batch
    base.evaluate_model = evaluate_name_residual_model
    base.load_fixed_split_frames = _load_frames
    base._build_model_config = _build_model_config
    base.SerumMutationSetMinusModel = _ConfiguredNameResidualModel
    result = base.run_training(args)
    _augment_saved_metadata(result)
    return result


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
