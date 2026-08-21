#!/usr/bin/env python3
"""Train the independent HA-only SerumMutationSet-Minus model."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
import sys
from dataclasses import dataclass, fields
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_REPO_ROOT / "src"))
sys.path.append(str(_REPO_ROOT / "experiments" / "reverse_tests"))

from experiment_tools import GpuEmbeddingCache  # noqa: E402
from fluprofiler.models.serum_mutation_set_model import (  # noqa: E402
    SerumMutationSetBatch,
    SerumMutationSetConfig,
    SerumMutationSetMinusModel,
)


SPLIT_FILENAMES = ("train.csv", "valid.csv", "test.csv")
DEFAULT_TASK_COLS = ["seq_id_a", "serumPassCat", "serumName"]
HA_ONLY_REQUIRED_COLUMNS = {
    "seq_id_a",
    "seq_id_c",
    "serumHA",
    "virusHA",
    "serumPassCat",
    "virusPassCat",
    "label",
}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_ID = 0
AMINO_ACID_TO_ID = {residue: index + 1 for index, residue in enumerate(AMINO_ACIDS)}
UNKNOWN_ID = len(AMINO_ACID_TO_ID) + 1
GAP_ID = UNKNOWN_ID + 1
AMINO_ACID_VOCAB_SIZE = GAP_ID + 1


@dataclass(frozen=True)
class MutationSetVocabs:
    passage_to_id: dict[str, int]
    subtype_to_id: dict[str, int]
    serum_name_to_id: dict[str, int]
    query_virus_to_id: dict[str, int]


def normalize_passage(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    text = str(value).strip()
    if not text:
        return "unknown"
    lowered = text.lower().replace("<", "").replace(">", "")
    return "unknown" if lowered in {"none", "nan", "na", "unknown"} else lowered


def normalize_subtype(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def normalize_identity(value: Any) -> str:
    """Return a stable ID key, reserving 0 for missing/unseen identities."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def subtype_column(frame: pd.DataFrame) -> str:
    if "Type" in frame.columns:
        return "Type"
    if "virusType" in frame.columns:
        return "virusType"
    raise ValueError("frame is missing required subtype column: Type or virusType")


def parse_column_list(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def parse_float_list(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, str):
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    return [float(part) for part in value]


def validate_ha_only_frame(frame: pd.DataFrame, source: str | Path) -> None:
    missing = sorted(HA_ONLY_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required HA column(s): {', '.join(missing)}")
    subtype_column(frame)


def filter_frames_by_type(
    frames: dict[str, pd.DataFrame],
    type_filter: str,
) -> dict[str, pd.DataFrame]:
    target = normalize_subtype(type_filter)
    target_key = target.casefold()
    filtered: dict[str, pd.DataFrame] = {}
    observed: set[str] = set()
    for name, frame in frames.items():
        values = frame[subtype_column(frame)].map(normalize_subtype)
        observed.update(value for value in values.tolist() if value != "unknown")
        filtered[name] = frame.loc[
            values.astype(str).str.casefold() == target_key
        ].reset_index(drop=True)
    if filtered.get("train", pd.DataFrame()).empty:
        available = ", ".join(sorted(observed)) if observed else "none"
        raise ValueError(
            f"No training rows remain for --type {target!r}; available: {available}"
        )
    return filtered


def merge_train_valid_for_refit(
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        "train": pd.concat(
            [frames["train"], frames["valid"]],
            ignore_index=True,
        ),
        "valid": frames["valid"].iloc[0:0].copy().reset_index(drop=True),
        "test": frames["test"].copy().reset_index(drop=True),
    }


def load_fixed_split_frames(
    data_dir: Path,
    sample_limit: int | None = None,
    refit_train_valid: bool = False,
    type_filter: str = "",
) -> dict[str, pd.DataFrame]:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    missing = [name for name in SPLIT_FILENAMES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing split file(s): {', '.join(missing)}")
    frames = {
        name.removesuffix(".csv"): pd.read_csv(root / name)
        for name in SPLIT_FILENAMES
    }
    for name, frame in frames.items():
        validate_ha_only_frame(frame, root / f"{name}.csv")
    if type_filter:
        frames = filter_frames_by_type(frames, type_filter)
    if sample_limit is not None:
        frames = {
            name: frame.iloc[:sample_limit].copy().reset_index(drop=True)
            for name, frame in frames.items()
        }
    if refit_train_valid:
        frames = merge_train_valid_for_refit(frames)
    return frames


def make_task_keys(
    frame: pd.DataFrame,
    task_cols: list[str] | tuple[str, ...],
) -> pd.Series:
    missing = [column for column in task_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing serum task column(s): {', '.join(missing)}")
    if frame.empty:
        return pd.Series([], index=frame.index, dtype=str)
    values = frame.loc[:, list(task_cols)].fillna("unknown").astype(str)
    return values.apply(lambda row: "||".join(row.tolist()), axis=1)


def task_overlap_counts(
    frames: dict[str, pd.DataFrame],
    task_cols: list[str],
) -> dict[str, int]:
    keys = {name: set(make_task_keys(frame, task_cols)) for name, frame in frames.items()}
    return {
        "train_valid": len(keys["train"] & keys["valid"]),
        "train_test": len(keys["train"] & keys["test"]),
        "valid_test": len(keys["valid"] & keys["test"]),
    }


def identity_coverage(
    frames: dict[str, pd.DataFrame],
    vocabs: MutationSetVocabs,
) -> dict[str, dict[str, float | int]]:
    """Report which evaluation identities have fitted scalar offsets."""
    result: dict[str, dict[str, float | int]] = {}
    for split_name, frame in frames.items():
        serum_values = (
            frame["serumName"]
            if "serumName" in frame.columns
            else pd.Series("", index=frame.index)
        )
        serum_seen = serum_values.map(
            lambda value: vocabs.serum_name_to_id.get(normalize_identity(value), 0) != 0
        )
        virus_seen = frame["seq_id_c"].map(
            lambda value: vocabs.query_virus_to_id.get(normalize_identity(value), 0) != 0
        )
        result[split_name] = {
            "rows": int(len(frame)),
            "serum_name_seen_rows": int(serum_seen.sum()),
            "serum_name_seen_fraction": float(serum_seen.mean()) if len(frame) else 0.0,
            "query_virus_seen_rows": int(virus_seen.sum()),
            "query_virus_seen_fraction": float(virus_seen.mean()) if len(frame) else 0.0,
        }
    return result


def build_vocabs(
    frames: dict[str, pd.DataFrame],
    use_subtype_feature: bool = True,
    use_output_identity_bias: bool = False,
) -> MutationSetVocabs:
    passages = {"unknown"}
    subtypes = {"unknown"}
    for frame in frames.values():
        passages.update(frame["serumPassCat"].map(normalize_passage).tolist())
        passages.update(frame["virusPassCat"].map(normalize_passage).tolist())
        if use_subtype_feature:
            subtypes.update(frame[subtype_column(frame)].map(normalize_subtype).tolist())
    passage_order = ["unknown"] + sorted(value for value in passages if value != "unknown")
    if use_subtype_feature:
        subtype_order = ["unknown"] + sorted(
            value for value in subtypes if value != "unknown"
        )
        subtype_to_id = {
            value: index for index, value in enumerate(subtype_order)
        }
    else:
        subtype_to_id = {"constant": 0}
    if use_output_identity_bias:
        # Identity effects are fitted effects, not generic categorical
        # embeddings.  Deliberately build these two tables from fit data only;
        # names or viruses first encountered at evaluation use ID 0 (zero bias).
        fit_frame = frames["train"]
        serum_names = sorted(
            value
            for value in fit_frame["serumName"].map(normalize_identity).unique()
            if value
        )
        query_viruses = sorted(
            value
            for value in fit_frame["seq_id_c"].map(normalize_identity).unique()
            if value
        )
        serum_name_to_id = {"": 0, **{value: index + 1 for index, value in enumerate(serum_names)}}
        query_virus_to_id = {"": 0, **{value: index + 1 for index, value in enumerate(query_viruses)}}
    else:
        serum_name_to_id = {"": 0}
        query_virus_to_id = {"": 0}
    return MutationSetVocabs(
        passage_to_id={
            value: index for index, value in enumerate(passage_order)
        },
        subtype_to_id=subtype_to_id,
        serum_name_to_id=serum_name_to_id,
        query_virus_to_id=query_virus_to_id,
    )


def required_ha_embedding_files(frames: dict[str, pd.DataFrame]) -> list[str]:
    combined = pd.concat(list(frames.values()), ignore_index=True)
    sequence_ids = pd.concat(
        [combined["seq_id_a"], combined["seq_id_c"]],
        ignore_index=True,
    ).dropna().astype(str).unique()
    return sorted(f"matrix_{sequence_id}.pt" for sequence_id in sequence_ids)


def validate_embedding_files(embedding_dir: Path, files: list[str]) -> Path:
    root = Path(embedding_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Embedding directory does not exist: {root}")
    missing = [name for name in files if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} HA embedding file(s): {', '.join(missing[:10])}"
        )
    return root


def load_embeddings(
    embedding_dir: Path,
    files: list[str],
    show_progress: bool = True,
) -> dict[str, torch.Tensor]:
    store: dict[str, torch.Tensor] = {}
    for filename in tqdm(
        files,
        desc="loading HA embeddings",
        disable=not show_progress,
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        value = torch.load(
            Path(embedding_dir) / filename,
            map_location="cpu",
            weights_only=False,
        )
        matrix = torch.as_tensor(value).float()
        if matrix.ndim != 2:
            raise ValueError(f"{filename} must contain a 2D embedding matrix")
        store[filename.removesuffix(".pt")] = matrix
    return store


def infer_hidden_size(embeddings: dict[str, torch.Tensor]) -> int:
    if not embeddings:
        raise ValueError("No HA embeddings loaded")
    widths = {int(matrix.shape[1]) for matrix in embeddings.values()}
    if len(widths) != 1:
        raise ValueError(f"HA embeddings have inconsistent widths: {sorted(widths)}")
    return widths.pop()


def encode_aligned_sequence(aligned_sequence: str) -> torch.Tensor:
    text = str(aligned_sequence).strip().upper()
    ids = []
    for residue in text:
        if residue == "-":
            ids.append(GAP_ID)
        else:
            ids.append(AMINO_ACID_TO_ID.get(residue, UNKNOWN_ID))
    return torch.tensor(ids, dtype=torch.long)


def align_embedding_to_sequence(
    matrix: torch.Tensor,
    aligned_sequence: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence = str(aligned_sequence).strip().upper()
    if not sequence:
        raise ValueError("aligned HA sequence must not be empty")
    matrix = torch.as_tensor(matrix).float()
    if matrix.ndim != 2:
        raise ValueError("HA embedding must be a 2D matrix")
    non_gap_count = sum(residue != "-" for residue in sequence)
    if int(matrix.shape[0]) == non_gap_count + 2:
        matrix = matrix[1:-1]
    if int(matrix.shape[0]) != non_gap_count:
        raise ValueError(
            "HA embedding row count does not equal aligned non-gap residue count: "
            f"{matrix.shape[0]} != {non_gap_count}"
        )
    aligned = matrix.new_zeros((len(sequence), int(matrix.shape[1])))
    aligned_mask = matrix.new_ones(len(sequence))
    embedding_mask = matrix.new_zeros(len(sequence))
    source_index = 0
    for position, residue in enumerate(sequence):
        if residue == "-":
            continue
        aligned[position] = matrix[source_index]
        embedding_mask[position] = 1.0
        source_index += 1
    return (
        aligned,
        aligned_mask,
        embedding_mask,
        encode_aligned_sequence(sequence),
    )


def _load_delimited_distance(path: Path, separator: str) -> np.ndarray:
    frame = pd.read_csv(path, sep=separator)
    if frame.shape[1] == frame.shape[0] + 1:
        first_name = str(frame.columns[0]).lower()
        first_values = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
        expected_index = np.arange(len(frame), dtype=float)
        index_like = (
            first_name.startswith("unnamed")
            or (
                first_values.notna().all()
                and np.array_equal(first_values.to_numpy(dtype=float), expected_index)
            )
        )
        if index_like:
            frame = frame.iloc[:, 1:]
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.to_numpy(dtype=np.float32)


def load_ha_distance_matrix(path: Path) -> torch.Tensor:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"HA distance matrix does not exist: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".npy":
        array = np.load(resolved)
    elif suffix == ".npz":
        archive = np.load(resolved)
        keys = list(archive.files)
        if len(keys) != 1:
            raise ValueError("NPZ distance file must contain exactly one array")
        array = archive[keys[0]]
    elif suffix == ".csv":
        array = _load_delimited_distance(resolved, ",")
    elif suffix == ".tsv":
        array = _load_delimited_distance(resolved, "\t")
    else:
        raise ValueError("HA distance matrix must be .csv, .tsv, .npy, or .npz")
    distance = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("HA distance matrix must be square")
    finite = torch.isfinite(distance)
    if bool((distance[finite] < 0).any()):
        raise ValueError("finite HA distances must be non-negative")
    return distance


class SerumMutationSetTaskDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocabs: MutationSetVocabs,
        task_cols: list[str] | None = None,
        max_queries_per_task: int | None = 128,
    ):
        if max_queries_per_task is not None and max_queries_per_task <= 0:
            raise ValueError("max_queries_per_task must be positive or None")
        self.frame = dataframe.reset_index(drop=True).copy()
        self.vocabs = vocabs
        self.task_cols = task_cols or DEFAULT_TASK_COLS
        self.max_queries_per_task = max_queries_per_task
        self.frame["_task_key"] = make_task_keys(self.frame, self.task_cols)
        self.task_groups: list[tuple[str, np.ndarray]] = [
            (str(task_key), group.index.to_numpy())
            for task_key, group in self.frame.groupby("_task_key", sort=False)
        ]
        self.task_chunks: list[tuple[str, np.ndarray]] = []
        self.task_chunk_groups: list[tuple[str, list[int]]] = []
        self._rebuild_task_chunks()

    def _rebuild_task_chunks(self, rng: np.random.Generator | None = None) -> None:
        self.task_chunks = []
        self.task_chunk_groups = []
        for task_key, original_indices in self.task_groups:
            indices = (
                rng.permutation(original_indices)
                if rng is not None
                else original_indices
            )
            chunk_size = self.max_queries_per_task or len(indices)
            chunk_indices: list[int] = []
            for start in range(0, len(indices), chunk_size):
                chunk_indices.append(len(self.task_chunks))
                self.task_chunks.append((task_key, indices[start : start + chunk_size]))
            self.task_chunk_groups.append((task_key, chunk_indices))

    def reshuffle_queries_within_tasks(self, seed: int) -> None:
        """Deterministically redraw Q-sized chunks within every serum task."""
        self._rebuild_task_chunks(np.random.default_rng(seed))

    def __len__(self) -> int:
        return len(self.task_chunks)

    def _passage_id(self, value: Any) -> int:
        return self.vocabs.passage_to_id.get(normalize_passage(value), 0)

    def _subtype_id(self, row: pd.Series, frame: pd.DataFrame) -> int:
        if self.vocabs.subtype_to_id == {"constant": 0}:
            return 0
        return self.vocabs.subtype_to_id.get(
            normalize_subtype(row[subtype_column(frame)]),
            0,
        )

    def _serum_name_id(self, value: Any) -> int:
        return self.vocabs.serum_name_to_id.get(normalize_identity(value), 0)

    def _query_virus_id(self, value: Any) -> int:
        return self.vocabs.query_virus_to_id.get(normalize_identity(value), 0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_key, row_indices = self.task_chunks[index]
        task_frame = self.frame.loc[row_indices].reset_index(drop=True)
        first = task_frame.iloc[0]
        reference_ids = task_frame["seq_id_a"].astype(str).unique().tolist()
        reference_sequences = task_frame["serumHA"].astype(str).unique().tolist()
        if len(reference_ids) != 1 or len(reference_sequences) != 1:
            raise ValueError(
                f"serum task {task_key!r} has inconsistent reference HA identity"
            )
        query_sequences = task_frame["virusHA"].astype(str).tolist()
        reference_length = len(reference_sequences[0])
        if any(len(sequence) != reference_length for sequence in query_sequences):
            raise ValueError(
                f"serum task {task_key!r} has reference/query alignment length mismatch"
            )
        serum_passage = self._passage_id(first["serumPassCat"])
        query_passage = torch.tensor(
            [self._passage_id(value) for value in task_frame["virusPassCat"]],
            dtype=torch.long,
        )
        passage_pair = serum_passage * len(self.vocabs.passage_to_id) + query_passage
        return {
            "task_key": task_key,
            "reference_ha_key": f"matrix_{reference_ids[0]}",
            "query_ha_keys": [
                f"matrix_{value}" for value in task_frame["seq_id_c"].astype(str)
            ],
            "reference_ha_aligned": reference_sequences[0],
            "query_ha_aligned": query_sequences,
            "serum_passage": torch.tensor(serum_passage, dtype=torch.long),
            "query_passage": query_passage,
            "passage_pair": passage_pair.long(),
            "subtype": torch.tensor(
                self._subtype_id(first, task_frame),
                dtype=torch.long,
            ),
            "serum_name": torch.tensor(
                self._serum_name_id(first.get("serumName", "")), dtype=torch.long
            ),
            "query_virus": torch.tensor(
                [self._query_virus_id(value) for value in task_frame["seq_id_c"]],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                task_frame["label"].to_numpy(dtype=float),
                dtype=torch.float32,
            ),
            "row_indices": torch.tensor(row_indices, dtype=torch.long),
            "query_meta": task_frame.drop(columns=["_task_key"]).reset_index(drop=True),
        }


def _fetch_embedding(store: Any, key: str) -> torch.Tensor:
    if isinstance(store, dict):
        return store[key]
    value = store.get(key)
    if value is None:
        raise KeyError(key)
    return value


class AlignedEmbeddingCache:
    """Memoize each sequence's gap-aligned representation within one run.

    The source cache contains ungapped residue embeddings. A serum task expands
    them repeatedly to shared alignment coordinates, including masks and
    amino-acid IDs. That operation is deterministic for an embedding key and
    aligned sequence, so cache it without changing the batch interface.
    """

    def __init__(self) -> None:
        self._values: dict[
            tuple[str, str],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self.hits = 0
        self.misses = 0

    def get(
        self,
        embeddings: Any,
        key: str,
        aligned_sequence: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = (str(key), str(aligned_sequence).strip().upper())
        value = self._values.get(cache_key)
        if value is not None:
            self.hits += 1
            return value
        self.misses += 1
        value = align_embedding_to_sequence(
            _fetch_embedding(embeddings, key),
            aligned_sequence,
        )
        self._values[cache_key] = value
        return value


def _aligned_embedding(
    embeddings: Any,
    key: str,
    aligned_sequence: str,
    aligned_cache: AlignedEmbeddingCache | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if aligned_cache is None:
        return align_embedding_to_sequence(
            _fetch_embedding(embeddings, key),
            aligned_sequence,
        )
    return aligned_cache.get(embeddings, key, aligned_sequence)


def collate_mutation_set_tasks(
    items: list[dict[str, Any]],
    embeddings: Any,
    aligned_cache: AlignedEmbeddingCache | None = None,
) -> tuple[SerumMutationSetBatch, list[dict[str, Any]]]:
    if len(items) != 1:
        raise ValueError("Serum mutation-set collator requires batch_size=1")
    item = items[0]
    reference = _aligned_embedding(
        embeddings,
        item["reference_ha_key"],
        item["reference_ha_aligned"],
        aligned_cache,
    )
    queries = [
        _aligned_embedding(
            embeddings,
            key,
            sequence,
            aligned_cache,
        )
        for key, sequence in zip(
            item["query_ha_keys"],
            item["query_ha_aligned"],
        )
    ]
    reference_matrix, reference_aligned_mask, reference_embedding_mask, reference_aa = reference
    query_matrix = torch.stack([value[0] for value in queries], dim=0)
    query_aligned_mask = torch.stack([value[1] for value in queries], dim=0)
    query_embedding_mask = torch.stack([value[2] for value in queries], dim=0)
    query_aa = torch.stack([value[3] for value in queries], dim=0)
    query_count = len(queries)
    batch = SerumMutationSetBatch(
        reference_ha=reference_matrix.unsqueeze(0),
        query_ha=query_matrix.unsqueeze(0),
        reference_aa=reference_aa.unsqueeze(0),
        query_aa=query_aa.unsqueeze(0),
        reference_aligned_mask=reference_aligned_mask.unsqueeze(0),
        query_aligned_mask=query_aligned_mask.unsqueeze(0),
        reference_embedding_mask=reference_embedding_mask.unsqueeze(0),
        query_embedding_mask=query_embedding_mask.unsqueeze(0),
        serum_passage=item["serum_passage"].view(1),
        query_passage=item["query_passage"].view(1, -1),
        passage_pair=item["passage_pair"].view(1, -1),
        subtype=item["subtype"].view(1),
        serum_name=item["serum_name"].view(1),
        query_virus=item["query_virus"].view(1, -1),
        labels=item["labels"].view(1, -1),
        query_mask=torch.ones(1, query_count, dtype=torch.float32),
    )
    return batch, items


def move_mutation_batch(
    batch: SerumMutationSetBatch,
    device: torch.device,
) -> SerumMutationSetBatch:
    values: dict[str, Any] = {}
    for field in fields(SerumMutationSetBatch):
        value = getattr(batch, field.name)
        values[field.name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return SerumMutationSetBatch(**values)


def build_loader(
    frame: pd.DataFrame,
    vocabs: MutationSetVocabs,
    embeddings: Any,
    batch_size: int,
    shuffle: bool,
    max_queries_per_task: int | None,
    task_cols: list[str],
    aligned_cache: AlignedEmbeddingCache | None = None,
) -> DataLoader:
    if batch_size != 1:
        raise ValueError("Use --batch-size 1 for variable-size serum tasks")
    dataset = SerumMutationSetTaskDataset(
        frame,
        vocabs,
        task_cols=task_cols,
        max_queries_per_task=max_queries_per_task,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=partial(
            collate_mutation_set_tasks,
            embeddings=embeddings,
            aligned_cache=aligned_cache or AlignedEmbeddingCache(),
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SerumMutationSet-Minus.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--ha-distance-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--type", dest="type_filter", default="")
    parser.add_argument("--serum-task-cols", default=",".join(DEFAULT_TASK_COLS))
    parser.add_argument(
        "--refit-train-valid",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--sample-limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-queries-per-task", type=int, default=128)
    parser.add_argument(
        "--shuffle-queries-within-task-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For training only, deterministically reshuffle queries within "
            "each serum task before forming max-query chunks every epoch."
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save-epoch", type=int, nargs="+", default=None,
                        help="Epoch number(s) whose checkpoints are retained; default: final epoch.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=-1)
    parser.add_argument("--site-dim", type=int, default=64)
    parser.add_argument(
        "--site-bottleneck-dim",
        type=int,
        default=0,
        help=(
            "Optional low-rank bottleneck for the site adapter; 0 keeps the "
            "direct hidden-size-to-site-dim projection."
        ),
    )
    parser.add_argument("--background-dim", type=int, default=128)
    parser.add_argument(
        "--direct-background",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the mean-pooled site projection directly as the reference "
            "background; requires --background-dim to equal --site-dim."
        ),
    )
    parser.add_argument("--mutation-dim", type=int, default=128)
    parser.add_argument("--position-dim", type=int, default=32)
    parser.add_argument("--amino-acid-dim", type=int, default=16)
    parser.add_argument("--presence-dim", type=int, default=4)
    parser.add_argument("--theta-dim", type=int, default=128)
    parser.add_argument("--passage-dim", type=int, default=8)
    parser.add_argument("--subtype-dim", type=int, default=8)
    parser.add_argument(
        "--use-subtype-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-passage-pair-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mutation-attention-heads", type=int, default=4)
    parser.add_argument("--mutation-attention-layers", type=int, default=1)
    parser.add_argument(
        "--bypass-mutation-transformer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the mutation Transformer initialized for seed-compatible "
            "ablation, but bypass it in the forward pass."
        ),
    )
    parser.add_argument(
        "--use-background-to-mutation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add the reference background projection to mutation tokens and "
            "the null-mutation token before pooling."
        ),
    )
    parser.add_argument("--mutation-ffn-dim", type=int, default=256)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--attention-alpha-init", type=float, default=0.05)
    parser.add_argument("--attention-tau-init", type=float, default=8.0)
    parser.add_argument("--predictor-hidden-dim", type=int, default=256)
    parser.add_argument("--predictor-dropout", type=float, default=0.1)
    parser.add_argument(
        "--zero-init-film",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize FiLM gamma/beta to zero so conditioning starts as identity.",
    )
    parser.add_argument(
        "--use-film-beta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the additive FiLM beta shift; disable it for gamma-only FiLM "
            "while retaining the same FiLM layer and parameter shapes."
        ),
    )
    parser.add_argument(
        "--use-pool-mutation-count",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include log mutation count inside mutation pooling; the predictor "
            "always retains its separate count feature."
        ),
    )
    parser.add_argument(
        "--use-attention-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use learned attention pooling in addition to mean mutation pooling.",
    )
    parser.add_argument(
        "--use-predictor-mutation-count",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Provide log mutation count to the predictor after mutation pooling; "
            "when disabled, retain the input dimension but fill this feature with zero."
        ),
    )
    parser.add_argument(
        "--use-output-identity-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add train-vocabulary serumName and query-virus scalar offsets only "
            "to the final prediction. Unseen evaluation identities use fixed zero bias."
        ),
    )
    parser.add_argument("--label-weight-thresholds", default="2,4,6")
    parser.add_argument(
        "--label-weight-threshold-mode",
        choices=("fixed", "quantile"),
        default="fixed",
    )
    parser.add_argument("--label-weight-quantiles", default="0.35,0.75,0.95")
    parser.add_argument("--label-weight-values", default="1,1.3,1.8,2.5")
    parser.add_argument(
        "--within-serum-rank-loss-weight",
        dest="rank_loss_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--within-serum-rank-margin",
        dest="rank_loss_margin",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--within-serum-rank-min-label-delta",
        dest="rank_loss_min_label_delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--task-bias-loss-weight",
        type=float,
        default=0.0,
        help="Weight for squared mean signed prediction error within each task chunk.",
    )
    parser.add_argument(
        "--full-task-bias-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace independent per-chunk bias penalties with one exact bias "
            "penalty over every query in the complete serum task. Training "
            "uses two Q-sized passes and one optimizer step per serum task."
        ),
    )
    parser.add_argument(
        "--skip-test-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-cache-gb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.save_epoch is None:
        args.save_epoch = [args.epochs]
    else:
        args.save_epoch = sorted(set(args.save_epoch))
        invalid_save_epochs = [epoch for epoch in args.save_epoch if epoch < 1 or epoch > args.epochs]
        if invalid_save_epochs:
            parser.error("--save-epoch values must be between 1 and --epochs: " + ", ".join(map(str, invalid_save_epochs)))
    if args.lr_min < 0:
        parser.error("--lr-min must be non-negative")
    if args.early_stopping_patience < -1:
        parser.error("--early-stopping-patience must be -1 or greater")
    if args.task_bias_loss_weight < 0:
        parser.error("--task-bias-loss-weight must be non-negative")
    if args.full_task_bias_loss and args.task_bias_loss_weight <= 0:
        parser.error(
            "--full-task-bias-loss requires --task-bias-loss-weight to be positive"
        )
    if args.site_bottleneck_dim < 0:
        parser.error("--site-bottleneck-dim must be non-negative")
    if args.direct_background and args.background_dim != args.site_dim:
        parser.error(
            "--direct-background requires --background-dim to equal --site-dim"
        )
    if args.type_filter:
        args.use_subtype_feature = False
    if not args.use_subtype_feature:
        args.subtype_dim = 0
    if args.max_queries_per_task <= 0:
        args.max_queries_per_task = None
    return args
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_label_weight_thresholds(
    frames: dict[str, pd.DataFrame],
    mode: str,
    thresholds_arg: str,
    quantiles_arg: str,
) -> list[float]:
    if mode == "fixed":
        return parse_float_list(thresholds_arg)
    quantiles = parse_float_list(quantiles_arg)
    if not quantiles or any(value <= 0.0 or value >= 1.0 for value in quantiles):
        raise ValueError("label-weight quantiles must be in (0, 1)")
    if quantiles != sorted(quantiles):
        raise ValueError("label-weight quantiles must be sorted")
    return [
        float(value)
        for value in pd.to_numeric(frames["train"]["label"], errors="raise")
        .quantile(quantiles)
        .tolist()
    ]


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    return _correlation(
        pd.Series(first).rank(method="average").to_numpy(),
        pd.Series(second).rank(method="average").to_numpy(),
    )


def serum_regression_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    keys = (
        "pooled_mae",
        "pooled_mse",
        "pooled_pearson",
        "pooled_spearman",
        "per_serum_mae_mean",
        "per_serum_mae_median",
        "within_serum_pearson_mean",
        "within_serum_spearman_mean",
        "serum_bias_mean",
        "serum_bias_abs_mean",
    )
    if predictions.empty:
        return {key: 0.0 for key in keys}
    label = predictions["label"].to_numpy(dtype=float)
    mean = predictions["mean"].to_numpy(dtype=float)
    per_task_mae: list[float] = []
    residual = mean - label
    task_bias: list[float] = []
    within_pearson: list[float] = []
    within_spearman: list[float] = []
    for _, group in predictions.groupby("task_key"):
        grouped_label = group["label"].to_numpy(dtype=float)
        grouped_mean = group["mean"].to_numpy(dtype=float)
        grouped_residual = grouped_mean - grouped_label
        per_task_mae.append(float(np.mean(np.abs(grouped_residual))))
        task_bias.append(float(np.mean(grouped_residual)))
        if len(group) >= 2:
            within_pearson.append(_correlation(grouped_label, grouped_mean))
            within_spearman.append(_spearman(grouped_label, grouped_mean))
    return {
        "pooled_mae": float(np.mean(np.abs(residual))),
        "pooled_mse": float(np.mean(residual**2)),
        "pooled_pearson": _correlation(label, mean),
        "pooled_spearman": _spearman(label, mean),
        "per_serum_mae_mean": float(np.mean(per_task_mae)),
        "per_serum_mae_median": float(np.median(per_task_mae)),
        "within_serum_pearson_mean": float(np.mean(within_pearson)) if within_pearson else 0.0,
        "within_serum_spearman_mean": float(np.mean(within_spearman)) if within_spearman else 0.0,
        "serum_bias_mean": float(np.mean(task_bias)),
        "serum_bias_abs_mean": float(np.mean(np.abs(task_bias))),
    }


def subtype_regression_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for subtype in ("H1N1", "H3N2"):
        if predictions.empty:
            subset = predictions
        else:
            values = predictions[subtype_column(predictions)].map(normalize_subtype)
            subset = predictions.loc[
                values.astype(str).str.casefold() == subtype.casefold()
            ]
        result[subtype.casefold()] = serum_regression_metrics(subset)
    return result


def flatten_epoch_metrics(
    epoch_metrics: dict[str, Any],
    decimals: int = 4,
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for name, nested in value.items():
                walk(f"{prefix}_{name}" if prefix else name, nested)
        elif isinstance(value, (float, np.floating)):
            flattened[prefix] = round(float(value), decimals)
        else:
            flattened[prefix] = value

    for name, value in epoch_metrics.items():
        walk(name, value)
    return flattened


def append_metrics_csv(path: Path, epoch_metrics: dict[str, Any]) -> None:
    pd.DataFrame([flatten_epoch_metrics(epoch_metrics)]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def write_rounded_predictions(predictions: pd.DataFrame, path: Path) -> None:
    rounded = predictions.copy()
    columns = rounded.select_dtypes(include=["floating"]).columns
    rounded.loc[:, columns] = rounded.loc[:, columns].round(4)
    rounded.to_csv(path, index=False)


def prepare_output_dir(output_dir: Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    return {
        "root": root,
        "checkpoints": checkpoints,
        "config": root / "run_config.json",
        "metrics": root / "metrics.csv",
        "metrics_by_subtype": root / "metrics_by_subtype.csv",
        "log": root / "log.txt",
    }


def _task_residual_sum_and_count(
    out: dict[str, torch.Tensor],
    batch: SerumMutationSetBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.labels is None:
        raise ValueError("full-task bias training requires labels")
    residual = out["mean"] - batch.labels.float().to(out["mean"].device)
    mask = (
        batch.query_mask.float().to(residual.device)
        if batch.query_mask is not None
        else torch.ones_like(residual)
    )
    return (residual * mask).sum(), mask.sum()


def _full_task_bias_gradient_term(
    chunk_residual_sum: torch.Tensor,
    full_task_bias: torch.Tensor,
    full_task_query_count: float,
    weight: float,
) -> torch.Tensor:
    """Return a surrogate with the exact gradient of weight * full_bias**2."""
    if full_task_query_count <= 0:
        raise ValueError("full task must contain at least one query")
    return (
        2.0
        * float(weight)
        * full_task_bias.detach()
        * chunk_residual_sum
        / float(full_task_query_count)
    )


def _capture_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else None
    )
    return cpu_state, cuda_state


def _restore_rng_state(
    state: tuple[torch.Tensor, torch.Tensor | None],
    device: torch.device,
) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def train_one_epoch_full_task_bias(
    model: SerumMutationSetMinusModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_order_seed: int,
    progress_bar: tqdm | None = None,
) -> float:
    """Train with one exact serum-level bias penalty using Q-sized microbatches."""
    dataset = loader.dataset
    if not isinstance(dataset, SerumMutationSetTaskDataset):
        raise TypeError("full-task bias training requires SerumMutationSetTaskDataset")
    if not callable(loader.collate_fn):
        raise TypeError("full-task bias training requires a callable collate function")
    weight = float(model.config.task_bias_loss_weight)
    if weight <= 0:
        raise ValueError("full-task bias training requires a positive bias weight")

    model.train()
    task_losses: list[float] = []
    order = np.random.default_rng(task_order_seed).permutation(
        len(dataset.task_chunk_groups)
    )
    for task_group_index in order:
        _, chunk_indices = dataset.task_chunk_groups[int(task_group_index)]
        rng_states: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        residual_sum = 0.0
        query_count = 0.0

        # First pass: measure the complete-task bias at one fixed parameter state.
        with torch.no_grad():
            for chunk_index in chunk_indices:
                batch, _ = loader.collate_fn([dataset[chunk_index]])
                moved = move_mutation_batch(batch, device)
                rng_states.append(_capture_rng_state(device))
                out = model(moved)
                chunk_sum, chunk_count = _task_residual_sum_and_count(out, moved)
                residual_sum += float(chunk_sum.item())
                query_count += float(chunk_count.item())
        if query_count <= 0:
            raise ValueError("full task must contain at least one unmasked query")
        full_task_bias = torch.tensor(
            residual_sum / query_count,
            dtype=torch.float32,
            device=device,
        )

        # Second pass: replay dropout and accumulate the exact full-task gradient.
        optimizer.zero_grad(set_to_none=True)
        base_loss_sum = 0.0
        for chunk_index, rng_state in zip(chunk_indices, rng_states):
            batch, _ = loader.collate_fn([dataset[chunk_index]])
            moved = move_mutation_batch(batch, device)
            _restore_rng_state(rng_state, device)
            out = model(moved)
            base_loss = (
                out["huber_loss"]
                - weight * out["task_bias_loss"]
            )
            chunk_sum, _ = _task_residual_sum_and_count(out, moved)
            loss = (
                base_loss / len(chunk_indices)
                + _full_task_bias_gradient_term(
                    chunk_sum,
                    full_task_bias,
                    query_count,
                    weight,
                )
            )
            loss.backward()
            base_loss_sum += float(base_loss.item())
            if progress_bar is not None:
                progress_bar.update(1)
        optimizer.step()
        task_losses.append(
            base_loss_sum / len(chunk_indices)
            + weight * float(full_task_bias.square().item())
        )
    return float(np.mean(task_losses)) if task_losses else 0.0


def train_one_epoch(
    model: SerumMutationSetMinusModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    progress_bar: tqdm | None = None,
    full_task_bias_loss: bool = False,
    task_order_seed: int = 0,
) -> float:
    if full_task_bias_loss:
        return train_one_epoch_full_task_bias(
            model,
            loader,
            optimizer,
            device,
            task_order_seed,
            progress_bar,
        )
    model.train()
    losses: list[float] = []
    for batch, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        out = model(move_mutation_batch(batch, device))
        loss = out["huber_loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if progress_bar is not None:
            progress_bar.update(1)
    return float(np.mean(losses)) if losses else 0.0


def evaluate_model(
    model: SerumMutationSetMinusModel,
    loader: DataLoader,
    device: torch.device,
    progress_bar: tqdm | None = None,
    full_task_bias_loss: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    task_base_losses: dict[str, list[float]] = defaultdict(list)
    task_residual_sums: dict[str, float] = defaultdict(float)
    task_query_counts: dict[str, float] = defaultdict(float)
    with torch.no_grad():
        for batch, items in loader:
            moved = move_mutation_batch(batch, device)
            out = model(moved)
            if full_task_bias_loss:
                task_key = str(items[0]["task_key"])
                weight = float(model.config.task_bias_loss_weight)
                base_loss = out["huber_loss"] - weight * out["task_bias_loss"]
                residual_sum, query_count = _task_residual_sum_and_count(out, moved)
                task_base_losses[task_key].append(float(base_loss.item()))
                task_residual_sums[task_key] += float(residual_sum.item())
                task_query_counts[task_key] += float(query_count.item())
            else:
                losses.append(float(out["huber_loss"].item()))
            metadata = items[0]["query_meta"]
            fields_to_record = {
                name: out[name].detach().cpu().reshape(-1).numpy()
                for name in (
                    "mean",
                    "self_score",
                    "query_score",
                    "mutation_count",
                )
            }
            labels = moved.labels.detach().cpu().reshape(-1).numpy()
            for index, label in enumerate(labels):
                row = metadata.iloc[index].to_dict()
                row.update(
                    {
                        "task_key": items[0]["task_key"],
                        "label": float(label),
                        **{
                            name: float(values[index])
                            for name, values in fields_to_record.items()
                        },
                    }
                )
                rows.append(row)
            if progress_bar is not None:
                progress_bar.update(1)
    prediction_frame = pd.DataFrame(rows)
    metrics = serum_regression_metrics(prediction_frame)
    if full_task_bias_loss:
        weight = float(model.config.task_bias_loss_weight)
        losses = [
            float(np.mean(task_base_losses[task_key]))
            + weight
            * (task_residual_sums[task_key] / task_query_counts[task_key]) ** 2
            for task_key in task_base_losses
        ]
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, prediction_frame


def _validate_distance_coverage(
    frames: dict[str, pd.DataFrame],
    distance_matrix: torch.Tensor,
) -> None:
    max_alignment_length = max(
        len(str(sequence))
        for frame in frames.values()
        for column in ("serumHA", "virusHA")
        for sequence in frame[column].tolist()
    )
    if max_alignment_length > int(distance_matrix.shape[0]):
        raise ValueError(
            "HA distance matrix does not cover aligned HA positions: "
            f"{distance_matrix.shape[0]} < {max_alignment_length}"
        )


def _build_model_config(
    args: argparse.Namespace,
    hidden_size: int,
    vocabs: MutationSetVocabs,
    distance_matrix: torch.Tensor,
    label_thresholds: list[float],
    label_values: list[float],
) -> SerumMutationSetConfig:
    return SerumMutationSetConfig(
        hidden_size=hidden_size,
        site_dim=args.site_dim,
        site_bottleneck_dim=args.site_bottleneck_dim,
        background_dim=args.background_dim,
        mutation_dim=args.mutation_dim,
        position_dim=args.position_dim,
        amino_acid_dim=args.amino_acid_dim,
        presence_dim=args.presence_dim,
        max_position_embeddings=int(distance_matrix.shape[0]),
        amino_acid_vocab_size=AMINO_ACID_VOCAB_SIZE,
        passage_vocab_size=len(vocabs.passage_to_id),
        passage_pair_vocab_size=len(vocabs.passage_to_id) ** 2,
        use_passage_pair_feature=args.use_passage_pair_feature,
        passage_dim=args.passage_dim,
        subtype_vocab_size=len(vocabs.subtype_to_id),
        subtype_dim=args.subtype_dim,
        theta_dim=args.theta_dim,
        mutation_attention_heads=args.mutation_attention_heads,
        mutation_attention_layers=args.mutation_attention_layers,
        mutation_ffn_dim=args.mutation_ffn_dim,
        attention_dropout=args.attention_dropout,
        attention_alpha_init=args.attention_alpha_init,
        attention_tau_init=args.attention_tau_init,
        predictor_hidden_dim=args.predictor_hidden_dim,
        predictor_dropout=args.predictor_dropout,
        label_weight_thresholds=tuple(label_thresholds),
        label_weight_values=tuple(label_values),
        rank_loss_weight=args.rank_loss_weight,
        rank_loss_margin=args.rank_loss_margin,
        rank_loss_min_label_delta=args.rank_loss_min_label_delta,
        zero_init_film=args.zero_init_film,
        use_film_beta=args.use_film_beta,
        use_pool_mutation_count=args.use_pool_mutation_count,
        use_attention_pool=args.use_attention_pool,
        use_predictor_mutation_count=args.use_predictor_mutation_count,
        use_background_to_mutation=args.use_background_to_mutation,
        bypass_mutation_transformer=args.bypass_mutation_transformer,
        task_bias_loss_weight=args.task_bias_loss_weight,
        direct_background=args.direct_background,
        use_output_identity_bias=args.use_output_identity_bias,
        serum_name_vocab_size=len(vocabs.serum_name_to_id),
        query_virus_vocab_size=len(vocabs.query_virus_to_id),
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    task_cols = parse_column_list(args.serum_task_cols)
    sample_limit = None if args.sample_limit < 0 else int(args.sample_limit)
    frames = load_fixed_split_frames(
        args.data_dir,
        sample_limit=sample_limit,
        refit_train_valid=args.refit_train_valid,
        type_filter=args.type_filter,
    )
    distance_matrix = load_ha_distance_matrix(args.ha_distance_matrix)
    _validate_distance_coverage(frames, distance_matrix)
    embedding_files = required_ha_embedding_files(frames)
    embedding_dir = validate_embedding_files(args.embedding_dir, embedding_files)
    embeddings = load_embeddings(
        embedding_dir,
        embedding_files,
        show_progress=args.progress,
    )
    hidden_size = infer_hidden_size(embeddings)
    vocabs = build_vocabs(
        frames,
        use_subtype_feature=args.use_subtype_feature,
        use_output_identity_bias=args.use_output_identity_bias,
    )
    label_thresholds = resolve_label_weight_thresholds(
        frames,
        args.label_weight_threshold_mode,
        args.label_weight_thresholds,
        args.label_weight_quantiles,
    )
    label_values = parse_float_list(args.label_weight_values)
    if len(label_values) != len(label_thresholds) + 1:
        raise ValueError(
            "--label-weight-values must have exactly one more entry than thresholds"
        )
    paths = prepare_output_dir(args.output_dir)
    set_seed(args.seed)
    device = torch.device(args.device)
    embedding_source: Any = embeddings
    if args.gpu_cache_gb > 0:
        if device.type != "cuda":
            raise ValueError("--gpu-cache-gb requires a CUDA device")
        embedding_source = GpuEmbeddingCache(
            cpu_store=embeddings,
            device=device,
            max_bytes=int(args.gpu_cache_gb * 1024**3),
        )
    aligned_cache = AlignedEmbeddingCache()
    loaders = {
        name: build_loader(
            frame,
            vocabs,
            embedding_source,
            batch_size=args.batch_size,
            shuffle=name == "train",
            max_queries_per_task=args.max_queries_per_task,
            task_cols=task_cols,
            aligned_cache=aligned_cache,
        )
        for name, frame in frames.items()
    }
    model_config = _build_model_config(
        args,
        hidden_size,
        vocabs,
        distance_matrix,
        label_thresholds,
        label_values,
    )
    model = SerumMutationSetMinusModel(model_config, distance_matrix).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_min,
        )
        if args.lr_scheduler == "cosine"
        else None
    )
    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "SerumMutationSet-Minus",
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "embedding_dir": str(embedding_dir),
        "ha_distance_matrix": str(Path(args.ha_distance_matrix).expanduser().resolve()),
        "distance_matrix_shape": list(distance_matrix.shape),
        "output_dir": str(paths["root"]),
        "type_filter": args.type_filter,
        "task_cols": task_cols,
        "refit_train_valid": bool(args.refit_train_valid),
        "task_overlap_counts": task_overlap_counts(frames, task_cols),
        "row_counts": {name: len(frame) for name, frame in frames.items()},
        "training_config": {
            "batch_size": args.batch_size,
            "max_queries_per_task": args.max_queries_per_task,
            "shuffle_queries_within_task_each_epoch": (
                args.shuffle_queries_within_task_each_epoch
            ),
            "full_task_bias_loss": args.full_task_bias_loss,
            "optimizer_step_unit": (
                "serum_task" if args.full_task_bias_loss else "query_chunk"
            ),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "lr_scheduler": args.lr_scheduler,
            "lr_min": args.lr_min,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "model_config": model_config.__dict__,
        "passage_to_id": vocabs.passage_to_id,
        "subtype_to_id": vocabs.subtype_to_id,
        "serum_name_to_id": vocabs.serum_name_to_id,
        "query_virus_to_id": vocabs.query_virus_to_id,
        "identity_coverage": identity_coverage(frames, vocabs),
        "loss": "weighted_huber",
        "epochs": args.epochs,
        "seed": args.seed,
    }
    paths["config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["log"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    best_valid = float("inf")
    best_result: dict[str, Any] = {}
    epochs_without_improvement = 0
    test_steps = 0 if args.skip_test_eval else len(loaders["test"])
    total_steps = args.epochs * (
        len(loaders["train"]) + len(loaders["valid"]) + test_steps
    )
    with tqdm(
        total=total_steps,
        desc="training",
        disable=not args.progress,
        dynamic_ncols=True,
        file=sys.stdout,
    ) as progress_bar:
        for epoch in range(args.epochs):
            if args.shuffle_queries_within_task_each_epoch:
                train_dataset = loaders["train"].dataset
                if not isinstance(train_dataset, SerumMutationSetTaskDataset):
                    raise TypeError("training loader must use SerumMutationSetTaskDataset")
                train_dataset.reshuffle_queries_within_tasks(args.seed + epoch)
            train_loss = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                device,
                progress_bar,
                full_task_bias_loss=args.full_task_bias_loss,
                task_order_seed=args.seed + epoch,
            )
            if args.refit_train_valid:
                valid_metrics = serum_regression_metrics(pd.DataFrame())
                valid_metrics["loss"] = 0.0
                valid_predictions = pd.DataFrame()
            else:
                valid_metrics, valid_predictions = evaluate_model(
                    model,
                    loaders["valid"],
                    device,
                    progress_bar,
                    full_task_bias_loss=args.full_task_bias_loss,
                )
            if args.skip_test_eval:
                test_metrics = serum_regression_metrics(pd.DataFrame())
                test_metrics["loss"] = 0.0
                test_predictions = pd.DataFrame()
            else:
                test_metrics, test_predictions = evaluate_model(
                    model,
                    loaders["test"],
                    device,
                    progress_bar,
                    full_task_bias_loss=args.full_task_bias_loss,
                )
            epoch_result = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "valid": valid_metrics,
                "test": test_metrics,
            }
            append_metrics_csv(paths["metrics"], epoch_result)
            if not args.type_filter.strip():
                append_metrics_csv(
                    paths["metrics_by_subtype"],
                    {
                        "epoch": epoch + 1,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "train_loss": train_loss,
                        "valid": subtype_regression_metrics(valid_predictions),
                        "test": subtype_regression_metrics(test_predictions),
                    },
                )
            if epoch + 1 in args.save_epoch:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": model_config.__dict__,
                        "passage_to_id": vocabs.passage_to_id,
                        "subtype_to_id": vocabs.subtype_to_id,
                        "serum_name_to_id": vocabs.serum_name_to_id,
                        "query_virus_to_id": vocabs.query_virus_to_id,
                        "task_cols": task_cols,
                        "epoch": epoch + 1,
                        "valid_metrics": valid_metrics,
                        "refit_train_valid": bool(args.refit_train_valid),
                    },
                    paths["checkpoints"] / f"epoch_{epoch + 1:04d}.pth",
                )
            final_refit_epoch = args.refit_train_valid and epoch + 1 == args.epochs
            best_valid_epoch = (
                not args.refit_train_valid
                and valid_metrics["pooled_mse"] < best_valid
            )
            if final_refit_epoch or best_valid_epoch:
                best_valid = valid_metrics["pooled_mse"]
                best_result = epoch_result
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": model_config.__dict__,
                        "passage_to_id": vocabs.passage_to_id,
                        "subtype_to_id": vocabs.subtype_to_id,
                        "task_cols": task_cols,
                        "epoch": epoch + 1,
                        "valid_metrics": valid_metrics,
                        "refit_train_valid": bool(args.refit_train_valid),
                    },
                    paths["checkpoints"] / "best_model.pth",
                )
                if not valid_predictions.empty:
                    write_rounded_predictions(
                        valid_predictions,
                        paths["root"] / "predictions_valid.csv",
                    )
                if not test_predictions.empty:
                    write_rounded_predictions(
                        test_predictions,
                        paths["root"] / "predictions_test.csv",
                    )
            elif not args.refit_train_valid:
                epochs_without_improvement += 1
            if scheduler is not None:
                scheduler.step()
            if (
                args.early_stopping_patience >= 0
                and not args.refit_train_valid
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                break
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "best": best_result,
    }


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
