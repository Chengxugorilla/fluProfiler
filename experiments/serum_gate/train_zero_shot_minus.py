#!/usr/bin/env python3
"""
Train SerumGate-Minus for serum-level zero-shot antigenic prediction.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
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

from experiment_tools import GpuEmbeddingCache, generate_matrix_on_device  # noqa: E402
from fluprofiler.features.na_glycan_features import na_head_glycan_mismatch  # noqa: E402
from fluprofiler.models.serum_gate_minus_model import (  # noqa: E402
    SerumGateBatch,
    SerumGateMinusConfig,
    SerumGateMinusModel,
)


SPLIT_FILENAMES = ("train.csv", "valid.csv", "test.csv")
DEFAULT_TASK_COLS = ["seq_id_a", "seq_id_b", "serumPassCat"]
REQUIRED_COLUMNS = {
    "seq_id_a",
    "seq_id_b",
    "seq_id_c",
    "seq_id_d",
    "seq_b",
    "seq_d",
    "serumPassCat",
    "virusPassCat",
    "label",
}
ALIGNED_HA_COLUMNS = {"serumHA", "virusHA"}


@dataclass(frozen=True)
class SerumGateVocabs:
    passage_to_id: dict[str, int]
    subtype_to_id: dict[str, int]


class NullWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def normalize_passage(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    text = str(value).strip()
    if not text:
        return "unknown"
    lowered = text.lower().replace("<", "").replace(">", "")
    if lowered in {"egg", "cell", "both"}:
        return lowered
    if lowered in {"none", "nan", "na", "unknown"}:
        return "unknown"
    return lowered


def normalize_subtype(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def parse_column_list(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def parse_float_list(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, str):
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    return [float(part) for part in value]


def resolve_label_weight_thresholds(
    frames: dict[str, pd.DataFrame],
    mode: str,
    thresholds_arg: str | list[float] | tuple[float, ...],
    quantiles_arg: str | list[float] | tuple[float, ...],
) -> list[float]:
    if mode == "fixed":
        return parse_float_list(thresholds_arg)
    if mode != "quantile":
        raise ValueError("--label-weight-threshold-mode must be fixed or quantile")

    quantiles = parse_float_list(quantiles_arg)
    if not quantiles:
        raise ValueError("--label-weight-quantiles must contain at least one quantile")
    if any(quantile <= 0.0 or quantile >= 1.0 for quantile in quantiles):
        raise ValueError("--label-weight-quantiles must be in (0, 1)")
    if quantiles != sorted(quantiles):
        raise ValueError("--label-weight-quantiles must be sorted in ascending order")
    train_frame = frames.get("train")
    if train_frame is None or train_frame.empty:
        raise ValueError("Cannot compute quantile label weights from an empty training frame")
    labels = pd.to_numeric(train_frame["label"], errors="raise")
    return [float(value) for value in labels.quantile(quantiles).tolist()]


def subtype_column(frame: pd.DataFrame) -> str:
    return "Type" if "Type" in frame.columns else "virusType"


def filter_frames_by_type(frames: dict[str, pd.DataFrame], type_filter: str) -> dict[str, pd.DataFrame]:
    target = normalize_subtype(type_filter)
    if target == "unknown":
        return frames

    filtered = {}
    observed = set()
    target_key = target.casefold()
    for name, frame in frames.items():
        st_col = subtype_column(frame)
        values = frame[st_col].map(normalize_subtype)
        observed.update(value for value in values.tolist() if value != "unknown")
        mask = values.astype(str).str.casefold() == target_key
        filtered[name] = frame.loc[mask].reset_index(drop=True)

    if sum(len(frame) for frame in filtered.values()) == 0:
        available = ", ".join(sorted(observed)) if observed else "none"
        raise ValueError(f"No rows found for --type {target!r}; available Type values: {available}")
    if filtered.get("train", pd.DataFrame()).empty:
        raise ValueError(f"No training rows remain after filtering --type {target!r}")
    return filtered


def validate_frame_columns(frame: pd.DataFrame, source: str | Path) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required column(s): {', '.join(missing)}")
    if "Type" not in frame.columns and "virusType" not in frame.columns:
        raise ValueError(f"{source} is missing required column(s): Type or virusType")


def validate_aligned_ha_columns(frames: dict[str, pd.DataFrame]) -> None:
    for split_name, frame in frames.items():
        missing = sorted(ALIGNED_HA_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(
                f"{split_name} split is missing required column(s) for --ha-pair-mode delta: {', '.join(missing)}"
            )


def make_task_keys(frame: pd.DataFrame, task_cols: list[str] | tuple[str, ...]) -> pd.Series:
    missing = [col for col in task_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing task key column(s): {', '.join(missing)}")
    if frame.empty:
        return pd.Series([], index=frame.index, dtype=str)
    values = frame.loc[:, list(task_cols)].fillna("unknown").astype(str)
    return values.apply(lambda row: "||".join(row.tolist()), axis=1)


def merge_train_valid_for_refit(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    train = pd.concat([frames["train"], frames["valid"]], axis=0, ignore_index=True)
    valid = frames["valid"].iloc[0:0].copy().reset_index(drop=True)
    return {
        "train": train.reset_index(drop=True),
        "valid": valid,
        "test": frames["test"].copy().reset_index(drop=True),
    }


def task_overlap_counts(frames: dict[str, pd.DataFrame], task_cols: list[str]) -> dict[str, int]:
    keys = {name: set(make_task_keys(frame, task_cols)) for name, frame in frames.items()}
    return {
        "train_valid": len(keys.get("train", set()) & keys.get("valid", set())),
        "train_test": len(keys.get("train", set()) & keys.get("test", set())),
        "valid_test": len(keys.get("valid", set()) & keys.get("test", set())),
    }


def load_fixed_split_frames(
    data_dir: Path,
    sample_limit: int | None = None,
    refit_train_valid: bool = False,
    type_filter: str = "",
) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    missing = [name for name in SPLIT_FILENAMES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing split file(s): {', '.join(missing)}")

    frames = {name.removesuffix(".csv"): pd.read_csv(data_dir / name) for name in SPLIT_FILENAMES}
    for split_name, frame in frames.items():
        validate_frame_columns(frame, data_dir / f"{split_name}.csv")
    if type_filter:
        frames = filter_frames_by_type(frames, type_filter)
    if sample_limit is not None:
        frames = {name: frame.iloc[:sample_limit].copy() for name, frame in frames.items()}

    if refit_train_valid:
        frames = merge_train_valid_for_refit(frames)
    return frames


def build_serum_gate_vocabs(
    frames: dict[str, pd.DataFrame],
    use_subtype_feature: bool = True,
) -> SerumGateVocabs:
    passages = {"unknown"}
    subtypes = {"unknown"}
    for frame in frames.values():
        passages.update(normalize_passage(value) for value in frame["serumPassCat"].tolist())
        passages.update(normalize_passage(value) for value in frame["virusPassCat"].tolist())
        if use_subtype_feature:
            st_col = subtype_column(frame)
            subtypes.update(normalize_subtype(value) for value in frame[st_col].tolist())
    passage_order = ["unknown"] + sorted(p for p in passages if p != "unknown")
    if not use_subtype_feature:
        return SerumGateVocabs(
            passage_to_id={value: idx for idx, value in enumerate(passage_order)},
            subtype_to_id={"constant": 0},
        )
    subtype_order = ["unknown"] + sorted(s for s in subtypes if s != "unknown")
    return SerumGateVocabs(
        passage_to_id={value: idx for idx, value in enumerate(passage_order)},
        subtype_to_id={value: idx for idx, value in enumerate(subtype_order)},
    )


def required_embedding_files(frames: dict[str, pd.DataFrame], include_na_embeddings: bool = False) -> list[str]:
    combined = pd.concat(list(frames.values()), axis=0, ignore_index=True)
    id_columns = [combined["seq_id_a"], combined["seq_id_c"]]
    if include_na_embeddings:
        id_columns.extend([combined["seq_id_b"], combined["seq_id_d"]])
    ids = pd.concat(id_columns).dropna().astype(str).unique().tolist()
    return sorted(f"matrix_{seq_id}.pt" for seq_id in ids)


def validate_embedding_files(embedding_dir: Path, files: list[str]) -> Path:
    embedding_dir = Path(embedding_dir).expanduser().resolve()
    if not embedding_dir.is_dir():
        raise FileNotFoundError(f"Embedding directory does not exist: {embedding_dir}")
    missing = [filename for filename in files if not (embedding_dir / filename).is_file()]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} HA embedding file(s): {preview}")
    return embedding_dir


def load_embeddings(
    embedding_dir: Path,
    files: list[str],
    show_progress: bool = True,
) -> dict[str, torch.Tensor]:
    store: dict[str, torch.Tensor] = {}
    for filename in tqdm(
        files,
        desc="loading embeddings",
        disable=not show_progress,
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        value = torch.load(embedding_dir / filename, map_location="cpu", weights_only=False)
        store[filename.removesuffix(".pt")] = torch.as_tensor(value).float()
    return store


class SerumGateTaskDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocabs: SerumGateVocabs,
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
        self.task_chunks: list[tuple[str, np.ndarray]] = []
        for task_key, group in self.frame.groupby("_task_key", sort=False):
            indices = group.index.to_numpy()
            if max_queries_per_task is None:
                self.task_chunks.append((task_key, indices))
            else:
                for start in range(0, len(indices), max_queries_per_task):
                    self.task_chunks.append((task_key, indices[start : start + max_queries_per_task]))

    def __len__(self) -> int:
        return len(self.task_chunks)

    def _passage_id(self, value: Any) -> int:
        return self.vocabs.passage_to_id.get(normalize_passage(value), 0)

    def _subtype_id(self, row: pd.Series) -> int:
        return self.vocabs.subtype_to_id.get(normalize_subtype(row[subtype_column(self.frame)]), 0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task_key, row_indices = self.task_chunks[idx]
        task_frame = self.frame.loc[row_indices].reset_index(drop=True)
        first = task_frame.iloc[0]
        serum_passage = self._passage_id(first["serumPassCat"])
        query_passage = torch.tensor(
            [self._passage_id(value) for value in task_frame["virusPassCat"].tolist()],
            dtype=torch.long,
        )
        passage_pair = serum_passage * len(self.vocabs.passage_to_id) + query_passage
        st_id = self._subtype_id(first)
        labels = torch.tensor(task_frame["label"].to_numpy(dtype=float), dtype=torch.float32)
        s_nagly = torch.tensor(
            [
                na_head_glycan_mismatch(str(row["seq_b"]), str(row["seq_d"]))
                for _, row in task_frame.iterrows()
            ],
            dtype=torch.float32,
        )
        return {
            "task_key": task_key,
            "support_size": 0,
            "reference_ha_key": f"matrix_{first['seq_id_a']}",
            "query_ha_keys": [f"matrix_{seq_id}" for seq_id in task_frame["seq_id_c"].tolist()],
            "reference_na_key": f"matrix_{first['seq_id_b']}",
            "query_na_keys": [f"matrix_{seq_id}" for seq_id in task_frame["seq_id_d"].tolist()],
            "reference_ha_aligned": str(first.get("serumHA", "")),
            "query_ha_aligned": [str(value) for value in task_frame.get("virusHA", pd.Series([], dtype=str)).tolist()],
            "serum_passage": torch.tensor(serum_passage, dtype=torch.long),
            "query_passage": query_passage,
            "passage_pair": passage_pair.long(),
            "subtype": torch.tensor(st_id, dtype=torch.long),
            "s_nagly": s_nagly,
            "labels": labels,
            "row_indices": torch.tensor(row_indices, dtype=torch.long),
            "query_meta": task_frame.drop(columns=["_task_key"]).reset_index(drop=True),
        }


def pad_matrices(matrices: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(matrix.shape[0]) for matrix in matrices)
    hidden = int(matrices[0].shape[1])
    out = torch.zeros(len(matrices), max_len, hidden, dtype=torch.float32)
    mask = torch.zeros(len(matrices), max_len, dtype=torch.float32)
    for idx, matrix in enumerate(matrices):
        length = int(matrix.shape[0])
        out[idx, :length] = matrix.float()
        mask[idx, :length] = 1.0
    return out, mask


def align_embedding_to_ha(matrix: torch.Tensor, aligned_ha: str) -> tuple[torch.Tensor, torch.Tensor]:
    aligned_len = len(aligned_ha)
    hidden = int(matrix.shape[1])
    out = torch.zeros(aligned_len, hidden, dtype=torch.float32, device=matrix.device)
    mask = torch.zeros(aligned_len, dtype=torch.float32, device=matrix.device)
    source_idx = 0
    for aligned_idx, residue in enumerate(aligned_ha):
        if residue == "-":
            continue
        if source_idx >= int(matrix.shape[0]):
            break
        out[aligned_idx] = matrix[source_idx].float()
        mask[aligned_idx] = 1.0
        source_idx += 1
    return out, mask


def pad_aligned_matrices(matrices: list[torch.Tensor], masks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(matrix.shape[0]) for matrix in matrices)
    hidden = int(matrices[0].shape[1])
    out = torch.zeros(len(matrices), max_len, hidden, dtype=torch.float32, device=matrices[0].device)
    out_mask = torch.zeros(len(matrices), max_len, dtype=torch.float32, device=matrices[0].device)
    for idx, (matrix, mask) in enumerate(zip(matrices, masks)):
        length = int(matrix.shape[0])
        out[idx, :length] = matrix.float()
        out_mask[idx, :length] = mask.float()
    return out, out_mask


def align_task_ha_matrices(
    reference_matrix: torch.Tensor,
    query_matrices: list[torch.Tensor],
    reference_aligned_ha: str,
    query_aligned_ha: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(query_matrices) != len(query_aligned_ha):
        raise ValueError("query_matrices and query_aligned_ha must have the same length")
    aligned_matrices: list[torch.Tensor] = []
    aligned_masks: list[torch.Tensor] = []
    ref_matrix, ref_mask = align_embedding_to_ha(reference_matrix, reference_aligned_ha)
    aligned_matrices.append(ref_matrix)
    aligned_masks.append(ref_mask)
    for matrix, aligned_ha in zip(query_matrices, query_aligned_ha):
        query_matrix, query_mask = align_embedding_to_ha(matrix, aligned_ha)
        aligned_matrices.append(query_matrix)
        aligned_masks.append(query_mask)
    padded, masks = pad_aligned_matrices(aligned_matrices, aligned_masks)
    return padded[0:1], masks[0:1], padded[1:], masks[1:]


def collate_serum_gate_tasks(
    items: list[dict[str, Any]],
    embeddings: dict[str, torch.Tensor],
    align_ha_embeddings: bool = False,
    include_na_embeddings: bool = False,
) -> tuple[SerumGateBatch, list[dict[str, Any]]]:
    if len(items) != 1:
        raise ValueError("Zero-shot serum task collator currently expects batch_size=1")
    item = items[0]
    reference_matrix = embeddings[item["reference_ha_key"]]
    query_matrices = [embeddings[key] for key in item["query_ha_keys"]]
    if align_ha_embeddings:
        reference, reference_mask, query, query_mask = align_task_ha_matrices(
            reference_matrix,
            query_matrices,
            item["reference_ha_aligned"],
            item["query_ha_aligned"],
        )
    else:
        reference, reference_mask = pad_matrices([reference_matrix])
        query, query_mask = pad_matrices(query_matrices)
    reference_na = None
    query_na = None
    reference_na_mask = None
    query_na_mask = None
    if include_na_embeddings:
        reference_na, reference_na_mask = pad_matrices([embeddings[item["reference_na_key"]]])
        query_na, query_na_mask = pad_matrices([embeddings[key] for key in item["query_na_keys"]])
        query_na = query_na.unsqueeze(0)
        query_na_mask = query_na_mask.unsqueeze(0)
    query = query.unsqueeze(0)
    query_mask = query_mask.unsqueeze(0)
    batch = SerumGateBatch(
        reference_ha=reference,
        query_ha=query,
        reference_ha_mask=reference_mask,
        query_ha_mask=query_mask,
        reference_na=reference_na,
        query_na=query_na,
        reference_na_mask=reference_na_mask,
        query_na_mask=query_na_mask,
        serum_passage=item["serum_passage"].view(1),
        query_passage=item["query_passage"].view(1, -1),
        passage_pair=item["passage_pair"].view(1, -1),
        subtype=item["subtype"].view(1),
        s_nagly=item["s_nagly"].view(1, -1),
        labels=item["labels"].view(1, -1),
        query_mask=torch.ones(1, len(item["labels"]), dtype=torch.float32),
    )
    return batch, items


def collate_serum_gate_tasks_with_gpu_cache(
    items: list[dict[str, Any]],
    cache: GpuEmbeddingCache,
    device: torch.device,
    align_ha_embeddings: bool = False,
    include_na_embeddings: bool = False,
) -> tuple[SerumGateBatch, list[dict[str, Any]]]:
    if len(items) != 1:
        raise ValueError("Zero-shot serum task collator currently expects batch_size=1")
    item = items[0]
    reference_matrix = cache.get(item["reference_ha_key"])
    query_matrices = [cache.get(key) for key in item["query_ha_keys"]]
    if align_ha_embeddings:
        reference, reference_mask, query, query_mask = align_task_ha_matrices(
            reference_matrix,
            query_matrices,
            item["reference_ha_aligned"],
            item["query_ha_aligned"],
        )
    else:
        reference, reference_mask = generate_matrix_on_device(
            [reference_matrix],
            device=device,
        )
        query, query_mask = generate_matrix_on_device(
            query_matrices,
            device=device,
        )
    reference_na = None
    query_na = None
    reference_na_mask = None
    query_na_mask = None
    if include_na_embeddings:
        reference_na, reference_na_mask = generate_matrix_on_device(
            [cache.get(item["reference_na_key"])],
            device=device,
        )
        query_na, query_na_mask = generate_matrix_on_device(
            [cache.get(key) for key in item["query_na_keys"]],
            device=device,
        )
    batch = SerumGateBatch(
        reference_ha=reference,
        query_ha=query.unsqueeze(0),
        reference_ha_mask=reference_mask,
        query_ha_mask=query_mask.unsqueeze(0),
        reference_na=reference_na,
        query_na=query_na.unsqueeze(0) if query_na is not None else None,
        reference_na_mask=reference_na_mask,
        query_na_mask=query_na_mask.unsqueeze(0) if query_na_mask is not None else None,
        serum_passage=item["serum_passage"].view(1).to(device),
        query_passage=item["query_passage"].view(1, -1).to(device),
        passage_pair=item["passage_pair"].view(1, -1).to(device),
        subtype=item["subtype"].view(1).to(device),
        s_nagly=item["s_nagly"].view(1, -1).to(device),
        labels=item["labels"].view(1, -1).to(device),
        query_mask=torch.ones(1, len(item["labels"]), dtype=torch.float32, device=device),
    )
    return batch, items


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _corr(pd.Series(a).rank(method="average").to_numpy(), pd.Series(b).rank(method="average").to_numpy())


def serum_regression_metrics(prediction_frame: pd.DataFrame) -> dict[str, float]:
    if prediction_frame.empty:
        return {
            key: 0.0
            for key in (
                "pooled_mae", "pooled_mse", "pooled_pearson", "pooled_spearman",
                "per_serum_mae_mean", "per_serum_mae_median", "within_serum_pearson_mean",
                "within_serum_spearman_mean", "serum_bias_mean", "serum_bias_abs_mean", "nll", "coverage_80", "coverage_95",
            )
        }
    y = prediction_frame["label"].to_numpy(dtype=float)
    mean = prediction_frame["mean"].to_numpy(dtype=float)
    log_var = prediction_frame["log_var"].to_numpy(dtype=float)
    diff = mean - y
    var = np.exp(np.clip(log_var, -12.0, 12.0))
    nll = 0.5 * (log_var + diff**2 / var)
    sigma = np.sqrt(var)

    per_serum_mae = []
    serum_bias = []
    within_pearson = []
    within_spearman = []
    for _, group in prediction_frame.groupby("task_key"):
        gy = group["label"].to_numpy(dtype=float)
        gm = group["mean"].to_numpy(dtype=float)
        gd = gm - gy
        per_serum_mae.append(float(np.mean(np.abs(gd))))
        serum_bias.append(float(np.mean(gd)))
        if len(group) >= 2:
            within_pearson.append(_corr(gy, gm))
            within_spearman.append(_spearman(gy, gm))

    return {
        "pooled_mae": float(np.mean(np.abs(diff))),
        "pooled_mse": float(np.mean(diff**2)),
        "pooled_pearson": _corr(y, mean),
        "pooled_spearman": _spearman(y, mean),
        "per_serum_mae_mean": float(np.mean(per_serum_mae)),
        "per_serum_mae_median": float(np.median(per_serum_mae)),
        "within_serum_pearson_mean": float(np.mean(within_pearson)) if within_pearson else 0.0,
        "within_serum_spearman_mean": float(np.mean(within_spearman)) if within_spearman else 0.0,
        "serum_bias_mean": float(np.mean(serum_bias)),
        "serum_bias_abs_mean": float(np.mean(np.abs(serum_bias))),
        "nll": float(np.mean(nll)),
        "coverage_80": float(np.mean(np.abs(diff) <= 1.2815515655446004 * sigma)),
        "coverage_95": float(np.mean(np.abs(diff) <= 1.959963984540054 * sigma)),
    }

def subtype_regression_metrics(prediction_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute standard regression metrics separately for the supported subtypes."""
    subtypes = ("H1N1", "H3N2")
    if prediction_frame.empty:
        return {
            subtype.casefold(): serum_regression_metrics(prediction_frame)
            for subtype in subtypes
        }
    column = subtype_column(prediction_frame)
    normalized = prediction_frame[column].map(normalize_subtype).astype(str)
    return {
        subtype.casefold(): serum_regression_metrics(
            prediction_frame.loc[normalized.str.casefold() == subtype.casefold()]
        )
        for subtype in subtypes
    }



def subtype_metrics_enabled(type_filter: str) -> bool:
    """Enable subtype metrics only for joint (ALL) training runs."""
    return not type_filter.strip()

def round_float_value(value: Any, decimals: int = 4) -> Any:
    if isinstance(value, (float, np.floating)):
        return round(float(value), decimals)
    return value


def flatten_epoch_metrics(epoch_metrics: dict[str, Any], decimals: int = 4) -> dict[str, Any]:
    row: dict[str, Any] = {}
    def flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for metric_name, metric_value in value.items():
                nested_prefix = f"{prefix}_{metric_name}" if prefix else metric_name
                flatten(nested_prefix, metric_value)
        else:
            row[prefix] = round_float_value(value, decimals)

    for key, value in epoch_metrics.items():
        flatten(key, value)
    return row


def append_metrics_csv(path: Path, epoch_metrics: dict[str, Any]) -> None:
    row = flatten_epoch_metrics(epoch_metrics)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def write_rounded_predictions(frame: pd.DataFrame, path: Path, decimals: int = 4) -> None:
    rounded = frame.copy()
    float_columns = rounded.select_dtypes(include=["floating"]).columns
    if len(float_columns) > 0:
        rounded.loc[:, float_columns] = rounded.loc[:, float_columns].round(decimals)
    rounded.to_csv(path, index=False)


def infer_hidden_size(embeddings: dict[str, torch.Tensor]) -> int:
    if not embeddings:
        raise ValueError("No embeddings loaded")
    first = next(iter(embeddings.values()))
    if first.ndim != 2:
        raise ValueError(f"Expected 2D embedding, got {tuple(first.shape)}")
    return int(first.shape[1])


def prepare_output_dir(output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir()
    return {
        "root": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "config": output_dir / "run_config.json",
        "metrics": output_dir / "metrics.csv",
        "log": output_dir / "log.txt",
        "metrics_by_subtype": output_dir / "metrics_by_subtype.csv",
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loader(
    frame: pd.DataFrame,
    vocabs: SerumGateVocabs,
    embeddings: dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool,
    max_queries_per_task: int | None = 128,
    gpu_cache: GpuEmbeddingCache | None = None,
    device: torch.device | None = None,
    task_cols: list[str] | None = None,
    align_ha_embeddings: bool = False,
    include_na_embeddings: bool = False,
) -> DataLoader:
    if batch_size != 1:
        raise ValueError("Use --batch-size 1 for variable-size SerumGate tasks")
    dataset = SerumGateTaskDataset(frame, vocabs, task_cols=task_cols, max_queries_per_task=max_queries_per_task)
    if gpu_cache is not None:
        if device is None:
            raise ValueError("device is required when gpu_cache is enabled")
        collate_fn = lambda items: collate_serum_gate_tasks_with_gpu_cache(
            items,
            gpu_cache,
            device,
            align_ha_embeddings=align_ha_embeddings,
            include_na_embeddings=include_na_embeddings,
        )
    else:
        collate_fn = lambda items: collate_serum_gate_tasks(
            items,
            embeddings,
            align_ha_embeddings=align_ha_embeddings,
            include_na_embeddings=include_na_embeddings,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def evaluate_model(
    model: SerumGateMinusModel,
    loader: DataLoader,
    device: torch.device,
    progress_bar=None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch, items in loader:
            item = items[0]
            batch = move_batch(batch, device)
            out = model(batch)
            losses.append(float(out["nll_loss"].item()))
            mean = out["mean"].detach().cpu().view(-1).numpy()
            log_var = out["log_var"].detach().cpu().view(-1).numpy()
            self_score = out["self_score"].detach().cpu().view(-1).numpy()
            query_score = out["query_score"].detach().cpu().view(-1).numpy()
            self_log_var = out["self_log_var"].detach().cpu().view(-1).numpy()
            query_log_var = out["query_log_var"].detach().cpu().view(-1).numpy()
            labels = batch.labels.detach().cpu().view(-1).numpy()
            meta = item["query_meta"]
            for idx in range(len(labels)):
                row = meta.iloc[idx].to_dict()
                row.update(
                    {
                        "task_key": item["task_key"],
                        "label": float(labels[idx]),
                        "mean": float(mean[idx]),
                        "log_var": float(log_var[idx]),
                        "self_score": float(self_score[idx]),
                        "query_score": float(query_score[idx]),
                        "self_log_var": float(self_log_var[idx]),
                        "query_log_var": float(query_log_var[idx]),
                    }
                )
                rows.append(row)
            if progress_bar is not None:
                progress_bar.update(1)
    pred_frame = pd.DataFrame(rows)
    metrics = serum_regression_metrics(pred_frame)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, pred_frame


def move_batch(batch: SerumGateBatch, device: torch.device) -> SerumGateBatch:
    return SerumGateBatch(
        reference_ha=batch.reference_ha.to(device),
        query_ha=batch.query_ha.to(device),
        reference_ha_mask=batch.reference_ha_mask.to(device) if batch.reference_ha_mask is not None else None,
        query_ha_mask=batch.query_ha_mask.to(device) if batch.query_ha_mask is not None else None,
        reference_na=batch.reference_na.to(device) if batch.reference_na is not None else None,
        query_na=batch.query_na.to(device) if batch.query_na is not None else None,
        reference_na_mask=batch.reference_na_mask.to(device) if batch.reference_na_mask is not None else None,
        query_na_mask=batch.query_na_mask.to(device) if batch.query_na_mask is not None else None,
        serum_passage=batch.serum_passage.to(device) if batch.serum_passage is not None else None,
        query_passage=batch.query_passage.to(device) if batch.query_passage is not None else None,
        passage_pair=batch.passage_pair.to(device) if batch.passage_pair is not None else None,
        subtype=batch.subtype.to(device) if batch.subtype is not None else None,
        s_nagly=batch.s_nagly.to(device) if batch.s_nagly is not None else None,
        labels=batch.labels.to(device) if batch.labels is not None else None,
        query_mask=batch.query_mask.to(device) if batch.query_mask is not None else None,
    )


def train_one_epoch(
    model: SerumGateMinusModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_name: str,
    progress_bar=None,
) -> float:
    model.train()
    losses = []
    for batch, _items in loader:
        optimizer.zero_grad()
        out = model(move_batch(batch, device))
        loss = out["nll_loss"] if loss_name == "nll" else out["huber_loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if progress_bar is not None:
            progress_bar.update(1)
    return float(np.mean(losses)) if losses else 0.0


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    task_cols = parse_column_list(args.serum_task_cols)
    sample_limit = None if args.sample_limit < 0 else int(args.sample_limit)
    frames = load_fixed_split_frames(
        args.data_dir,
        sample_limit=sample_limit,
        refit_train_valid=args.refit_train_valid,
        type_filter=args.type_filter,
    )
    if args.ha_pair_mode == "delta":
        validate_aligned_ha_columns(frames)
    include_na_embeddings = args.na_branch != "none"
    embedding_files = required_embedding_files(frames, include_na_embeddings=include_na_embeddings)
    embedding_dir = validate_embedding_files(args.embedding_dir, embedding_files)
    embeddings = load_embeddings(embedding_dir, embedding_files, show_progress=args.progress)
    hidden_size = infer_hidden_size(embeddings)
    vocabs = build_serum_gate_vocabs(frames, use_subtype_feature=args.use_subtype_feature)
    paths = prepare_output_dir(args.output_dir)
    set_seed(args.seed)

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
    loaders = {
        name: build_loader(
            frame,
            vocabs,
            embeddings,
            args.batch_size,
            shuffle=(name == "train"),
            max_queries_per_task=args.max_queries_per_task,
            gpu_cache=gpu_cache,
            device=device,
            task_cols=task_cols,
            align_ha_embeddings=(args.ha_pair_mode == "delta"),
            include_na_embeddings=include_na_embeddings,
        )
        for name, frame in frames.items()
    }
    label_weight_values = parse_float_list(args.label_weight_values)
    label_weight_thresholds = resolve_label_weight_thresholds(
        frames,
        mode=args.label_weight_threshold_mode,
        thresholds_arg=args.label_weight_thresholds,
        quantiles_arg=args.label_weight_quantiles,
    )
    label_weight_quantiles = parse_float_list(args.label_weight_quantiles)
    if len(label_weight_values) != len(label_weight_thresholds) + 1:
        raise ValueError("--label-weight-values must have exactly one more entry than --label-weight-thresholds")
    model_config = SerumGateMinusConfig(
        hidden_size=hidden_size,
        latent_dim=args.latent_dim,
        theta_dim=args.theta_dim,
        passage_vocab_size=len(vocabs.passage_to_id),
        passage_pair_vocab_size=len(vocabs.passage_to_id) ** 2,
        subtype_vocab_size=len(vocabs.subtype_to_id),
        subtype_dim=args.subtype_dim,
        predictor_arch=args.predictor_arch,
        predictor_hidden_dim=args.predictor_hidden_dim,
        distance_hidden_dim=args.distance_hidden_dim,
        calibration_hidden_dim=args.calibration_hidden_dim,
        residual_hidden_dim=args.residual_hidden_dim,
        residual_scale=args.residual_scale,
        na_branch=args.na_branch,
        na_pooling=args.na_pooling,
        na_latent_dim=args.na_latent_dim,
        na_hidden_dim=args.na_hidden_dim,
        na_effect_init=args.na_effect_init,
        label_weight_thresholds=tuple(label_weight_thresholds),
        label_weight_values=tuple(label_weight_values),
        ha_pooling=args.ha_pooling,
        ha_pair_mode=args.ha_pair_mode,
        ha_attention_dim=args.ha_attention_dim,
        ha_attention_heads=args.ha_attention_heads,
        ha_attention_dropout=args.ha_attention_dropout,
        ha_mean_gate_init=args.ha_mean_gate_init,
        rank_loss_weight=args.rank_loss_weight,
        rank_loss_margin=args.rank_loss_margin,
        rank_loss_min_label_delta=args.rank_loss_min_label_delta,
        score_log_var_mode=args.score_log_var_mode,
    )
    model = SerumGateMinusModel(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "SerumGate-Minus",
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "embedding_dir": str(embedding_dir),
        "output_dir": str(paths["root"]),
        "type_filter": args.type_filter,
        "task_cols": task_cols,
        "refit_train_valid": bool(args.refit_train_valid),
        "task_overlap_counts": task_overlap_counts(frames, task_cols),
        "row_counts": {name: len(frame) for name, frame in frames.items()},
        "task_counts": {
            name: len(make_task_keys(frame, task_cols).drop_duplicates())
            for name, frame in frames.items()
        },
        "episode_counts": {
            name: len(SerumGateTaskDataset(frame, vocabs, task_cols, max_queries_per_task=args.max_queries_per_task))
            for name, frame in frames.items()
        },
        "max_queries_per_task": args.max_queries_per_task,
        "gpu_cache_gb": args.gpu_cache_gb,
        "label_weight_threshold_mode": args.label_weight_threshold_mode,
        "label_weight_quantiles": label_weight_quantiles,
        "label_weight_thresholds": label_weight_thresholds,
        "label_weight_values": label_weight_values,
        "model_config": model_config.__dict__,
        "passage_to_id": vocabs.passage_to_id,
        "subtype_to_id": vocabs.subtype_to_id,
        "use_subtype_feature": bool(args.use_subtype_feature),
        "loss": args.loss,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler": args.lr_scheduler,
        "lr_min": args.lr_min,
        "early_stopping_patience": args.early_stopping_patience,
        "skip_test_eval": bool(args.skip_test_eval),
        "seed": args.seed,
    }
    paths["config"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["log"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    best_valid = float("inf")
    best_result: dict[str, Any] = {}
    epochs_without_improvement = 0
    eval_steps = 0 if args.skip_test_eval else len(loaders["test"])
    total_steps = args.epochs * (len(loaders["train"]) + len(loaders["valid"]) + eval_steps)
    with tqdm(
        total=total_steps,
        desc="training",
        disable=not args.progress,
        dynamic_ncols=True,
        file=sys.stdout,
    ) as progress_bar:
        for epoch in range(args.epochs):
            progress_bar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", split="train")
            train_loss = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                device,
                args.loss,
                progress_bar=progress_bar,
            )
            if args.refit_train_valid:
                valid_metrics = serum_regression_metrics(pd.DataFrame())
                valid_metrics["loss"] = 0.0
                valid_preds = pd.DataFrame()
            else:
                progress_bar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", split="valid")
                valid_metrics, valid_preds = evaluate_model(
                    model,
                    loaders["valid"],
                    device,
                    progress_bar=progress_bar,
                )
            if args.skip_test_eval:
                test_metrics = serum_regression_metrics(pd.DataFrame())
                test_metrics["loss"] = 0.0
                test_preds = pd.DataFrame()
            else:
                progress_bar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", split="test")
                test_metrics, test_preds = evaluate_model(
                    model,
                    loaders["test"],
                    device,
                    progress_bar=progress_bar,
                )
            epoch_result = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "valid": valid_metrics,
                "test": test_metrics,
            }
            append_metrics_csv(paths["metrics"], epoch_result)
            is_refit_final_epoch = args.refit_train_valid and epoch + 1 == args.epochs
            if subtype_metrics_enabled(args.type_filter):
                subtype_epoch_result = {
                    "epoch": epoch + 1,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_loss": train_loss,
                    "valid": subtype_regression_metrics(valid_preds),
                    "test": subtype_regression_metrics(test_preds),
                }
                append_metrics_csv(paths["metrics_by_subtype"], subtype_epoch_result)
            is_best_valid_epoch = (not args.refit_train_valid) and valid_metrics["pooled_mse"] < best_valid
            if is_refit_final_epoch or is_best_valid_epoch:
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
                if not valid_preds.empty:
                    write_rounded_predictions(valid_preds, paths["root"] / "predictions_valid.csv")
                if not test_preds.empty:
                    write_rounded_predictions(test_preds, paths["root"] / "predictions_test.csv")
            elif not args.refit_train_valid:
                epochs_without_improvement += 1
            if scheduler is not None:
                scheduler.step()
            if (
                args.early_stopping_patience >= 0
                and not args.refit_train_valid
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                with paths["log"].open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"no valid_pooled_mse improvement for {epochs_without_improvement} epoch(s).\\n"
                    )
                break
    return {"paths": {key: str(value) for key, value in paths.items()}, "best": best_result}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SerumGate-Minus.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--type",
        dest="type_filter",
        default="",
        help="Optional Type/virusType value to keep, e.g. H1N1 or H3N2.",
    )
    parser.add_argument("--serum-task-cols", type=str, default=",".join(DEFAULT_TASK_COLS))
    parser.add_argument(
        "--refit-train-valid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Merge fixed train and valid splits, then train for a fixed epoch count.",
    )
    parser.add_argument("--sample-limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-queries-per-task",
        type=int,
        default=128,
        help="Split large serum profiles into query chunks to bound GPU memory. Use <=0 to disable.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "cosine"),
        default="none",
        help="Optional learning-rate scheduler.",
    )
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum LR for --lr-scheduler cosine.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=-1,
        help="Stop after this many epochs without valid pooled MSE improvement. Negative disables.",
    )
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--theta-dim", type=int, default=128)
    parser.add_argument("--subtype-dim", type=int, default=8)
    parser.add_argument(
        "--use-subtype-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Type/virusType as a model feature. Disable for subtype-specific models.",
    )
    parser.add_argument(
        "--predictor-arch",
        choices=("conditioned_mlp", "calibrated_metric"),
        default="conditioned_mlp",
        help="Prediction head architecture.",
    )
    parser.add_argument("--predictor-hidden-dim", type=int, default=256)
    parser.add_argument("--distance-hidden-dim", type=int, default=64)
    parser.add_argument("--calibration-hidden-dim", type=int, default=64)
    parser.add_argument("--residual-hidden-dim", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument(
        "--na-branch",
        choices=("none", "pair"),
        default="none",
        help="Optional NA embedding correction branch. Use pair to add an NA pair effect to the mean.",
    )
    parser.add_argument(
        "--na-pooling",
        choices=("attention", "mean", "lowrank_attention"),
        default="mean",
        help="NA token pooling used by --na-branch pair. Defaults to mean for a lightweight NA branch.",
    )
    parser.add_argument("--na-latent-dim", type=int, default=32)
    parser.add_argument("--na-hidden-dim", type=int, default=32)
    parser.add_argument(
        "--na-effect-init",
        type=float,
        default=0.1,
        help="Initial sigmoid gate for the additive NA effect; must be in (0, 1).",
    )
    parser.add_argument(
        "--ha-pooling",
        choices=("attention", "mean", "lowrank_attention"),
        default="attention",
        help="HA token pooling used before projection.",
    )
    parser.add_argument(
        "--ha-attention-dim",
        type=int,
        default=128,
        help="Low-rank hidden size for --ha-pooling lowrank_attention.",
    )
    parser.add_argument(
        "--ha-attention-heads",
        type=int,
        default=4,
        help="Number of self-attention heads for --ha-pooling lowrank_attention.",
    )
    parser.add_argument(
        "--ha-attention-dropout",
        type=float,
        default=0.1,
        help="Dropout used inside --ha-pooling lowrank_attention.",
    )
    parser.add_argument(
        "--ha-mean-gate-init",
        type=float,
        default=0.25,
        help="Initial attention-vs-mean mixing gate for --ha-pooling lowrank_attention.",
    )
    parser.add_argument(
        "--ha-pair-mode",
        choices=("independent", "delta"),
        default="independent",
        help="HA pair representation: independent encodes serum/query separately; delta pools query-serum site deltas.",
    )
    parser.add_argument(
        "--label-weight-thresholds",
        type=str,
        default="2,4,6",
        help="Comma-separated label thresholds for loss weighting.",
    )
    parser.add_argument(
        "--label-weight-threshold-mode",
        choices=("fixed", "quantile"),
        default="fixed",
        help="Use fixed thresholds or compute them from the current training label quantiles.",
    )
    parser.add_argument(
        "--label-weight-quantiles",
        type=str,
        default="0.35,0.75,0.95",
        help="Comma-separated training label quantiles used when --label-weight-threshold-mode quantile.",
    )
    parser.add_argument(
        "--label-weight-values",
        type=str,
        default="1,1.3,1.8,2.5",
        help="Comma-separated loss weights; must have one more value than thresholds.",
    )
    parser.add_argument(
        "--within-serum-rank-loss-weight",
        dest="rank_loss_weight",
        type=float,
        default=0.0,
        help="Weight for within-serum pairwise ranking loss. 0 disables it.",
    )
    parser.add_argument(
        "--within-serum-rank-margin",
        dest="rank_loss_margin",
        type=float,
        default=0.1,
        help="Hinge margin for within-serum pairwise ranking loss.",
    )
    parser.add_argument(
        "--within-serum-rank-min-label-delta",
        dest="rank_loss_min_label_delta",
        type=float,
        default=0.0,
        help="Minimum absolute label difference required for a pair to enter ranking loss.",
    )
    parser.add_argument("--loss", choices=("huber", "nll"), default="nll")
    parser.add_argument(
        "--score-log-var-mode",
        choices=("sum", "query"),
        default="sum",
        help="How to form distance log variance from self/query score variances.",
    )
    parser.add_argument(
        "--skip-test-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip per-epoch test-set evaluation while selecting checkpoints by validation metrics.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--gpu-cache-gb",
        type=float,
        default=0.0,
        help="Optional GPU embedding LRU cache size. 0 disables GPU cache.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.lr_min < 0:
        parser.error("--lr-min must be non-negative")
    if args.early_stopping_patience < -1:
        parser.error("--early-stopping-patience must be -1 or greater")
    if args.type_filter:
        args.use_subtype_feature = False
    if not args.use_subtype_feature:
        args.subtype_dim = 0
    if args.max_queries_per_task <= 0:
        args.max_queries_per_task = None
    return args


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
