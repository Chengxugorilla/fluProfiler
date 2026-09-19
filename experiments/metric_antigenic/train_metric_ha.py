#!/usr/bin/env python3
"""
Train the Metric HA antigenic-space observation model from fixed split CSVs.
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
sys.path.append(str(_REPO_ROOT / "src" / "fluprofiler"))
sys.path.append(str(_REPO_ROOT / "experiments" / "reverse_tests"))

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean envs
    SummaryWriter = None

from experiment_tools import GpuEmbeddingCache, generate_matrix_on_device  # noqa: E402
from fluprofiler.data.loaders import load_embedding  # noqa: E402
from fluprofiler.features.na_glycan_features import na_head_glycan_mismatch  # noqa: E402
from fluprofiler.models.metric_antigenic_model import (  # noqa: E402
    MetricAntigenicBatch,
    MetricHAAntigenicModel,
    MetricHAAntigenicModelConfig,
)


SPLIT_FILENAMES = ("train.csv", "valid.csv", "test.csv")
REQUIRED_COLUMNS = {
    "seq_id_a",
    "seq_id_c",
    "seq_b",
    "seq_d",
    "serumHA",
    "virusHA",
    "serumPassCat",
    "virusPassCat",
    "label",
}

REVERSE_COLUMN_PAIRS = (
    ("seq_id_a", "seq_id_c"),
    ("seq_id_b", "seq_id_d"),
    ("seq_a", "seq_c"),
    ("seq_b", "seq_d"),
    ("serumPassCat", "virusPassCat"),
    ("serumName", "virusName"),
    ("serumDate", "virusDate"),
    ("serumIslID", "virusIslID"),
    ("serumHA", "virusHA"),
)


@dataclass(frozen=True)
class FeatureVocabs:
    passage_to_id: dict[str, int]
    subtype_to_id: dict[str, int]


@dataclass(frozen=True)
class NameVocabs:
    serum_to_id: dict[str, int]
    virus_to_id: dict[str, int]


class NullWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def normalize_passage(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    text = text.replace("<", "").replace(">", "")
    if text in {"egg", "cell", "both"}:
        return text
    if text in {"none", "nan", "na", "unknown"}:
        return "unknown"
    return text


def normalize_subtype(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def load_fixed_split_frames(data_dir: Path, sample_limit: int | None = None) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    missing_files = [filename for filename in SPLIT_FILENAMES if not (data_dir / filename).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required split file(s) under {data_dir}: {', '.join(missing_files)}"
        )

    frames = {name.removesuffix(".csv"): pd.read_csv(data_dir / name) for name in SPLIT_FILENAMES}
    for split_name, frame in frames.items():
        validate_training_frame_columns(frame, data_dir / (split_name + ".csv"))

    if sample_limit is not None:
        frames = {name: frame.iloc[:sample_limit].copy() for name, frame in frames.items()}
    return frames


def validate_training_frame_columns(frame: pd.DataFrame, source: Path | str) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    subtype_columns = {"Type", "virusType"} & set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{source} is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )
    if not subtype_columns:
        raise ValueError(f"{source} is missing required column(s): Type or virusType")


def load_artificial_train_frame(path: Path, sample_limit: int | None = None) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Artificial train CSV does not exist: {path}")

    frame = pd.read_csv(path)
    validate_training_frame_columns(frame, path)
    if sample_limit is not None:
        frame = frame.iloc[:sample_limit].copy()
    return frame


def required_embedding_files(frames: dict[str, pd.DataFrame]) -> list[str]:
    combined = pd.concat(list(frames.values()), axis=0, ignore_index=True)
    sequence_ids = pd.concat([combined["seq_id_a"], combined["seq_id_c"]]).dropna().unique().tolist()
    return sorted(f"matrix_{seq_id}.pt" for seq_id in sequence_ids)


def validate_embedding_files(embedding_dir: Path, embedding_files: list[str]) -> Path:
    embedding_dir = Path(embedding_dir).expanduser().resolve()
    if not embedding_dir.is_dir():
        raise FileNotFoundError(f"Embedding directory does not exist: {embedding_dir}")

    missing = [filename for filename in embedding_files if not (embedding_dir / filename).is_file()]
    if missing:
        preview = ", ".join(missing[:10])
        remainder = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} required HA embedding file(s) in {embedding_dir}: "
            f"{preview}{remainder}"
        )
    return embedding_dir


def subtype_column(frame: pd.DataFrame) -> str:
    return "Type" if "Type" in frame.columns else "virusType"


def build_feature_vocabs(frames: dict[str, pd.DataFrame]) -> FeatureVocabs:
    passages = {"unknown"}
    subtypes = {"unknown"}
    for frame in frames.values():
        passages.update(normalize_passage(value) for value in frame["serumPassCat"].tolist())
        passages.update(normalize_passage(value) for value in frame["virusPassCat"].tolist())
        st_col = subtype_column(frame)
        subtypes.update(normalize_subtype(value) for value in frame[st_col].tolist())

    passage_order = ["unknown"] + sorted(p for p in passages if p != "unknown")
    subtype_order = ["unknown"] + sorted(s for s in subtypes if s != "unknown")
    return FeatureVocabs(
        passage_to_id={value: idx for idx, value in enumerate(passage_order)},
        subtype_to_id={value: idx for idx, value in enumerate(subtype_order)},
    )


def build_name_vocabs(train_frame: pd.DataFrame) -> NameVocabs:
    missing = [col for col in ("serumName", "virusName") if col not in train_frame.columns]
    if missing:
        raise ValueError(f"Name bias requires column(s): {', '.join(missing)}")
    serum_names = sorted(train_frame["serumName"].fillna("").astype(str).unique().tolist())
    virus_names = sorted(train_frame["virusName"].fillna("").astype(str).unique().tolist())
    return NameVocabs(
        serum_to_id={name: idx + 1 for idx, name in enumerate(serum_names) if name},
        virus_to_id={name: idx + 1 for idx, name in enumerate(virus_names) if name},
    )


def build_reverse_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    reversed_frame = frame.copy()
    for left, right in REVERSE_COLUMN_PAIRS:
        if left in reversed_frame.columns and right in reversed_frame.columns:
            left_values = reversed_frame[left].copy()
            reversed_frame[left] = reversed_frame[right]
            reversed_frame[right] = left_values
    reversed_frame["is_reverse_artificial"] = True
    return reversed_frame


def build_artificial_augmented_train_frame(
    train_frame: pd.DataFrame,
    artificial_train_frame: pd.DataFrame,
) -> pd.DataFrame:
    real_frame = train_frame.copy()
    real_frame["is_artificial"] = False
    if "is_reverse_artificial" not in real_frame.columns:
        real_frame["is_reverse_artificial"] = False

    artificial_frame = artificial_train_frame.copy()
    artificial_frame["is_artificial"] = True
    if "is_reverse_artificial" not in artificial_frame.columns:
        artificial_frame["is_reverse_artificial"] = False
    return pd.concat([real_frame, artificial_frame], axis=0, ignore_index=True)


def build_reverse_augmented_train_frame(train_frame: pd.DataFrame) -> pd.DataFrame:
    reverse_frame = build_reverse_pair_frame(train_frame)
    return build_artificial_augmented_train_frame(train_frame, reverse_frame)


def ha_mismatch_vector(serum_ha: Any, virus_ha: Any, start: int = 16, end: int = 345) -> torch.Tensor:
    serum = "" if pd.isna(serum_ha) else str(serum_ha)
    virus = "" if pd.isna(virus_ha) else str(virus_ha)
    values = []
    for pos in range(start, end):
        if pos >= len(serum) or pos >= len(virus):
            values.append(0.0)
        else:
            values.append(0.0 if serum[pos] == virus[pos] else 1.0)
    return torch.tensor(values, dtype=torch.float32)


class MetricHADataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocabs: FeatureVocabs,
        name_vocabs: NameVocabs | None = None,
        ha_mismatch_start: int = 16,
        ha_mismatch_end: int = 345,
    ):
        self.frame = dataframe.reset_index(drop=True)
        self.vocabs = vocabs
        self.name_vocabs = name_vocabs
        self.passage_vocab_size = len(vocabs.passage_to_id)
        self.ha_mismatch_start = ha_mismatch_start
        self.ha_mismatch_end = ha_mismatch_end

    def __len__(self) -> int:
        return len(self.frame)

    def _passage_id(self, value: Any) -> int:
        return self.vocabs.passage_to_id.get(normalize_passage(value), 0)

    def _subtype_id(self, row: pd.Series) -> int:
        return self.vocabs.subtype_to_id.get(normalize_subtype(row[subtype_column(self.frame)]), 0)

    def _serum_name_id(self, row: pd.Series) -> int:
        if self.name_vocabs is None:
            return 0
        return self.name_vocabs.serum_to_id.get(str(row.get("serumName", "") or ""), 0)

    def _virus_name_id(self, row: pd.Series) -> int:
        if self.name_vocabs is None:
            return 0
        return self.name_vocabs.virus_to_id.get(str(row.get("virusName", "") or ""), 0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        serum_passage = self._passage_id(row["serumPassCat"])
        test_passage = self._passage_id(row["virusPassCat"])
        passage_pair = serum_passage * self.passage_vocab_size + test_passage
        return {
            "serum_ha_key": f"matrix_{row['seq_id_a']}",
            "virus_ha_key": f"matrix_{row['seq_id_c']}",
            "serum_passage": torch.tensor(serum_passage, dtype=torch.long),
            "test_passage": torch.tensor(test_passage, dtype=torch.long),
            "passage_pair": torch.tensor(passage_pair, dtype=torch.long),
            "subtype": torch.tensor(self._subtype_id(row), dtype=torch.long),
            "serum_name": torch.tensor(self._serum_name_id(row), dtype=torch.long),
            "virus_name": torch.tensor(self._virus_name_id(row), dtype=torch.long),
            "s_nagly": torch.tensor(
                na_head_glycan_mismatch(str(row["seq_b"]), str(row["seq_d"])),
                dtype=torch.float32,
            ),
            "ha_mismatch": ha_mismatch_vector(
                row["serumHA"],
                row["virusHA"],
                start=self.ha_mismatch_start,
                end=self.ha_mismatch_end,
            ),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }


def build_dataloaders(
    frames: dict[str, pd.DataFrame],
    vocabs: FeatureVocabs,
    batch_size: int,
    name_vocabs: NameVocabs | None = None,
    ha_mismatch_start: int = 16,
    ha_mismatch_end: int = 345,
) -> dict[str, DataLoader]:
    datasets = {
        name: MetricHADataset(
            frame,
            vocabs,
            name_vocabs=name_vocabs,
            ha_mismatch_start=ha_mismatch_start,
            ha_mismatch_end=ha_mismatch_end,
        )
        for name, frame in frames.items()
    }
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "valid": DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }


def build_artificial_warmup_train_loader(
    train_frame: pd.DataFrame,
    artificial_train_frame: pd.DataFrame | None,
    vocabs: FeatureVocabs,
    batch_size: int,
    enabled: bool,
    name_vocabs: NameVocabs | None = None,
    ha_mismatch_start: int = 16,
    ha_mismatch_end: int = 345,
) -> DataLoader | None:
    if not enabled:
        return None
    if artificial_train_frame is None:
        augmented_train = build_reverse_augmented_train_frame(train_frame)
    else:
        augmented_train = build_artificial_augmented_train_frame(train_frame, artificial_train_frame)
    dataset = MetricHADataset(
        augmented_train,
        vocabs,
        name_vocabs=name_vocabs,
        ha_mismatch_start=ha_mismatch_start,
        ha_mismatch_end=ha_mismatch_end,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def build_reverse_warmup_train_loader(
    train_frame: pd.DataFrame,
    vocabs: FeatureVocabs,
    batch_size: int,
    enabled: bool,
    name_vocabs: NameVocabs | None = None,
    ha_mismatch_start: int = 16,
    ha_mismatch_end: int = 345,
) -> DataLoader | None:
    return build_artificial_warmup_train_loader(
        train_frame,
        None,
        vocabs,
        batch_size,
        enabled,
        name_vocabs=name_vocabs,
        ha_mismatch_start=ha_mismatch_start,
        ha_mismatch_end=ha_mismatch_end,
    )


def train_loader_for_epoch(
    real_train_loader: DataLoader,
    augmented_train_loader: DataLoader | None,
    epoch_index: int,
    reverse_warmup_epochs: int,
) -> DataLoader:
    if augmented_train_loader is not None and epoch_index < reverse_warmup_epochs:
        return augmented_train_loader
    return real_train_loader


def prepare_output_dir(output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    tensorboard = output_dir / "tensorboard"
    checkpoints.mkdir()
    tensorboard.mkdir()
    return {
        "root": output_dir,
        "checkpoints": checkpoints,
        "tensorboard": tensorboard,
        "config": output_dir / "run_config.json",
        "metrics": output_dir / "metrics.jsonl",
        "log": output_dir / "log.txt",
    }


def batch_to_model_input(batch: dict[str, Any], device: torch.device, cache: GpuEmbeddingCache) -> MetricAntigenicBatch:
    serum_keys = batch["serum_ha_key"]
    virus_keys = batch["virus_ha_key"]
    serum_mats = [cache.get(key) for key in serum_keys]
    virus_mats = [cache.get(key) for key in virus_keys]
    serum_ha, serum_mask = generate_matrix_on_device(serum_mats, device=device)
    virus_ha, virus_mask = generate_matrix_on_device(virus_mats, device=device)
    return MetricAntigenicBatch(
        serum_ha=serum_ha,
        virus_ha=virus_ha,
        serum_ha_mask=serum_mask,
        virus_ha_mask=virus_mask,
        serum_passage=batch["serum_passage"].to(device),
        test_passage=batch["test_passage"].to(device),
        passage_pair=batch["passage_pair"].to(device),
        subtype=batch["subtype"].to(device),
        serum_name=batch["serum_name"].to(device),
        virus_name=batch["virus_name"].to(device),
        ha_mismatch=batch["ha_mismatch"].to(device),
        s_nagly=batch["s_nagly"].to(device),
        labels=batch["label"].to(device),
    )


def regression_metrics(reference: list[float], prediction: list[float]) -> dict[str, float]:
    y = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    diff = pred - y
    mae = float(np.mean(np.abs(diff))) if len(y) else 0.0
    mse = float(np.mean(diff**2)) if len(y) else 0.0
    r2_den = float(np.sum((y - y.mean()) ** 2)) if len(y) else 0.0
    r2 = 0.0 if r2_den == 0.0 else float(1.0 - np.sum(diff**2) / r2_den)
    pearson = 0.0
    if len(y) > 1 and float(np.std(y)) > 0.0 and float(np.std(pred)) > 0.0:
        pearson = float(np.corrcoef(y, pred)[0, 1])
    return {"mae": mae, "mse": mse, "pearson": pearson, "r2": r2}


def evaluate(
    model: MetricHAAntigenicModel,
    dataloader: DataLoader,
    device: torch.device,
    cache: GpuEmbeddingCache,
    use_name_bias: bool = True,
):
    model.eval()
    losses: list[float] = []
    prediction: list[float] = []
    reference: list[float] = []
    components: dict[str, list[float]] = {
        "d_ha": [],
        "rho_ha": [],
        "b_assay": [],
        "s_nagly": [],
        "r_na": [],
        "name_bias": [],
        "ha_mismatch_residual": [],
    }
    with torch.no_grad():
        for batch in dataloader:
            model_batch = batch_to_model_input(batch, device, cache)
            model_batch.use_name_bias = use_name_bias
            out = model(model_batch)
            if out["loss"] is not None:
                losses.append(float(out["loss"].item()))
            prediction.extend(out["pred"].detach().cpu().view(-1).tolist())
            reference.extend(model_batch.labels.detach().cpu().view(-1).tolist())
            for key in components:
                components[key].extend(out[key].detach().cpu().view(-1).tolist())

    metrics = regression_metrics(reference, prediction)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    for key, values in components.items():
        metrics[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
    return metrics


def train_one_epoch(
    model: MetricHAAntigenicModel,
    dataloader: DataLoader,
    device: torch.device,
    cache: GpuEmbeddingCache,
    optimizer: torch.optim.Optimizer,
    progress_bar=None,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in dataloader:
        optimizer.zero_grad()
        model_batch = batch_to_model_input(batch, device, cache)
        out = model(model_batch)
        loss = out["loss"]
        if loss is None:
            raise RuntimeError("MetricHAAntigenicModel did not return a training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if progress_bar is not None:
            progress_bar.update(1)
    return float(np.mean(losses)) if losses else 0.0


def infer_hidden_size(embeddings: dict[str, torch.Tensor]) -> int:
    if not embeddings:
        raise ValueError("No embeddings were loaded")
    first = next(iter(embeddings.values()))
    if first.ndim != 2:
        raise ValueError(f"Expected 2D embedding tensor, got shape {tuple(first.shape)}")
    return int(first.shape[1])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Metric HA antigenic model from fixed split CSVs.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gpu-cache-gb", type=float, default=20)
    parser.add_argument("--sample-limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--na-lambda-max", type=float, default=0.25)
    parser.add_argument(
        "--artificial-train-csv",
        type=Path,
        default=None,
        help="Optional artificial train CSV to mix with real train data during warmup epochs.",
    )
    parser.add_argument(
        "--artificial-warmup-epochs",
        "--reverse-warmup-epochs",
        dest="artificial_warmup_epochs",
        type=int,
        default=0,
        help=(
            "Use real train plus artificial train rows for the first N epoch(s). "
            "If --artificial-train-csv is omitted, 1:1 reversed train pairs are generated in memory."
        ),
    )
    parser.add_argument("--use-lr-schedule", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-name-bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-ha-mismatch-residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ha-mismatch-start", type=int, default=16)
    parser.add_argument("--ha-mismatch-end", type=int, default=345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_limit == 0 or args.sample_limit < -1:
        raise ValueError("--sample-limit must be -1 (all rows) or a positive integer")
    if args.artificial_warmup_epochs < 0:
        raise ValueError("--artificial-warmup-epochs must be >= 0")
    if args.artificial_train_csv is not None and args.artificial_warmup_epochs == 0:
        raise ValueError("--artificial-train-csv requires --artificial-warmup-epochs > 0")
    if args.ha_mismatch_end <= args.ha_mismatch_start:
        raise ValueError("--ha-mismatch-end must be greater than --ha-mismatch-start")

    sample_limit = None if args.sample_limit < 0 else int(args.sample_limit)
    frames = load_fixed_split_frames(args.data_dir, sample_limit=sample_limit)
    artificial_train_frame = (
        load_artificial_train_frame(args.artificial_train_csv, sample_limit=sample_limit)
        if args.artificial_train_csv is not None
        else None
    )
    feature_frames = dict(frames)
    if artificial_train_frame is not None:
        feature_frames["artificial_train"] = artificial_train_frame
    embedding_files = required_embedding_files(feature_frames)
    embedding_dir = validate_embedding_files(args.embedding_dir, embedding_files)
    vocabs = build_feature_vocabs(feature_frames)
    name_vocabs = build_name_vocabs(frames["train"]) if args.use_name_bias else None
    paths = prepare_output_dir(args.output_dir)

    set_seed(args.seed)
    device = torch.device(args.device)
    embeddings = load_embedding(str(embedding_dir), files=embedding_files)
    hidden_size = infer_hidden_size(embeddings)
    cache = GpuEmbeddingCache(
        cpu_store=embeddings,
        device=device,
        max_bytes=int(float(args.gpu_cache_gb) * 1024**3),
    )
    loaders = build_dataloaders(
        frames,
        vocabs,
        args.batch_size,
        name_vocabs=name_vocabs,
        ha_mismatch_start=args.ha_mismatch_start,
        ha_mismatch_end=args.ha_mismatch_end,
    )
    artificial_warmup_loader = build_artificial_warmup_train_loader(
        frames["train"],
        artificial_train_frame,
        vocabs,
        args.batch_size,
        enabled=args.artificial_warmup_epochs > 0,
        name_vocabs=name_vocabs,
        ha_mismatch_start=args.ha_mismatch_start,
        ha_mismatch_end=args.ha_mismatch_end,
    )

    model_config = MetricHAAntigenicModelConfig(
        hidden_size=hidden_size,
        latent_dim=args.latent_dim,
        passage_vocab_size=len(vocabs.passage_to_id),
        passage_pair_vocab_size=len(vocabs.passage_to_id) ** 2,
        subtype_vocab_size=len(vocabs.subtype_to_id),
        na_lambda_max=args.na_lambda_max,
        serum_name_vocab_size=(len(name_vocabs.serum_to_id) + 1 if name_vocabs is not None else 0),
        virus_name_vocab_size=(len(name_vocabs.virus_to_id) + 1 if name_vocabs is not None else 0),
        ha_mismatch_dim=(
            args.ha_mismatch_end - args.ha_mismatch_start
            if args.use_ha_mismatch_residual
            else 0
        ),
    )
    model = MetricHAAntigenicModel(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        if args.use_lr_schedule
        else None
    )
    writer = SummaryWriter(log_dir=str(paths["tensorboard"])) if SummaryWriter is not None else NullWriter()

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model.__class__.__name__,
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "embedding_dir": str(embedding_dir),
        "output_dir": str(paths["root"]),
        "row_counts": {name: len(frame) for name, frame in frames.items()},
        "embedding_file_count": len(embedding_files),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": str(device),
        "gpu_cache_gb": args.gpu_cache_gb,
        "sample_limit": sample_limit,
        "seed": args.seed,
        "artificial_train_csv": (
            str(Path(args.artificial_train_csv).expanduser().resolve())
            if args.artificial_train_csv is not None
            else None
        ),
        "artificial_warmup_epochs": args.artificial_warmup_epochs,
        "reverse_warmup_epochs": args.artificial_warmup_epochs,
        "artificial_train_rows": len(artificial_train_frame) if artificial_train_frame is not None else 0,
        "warmup_train_rows": len(artificial_warmup_loader.dataset) if artificial_warmup_loader else 0,
        "model_config": model_config.__dict__,
        "passage_to_id": vocabs.passage_to_id,
        "subtype_to_id": vocabs.subtype_to_id,
        "use_lr_schedule": args.use_lr_schedule,
        "use_name_bias": args.use_name_bias,
        "use_ha_mismatch_residual": args.use_ha_mismatch_residual,
        "ha_mismatch_start": args.ha_mismatch_start,
        "ha_mismatch_end": args.ha_mismatch_end,
        "name_vocab_sizes": {
            "serum": len(name_vocabs.serum_to_id) if name_vocabs is not None else 0,
            "virus": len(name_vocabs.virus_to_id) if name_vocabs is not None else 0,
        },
    }
    paths["config"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    if name_vocabs is not None:
        (paths["root"] / "name_vocabs.json").write_text(
            json.dumps(
                {"serum_to_id": name_vocabs.serum_to_id, "virus_to_id": name_vocabs.virus_to_id},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    with paths["log"].open("w", encoding="utf-8") as log_file:
        log_file.write("===== RUN CONFIG START =====\n")
        json.dump(run_config, log_file, indent=2, ensure_ascii=False)
        log_file.write("\n===== RUN CONFIG END =====\n\n")

    best_mse = float("inf")
    early_stop_counter = 0
    warmup_steps = (
        len(artificial_warmup_loader) * min(args.artificial_warmup_epochs, args.epochs)
        if artificial_warmup_loader
        else 0
    )
    real_steps = len(loaders["train"]) * max(
        args.epochs - min(args.artificial_warmup_epochs, args.epochs),
        0,
    )
    progress_bar = tqdm(range(warmup_steps + real_steps))
    try:
        for epoch in range(args.epochs):
            train_loader = train_loader_for_epoch(
                loaders["train"],
                artificial_warmup_loader,
                epoch_index=epoch,
                reverse_warmup_epochs=args.artificial_warmup_epochs,
            )
            train_loss = train_one_epoch(model, train_loader, device, cache, optimizer, progress_bar=progress_bar)
            valid_metrics = evaluate(model, loaders["valid"], device, cache)
            test_metrics = evaluate(model, loaders["test"], device, cache)
            test_without_name_metrics = None
            if args.use_name_bias:
                test_without_name_metrics = evaluate(model, loaders["test"], device, cache, use_name_bias=False)
            writer.add_scalar("loss/train", train_loss, epoch)
            for split_name, metrics in (("valid", valid_metrics), ("test", test_metrics)):
                for metric_name, value in metrics.items():
                    writer.add_scalar(f"{split_name}/{metric_name}", value, epoch)
            if test_without_name_metrics is not None:
                for metric_name, value in test_without_name_metrics.items():
                    writer.add_scalar(f"test_without_name/{metric_name}", value, epoch)

            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid": valid_metrics,
                "test": test_metrics,
            }
            if test_without_name_metrics is not None:
                epoch_metrics["test_without_name"] = test_without_name_metrics
            with paths["metrics"].open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
            with paths["log"].open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")

            if valid_metrics["mse"] < best_mse:
                best_mse = valid_metrics["mse"]
                early_stop_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": model_config.__dict__,
                        "passage_to_id": vocabs.passage_to_id,
                        "subtype_to_id": vocabs.subtype_to_id,
                        "name_vocabs": (
                            {"serum_to_id": name_vocabs.serum_to_id, "virus_to_id": name_vocabs.virus_to_id}
                            if name_vocabs is not None
                            else None
                        ),
                        "epoch": epoch + 1,
                        "valid_metrics": valid_metrics,
                    },
                    paths["checkpoints"] / "best_model.pth",
                )
            else:
                early_stop_counter += 1
                if early_stop_counter >= args.patience:
                    with paths["log"].open("a", encoding="utf-8") as log_file:
                        log_file.write("Early stopping\n")
                    break
            if scheduler is not None:
                scheduler.step()
    finally:
        progress_bar.close()
        writer.close()


if __name__ == "__main__":
    main()
