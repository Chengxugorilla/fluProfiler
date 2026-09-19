#!/usr/bin/env python3
"""
Train and evaluate the HA-only v2 model against an explicit fixed split.

The data directory must contain train.csv, valid.csv, and test.csv. This
entrypoint intentionally never creates its own validation split.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_THIS_FILE.parent))
sys.path.append(str(_REPO_ROOT / "src" / "fluprofiler"))
sys.path.append(str(_REPO_ROOT / "src"))
sys.path.append(str(_REPO_ROOT / "experiments" / "reverse_tests"))

from experiment_tools import (  # noqa: E402
    GpuEmbeddingCache,
    log_epoch_to_file,
    log_metrics_to_tensorboard,
    setup_optimizer,
)
from data.loaders import load_embedding  # noqa: E402
from models.architectures import fluProfiler_Config  # noqa: E402
from fluprofiler.models_v2 import fluProfiler_HA_only_distance_v2, fluProfiler_HA_only_v2  # noqa: E402
from fluprofiler.utils.model_args import build_model_args  # noqa: E402
from train_v2_ha_only import HAOnlyDataset, evaluate_step, train_step  # noqa: E402


SPLIT_FILENAMES = ("train.csv", "valid.csv", "test.csv")
REQUIRED_COLUMNS = {"seq_id_a", "seq_id_c", "serumPassCat", "virusPassCat", "label"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing_columns:
            raise ValueError(
                f"{data_dir / (split_name + '.csv')} is missing required column(s): "
                f"{', '.join(missing_columns)}"
            )

    if sample_limit is not None:
        frames = {name: frame.iloc[:sample_limit].copy() for name, frame in frames.items()}
    return frames


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


def build_name_vocabs(train_frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    missing = [col for col in ("serumName", "virusName") if col not in train_frame.columns]
    if missing:
        raise ValueError(f"Name bias requires column(s): {', '.join(missing)}")
    serum_names = sorted(train_frame["serumName"].fillna("").astype(str).unique().tolist())
    virus_names = sorted(train_frame["virusName"].fillna("").astype(str).unique().tolist())
    return {
        "serum": {name: idx + 1 for idx, name in enumerate(serum_names) if name},
        "virus": {name: idx + 1 for idx, name in enumerate(virus_names) if name},
    }


def build_datasets(
    frames: dict[str, pd.DataFrame], add_special_token: bool, name_vocabs=None
) -> dict[str, HAOnlyDataset]:
    return {
        name: HAOnlyDataset(frame, add_special_token=add_special_token, name_vocabs=name_vocabs)
        for name, frame in frames.items()
    }


def build_dataloaders(
    frames: dict[str, pd.DataFrame], batch_size: int, add_special_token: bool, name_vocabs=None
) -> dict[str, DataLoader]:
    datasets = build_datasets(frames, add_special_token=add_special_token, name_vocabs=name_vocabs)
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "valid": DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }


def build_ha_only_model(flu_config, flu_args, model_impl: str, name_vocabs=None):
    model_impl = model_impl.lower()
    if name_vocabs is not None and model_impl != "distance":
        raise ValueError("--use-name-bias is currently supported only with --model-impl distance")
    if model_impl == "v2":
        return fluProfiler_HA_only_v2(config=flu_config, args=flu_args)
    if model_impl == "distance":
        serum_vocab_size = len(name_vocabs["serum"]) + 1 if name_vocabs is not None else 0
        virus_vocab_size = len(name_vocabs["virus"]) + 1 if name_vocabs is not None else 0
        return fluProfiler_HA_only_distance_v2(
            config=flu_config,
            args=flu_args,
            serum_name_vocab_size=serum_vocab_size,
            virus_name_vocab_size=virus_vocab_size,
        )
    raise ValueError(f"Unsupported model_impl: {model_impl}")


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
        "name_vocabs": output_dir / "name_vocabs.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HA-only v2 from fixed train/valid/test CSV files.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=_REPO_ROOT / "configs" / "config_dict.json",
        help="Architecture configuration JSON used to construct the HA-only model.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gpu-cache-gb", type=float, default=20)
    parser.add_argument("--sample-limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-impl", choices=("v2", "distance"), default="v2")
    parser.add_argument("--add-special-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-lr-schedule", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-name-bias", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_limit == 0 or args.sample_limit < -1:
        raise ValueError("--sample-limit must be -1 (all rows) or a positive integer")

    data_dir = args.data_dir.expanduser().resolve()
    sample_limit = None if args.sample_limit < 0 else args.sample_limit
    frames = load_fixed_split_frames(data_dir, sample_limit=sample_limit)
    embedding_files = required_embedding_files(frames)
    embedding_dir = validate_embedding_files(args.embedding_dir, embedding_files)
    model_config = args.model_config.expanduser().resolve()
    if not model_config.is_file():
        raise FileNotFoundError(f"Model configuration file does not exist: {model_config}")
    paths = prepare_output_dir(args.output_dir)

    set_seed(args.seed)
    device = torch.device(args.device)
    name_vocabs = build_name_vocabs(frames["train"]) if args.use_name_bias else None
    loaders = build_dataloaders(frames, args.batch_size, args.add_special_token, name_vocabs=name_vocabs)
    embeddings = load_embedding(str(embedding_dir), files=embedding_files)
    gpu_cache = GpuEmbeddingCache(
        cpu_store=embeddings,
        device=device,
        max_bytes=int(args.gpu_cache_gb * 1024**3),
    )

    config_dict = json.loads(model_config.read_text(encoding="utf-8"))
    flu_config = fluProfiler_Config.from_dict(config_dict)
    flu_args = build_model_args(flu_config)
    model = build_ha_only_model(flu_config, flu_args, model_impl=args.model_impl, name_vocabs=name_vocabs).to(device)
    optimizer = setup_optimizer(model, flu_args, lr=args.learning_rate)
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        if args.use_lr_schedule
        else None
    )

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model.__class__.__name__,
        "model_impl": args.model_impl,
        "data_dir": str(data_dir),
        "train_csv": str(data_dir / "train.csv"),
        "valid_csv": str(data_dir / "valid.csv"),
        "test_csv": str(data_dir / "test.csv"),
        "embedding_dir": str(embedding_dir),
        "model_config": str(model_config),
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
        "add_special_token": args.add_special_token,
        "use_lr_schedule": args.use_lr_schedule,
        "use_name_bias": args.use_name_bias,
        "name_vocab_sizes": {
            "serum": len(name_vocabs["serum"]) if name_vocabs is not None else 0,
            "virus": len(name_vocabs["virus"]) if name_vocabs is not None else 0,
        },
    }
    paths["config"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    if name_vocabs is not None:
        paths["name_vocabs"].write_text(json.dumps(name_vocabs, indent=2, ensure_ascii=False), encoding="utf-8")
    with paths["log"].open("w", encoding="utf-8") as log_file:
        log_file.write("===== RUN CONFIG START =====\n")
        json.dump(run_config, log_file, indent=2, ensure_ascii=False)
        log_file.write("\n===== RUN CONFIG END =====\n\n")

    writer = SummaryWriter(log_dir=str(paths["tensorboard"]))
    progress_bar = tqdm(range(len(loaders["train"]) * args.epochs))
    best_mse = float("inf")
    early_stop_counter = 0
    try:
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for batch in loaders["train"]:
                loss = train_step(model, batch, device, gpu_cache, model_impl=args.model_impl)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            train_loss = float(np.mean(losses))

            valid_metrics = evaluate_step(model, loaders["valid"], device, gpu_cache, model_impl=args.model_impl)
            test_metrics = evaluate_step(model, loaders["test"], device, gpu_cache, model_impl=args.model_impl)
            test_without_name_metrics = None
            if args.use_name_bias:
                test_without_name_metrics = evaluate_step(
                    model,
                    loaders["test"],
                    device,
                    gpu_cache,
                    model_impl=args.model_impl,
                    use_name_bias=False,
                )
            log_metrics_to_tensorboard(writer, epoch, train_loss, valid_metrics, test_metrics)
            log_epoch_to_file(paths["log"], epoch, args.epochs, train_loss, valid_metrics, test_metrics)

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

            if valid_metrics["mse"] < best_mse:
                best_mse = valid_metrics["mse"]
                early_stop_counter = 0
                torch.save(model, paths["checkpoints"] / "best_model.pth")
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
