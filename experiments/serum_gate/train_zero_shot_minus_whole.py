"""Train the standard SerumGate-Minus model on a whole CSV dataset.

This entry point reuses the model, data, optimization, and checkpoint pipeline
from ``train_zero_shot_minus.py`` and forces the standard low-rank HA attention
pooling mode. When a data directory contains ``whole.csv`` but no complete
fixed split, the file is used as the sole training set without requiring or
evaluating validation/test CSV files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__:
    from . import train_zero_shot_minus as base_trainer
else:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import train_zero_shot_minus as base_trainer


def _load_whole_csv_frames(
    data_dir: Path,
    sample_limit: int | None = None,
    refit_train_valid: bool = False,
    type_filter: str = "",
) -> dict[str, pd.DataFrame]:
    """Load whole.csv as train and create empty validation/test frames in memory."""
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    whole_csv = data_dir / "whole.csv"
    if not whole_csv.is_file():
        raise FileNotFoundError(f"Missing training file: {whole_csv}")

    train = pd.read_csv(whole_csv)
    base_trainer.validate_frame_columns(train, whole_csv)
    empty = train.iloc[0:0].copy().reset_index(drop=True)
    frames = {
        "train": train.reset_index(drop=True),
        "valid": empty.copy(),
        "test": empty.copy(),
    }
    if type_filter:
        frames = base_trainer.filter_frames_by_type(frames, type_filter)
    if sample_limit is not None:
        frames = {name: frame.iloc[:sample_limit].copy() for name, frame in frames.items()}
    if refit_train_valid:
        frames = base_trainer.merge_train_valid_for_refit(frames)
    return frames


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run the standard SerumGate-Minus pipeline in optional whole-data mode."""
    training_args = argparse.Namespace(**vars(args))
    training_args.ha_pooling = "lowrank_attention"

    data_dir = Path(training_args.data_dir).expanduser().resolve()
    has_whole_csv = (data_dir / "whole.csv").is_file()
    has_fixed_splits = all((data_dir / name).is_file() for name in base_trainer.SPLIT_FILENAMES)
    whole_only_mode = has_whole_csv and not has_fixed_splits

    original_split_loader = base_trainer.load_fixed_split_frames
    if whole_only_mode:
        training_args.refit_train_valid = True
        training_args.skip_test_eval = True
        base_trainer.load_fixed_split_frames = _load_whole_csv_frames
    try:
        return base_trainer.run_training(training_args)
    finally:
        base_trainer.load_fixed_split_frames = original_split_loader


def main() -> None:
    run_training(base_trainer.parse_args())


if __name__ == "__main__":
    main()
