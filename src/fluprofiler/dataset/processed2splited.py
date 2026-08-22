#!/usr/bin/env python3
"""Build train/valid/test splits from processed/source.csv.

Example:

cd /home/chenyh/workspace/fluProfiler

python src/fluprofiler/dataset/processed2splited.py \
  --dataset-dir /home/chenyh/workspace/fluProfiler/data/dataset/H1H3_HA1 \
  --seed 42 \
  --test-ratio 0.1 \
  --valid-ratio 0.1 \
  --valid-split group \
  --serum-col seq_a \
  --strain-col seq_c \
  --split-modes titer,strain,serum,season \
  --season-col sheet \
  --test-seasons 39,40,41,42,42,44
  
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SOURCE_GROUP_COLS = ["seq_a", "seq_c", "serumPassCat", "virusPassCat"]
VALID_MODES = {"titer", "strain", "serum", "season"}
SplitFrames = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
SeedSplits = dict[int, SplitFrames]


def parse_cols(text: str) -> list[str]:
    cols = [col.strip() for col in text.split(",") if col.strip()]
    if not cols:
        raise ValueError("Column list cannot be empty")
    return list(dict.fromkeys(cols))


def parse_modes(text: str) -> list[str]:
    modes = [mode.lower() for mode in parse_cols(text)]
    unknown = [mode for mode in modes if mode not in VALID_MODES]
    if unknown:
        raise ValueError(f"Unknown split mode(s): {', '.join(unknown)}")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split processed/source.csv into train/valid/test CSVs."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--seed", type=lambda value: [int(seed) for seed in value.split(",")], default=[42])
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--valid-split", choices=("random", "group"), default="group")
    parser.add_argument("--serum-col", default="seq_a")
    parser.add_argument("--strain-col", default="seq_c")
    parser.add_argument("--split-modes", default="titer,strain,serum")
    parser.add_argument("--season-col", default="sheet")
    parser.add_argument("--test-seasons", default="")
    return parser.parse_args()


def validate_ratios(test_ratio: float, valid_ratio: float) -> None:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio must be in (0, 1)")
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be in [0, 1)")
    if test_ratio + valid_ratio >= 1.0:
        raise ValueError("--test-ratio + --valid-ratio must be < 1")


def split_counts(n: int, test_ratio: float, valid_ratio: float) -> tuple[int, int, int]:
    n_test = int(round(n * test_ratio))
    n_valid = int(round(n * valid_ratio))
    if n - n_test - n_valid <= 0 and n >= 3:
        n_valid = max(0, n_valid - 1)
    return n - n_test - n_valid, n_valid, n_test


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def load_source(dataset_dir: Path) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    source_csv = dataset_dir / "processed" / "source.csv"
    if not source_csv.is_file():
        raise FileNotFoundError(f"Input source.csv not found: {source_csv}")

    df = pd.read_csv(source_csv, keep_default_na=False)
    if "label" not in df.columns:
        raise ValueError("source.csv must contain a label column")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No labeled rows remain after dropping invalid labels")

    config_path = dataset_dir / "processed" / "source_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Input source_config.json not found: {config_path}")

    cols = json.loads(config_path.read_text(encoding="utf-8")).get("group_cols", [])
    group_cols = [str(col) for col in cols] if isinstance(cols, list) and cols else []

    duplicate_check = {"checked": False, "group_cols": [], "duplicate_rows": 0}
    if group_cols:
        require_columns(df, group_cols)
        duplicate_rows = int(df.duplicated(group_cols).sum())
        if duplicate_rows:
            raise ValueError(
                "source.csv has duplicate rows by source_config group_cols; "
                "rerun raw2processed.py or inspect source.csv"
            )
        duplicate_check = {
            "checked": True,
            "group_cols": group_cols,
            "duplicate_rows": duplicate_rows,
        }
    return df, source_csv, duplicate_check


def group_key(frame: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    return frame[group_cols].fillna("<NA>").astype(str).agg("|".join, axis=1)


def split_rows(df: pd.DataFrame, seeds: list[int], test_ratio: float, valid_ratio: float) -> SeedSplits:
    n_train, n_valid, n_test = split_counts(len(df), test_ratio, valid_ratio)
    splits: SeedSplits = {}
    for seed in seeds:
        order = list(df.index)
        random.Random(seed).shuffle(order)
        train_idx = order[:n_train]
        valid_idx = order[n_train : n_train + n_valid]
        test_idx = order[n_train + n_valid : n_train + n_valid + n_test]
        splits[seed] = (
            df.loc[train_idx].reset_index(drop=True),
            df.loc[valid_idx].reset_index(drop=True),
            df.loc[test_idx].reset_index(drop=True),
        )
    return splits


def split_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    seeds: list[int],
    test_ratio: float,
    valid_ratio: float,
    valid_split: str,
) -> SeedSplits:
    keys = group_key(df, group_cols)
    groups = keys.drop_duplicates().tolist()
    splits: SeedSplits = {}

    if valid_split == "group":
        n_train, n_valid, n_test = split_counts(len(groups), test_ratio, valid_ratio)
        for seed in seeds:
            shuffled_groups = groups.copy()
            random.Random(seed).shuffle(shuffled_groups)
            train_groups = set(shuffled_groups[:n_train])
            valid_groups = set(shuffled_groups[n_train : n_train + n_valid])
            test_groups = set(shuffled_groups[n_train + n_valid : n_train + n_valid + n_test])
            splits[seed] = (
                df[keys.isin(train_groups)].reset_index(drop=True),
                df[keys.isin(valid_groups)].reset_index(drop=True),
                df[keys.isin(test_groups)].reset_index(drop=True),
            )
        return splits

    _, _, n_test = split_counts(len(groups), test_ratio, 0.0)
    for seed in seeds:
        shuffled_groups = groups.copy()
        random.Random(seed).shuffle(shuffled_groups)
        test_groups = set(shuffled_groups[-n_test:]) if n_test else set()
        train_df, valid_df, _ = split_rows(
            df[~keys.isin(test_groups)].reset_index(drop=True),
            [seed],
            test_ratio=0.0,
            valid_ratio=valid_ratio / (1.0 - test_ratio),
        )[seed]
        splits[seed] = train_df, valid_df, df[keys.isin(test_groups)].reset_index(drop=True)
    return splits


def season_key(value: Any) -> str:
    return str(value).split("-", 1)[0]


def season_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def split_by_season(
    df: pd.DataFrame,
    test_season: str,
    season_col: str,
    valid_ratio: float,
) -> tuple[SplitFrames, dict[str, Any]]:
    keyed = df.copy()
    keyed["_season_key"] = keyed[season_col].map(season_key)
    if (keyed["_season_key"] == "").any():
        raise ValueError(f"{season_col} contains empty season keys")

    seasons = sorted(keyed["_season_key"].drop_duplicates().tolist(), key=season_sort_key)
    if test_season not in seasons:
        raise ValueError(f"Requested test season {test_season!r} not found in {season_col}")

    prior = [season for season in seasons if season_sort_key(season) < season_sort_key(test_season)]
    future = [season for season in seasons if season_sort_key(season) > season_sort_key(test_season)]
    if not prior:
        raise ValueError(f"Requested test season {test_season!r} has no earlier training season")

    valid_seasons = [prior[-1]] if valid_ratio > 0.0 and len(prior) >= 2 else []
    train_seasons = [season for season in prior if season not in set(valid_seasons)]
    frames = (
        keyed[keyed["_season_key"].isin(train_seasons)].drop(columns=["_season_key"]).reset_index(drop=True),
        keyed[keyed["_season_key"].isin(valid_seasons)].drop(columns=["_season_key"]).reset_index(drop=True),
        keyed[keyed["_season_key"] == test_season].drop(columns=["_season_key"]).reset_index(drop=True),
    )
    return frames, {
        "test_season": test_season,
        "train_seasons": train_seasons,
        "valid_seasons": valid_seasons,
        "unused_seasons": future,
        "valid_split_strategy": "previous_season",
        "season_col": season_col,
        "season_key_rule": "prefix_before_dash",
    }


def row_overlap(frames: SplitFrames) -> dict[str, int]:
    train, valid, test = [
        set(pd.util.hash_pandas_object(frame, index=False).tolist()) for frame in frames
    ]
    return {
        "train_valid_overlap": len(train & valid),
        "train_test_overlap": len(train & test),
        "valid_test_overlap": len(valid & test),
    }


def group_overlap(frames: SplitFrames, group_cols: list[str]) -> dict[str, int]:
    train, valid, test = [set(group_key(frame, group_cols).tolist()) for frame in frames]
    return {
        "train_valid_group_overlap": len(train & valid),
        "train_test_group_overlap": len(train & test),
        "valid_test_group_overlap": len(valid & test),
    }


def auto_split_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def manifest(
    mode: str,
    args: argparse.Namespace,
    source_csv: Path,
    duplicate_check: dict[str, Any],
    frames: SplitFrames,
    group_cols: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    train_df, valid_df, test_df = frames
    out = {
        "mode": mode,
        "dataset_name": args.dataset_dir.name,
        "seed": args.seed if seed is None else seed,
        "ratios": {
            "train_ratio": 1.0 - args.test_ratio - args.valid_ratio,
            "valid_ratio": args.valid_ratio,
            "test_ratio": args.test_ratio,
        },
        "valid_split": args.valid_split,
        "counts": {
            "all": int(len(train_df) + len(valid_df) + len(test_df)),
            "train": int(len(train_df)),
            "valid": int(len(valid_df)),
            "test": int(len(test_df)),
        },
        "source": {"input_csv": str(source_csv)},
        "source_duplicate_check": duplicate_check,
        "overlap": row_overlap(frames),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if group_cols:
        out["group_columns"] = group_cols
        out["group_leakage"] = group_overlap(frames, group_cols)
    if extra:
        out.update(extra)
    return out


def write_split(
    out_dir: Path,
    label: str,
    mode: str,
    args: argparse.Namespace,
    source_csv: Path,
    duplicate_check: dict[str, Any],
    frames: SplitFrames,
    group_cols: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    seed: int | None = None,
) -> None:
    train_df, valid_df, test_df = frames
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    valid_df.to_csv(out_dir / "valid.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            manifest(mode, args, source_csv, duplicate_check, frames, group_cols, extra, seed),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[{label}] train={len(train_df)} valid={len(valid_df)} test={len(test_df)} -> {out_dir}")


def main() -> None:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    validate_ratios(args.test_ratio, args.valid_ratio)

    modes = parse_modes(args.split_modes)
    strain_cols = parse_cols(args.strain_col)
    serum_cols = parse_cols(args.serum_col)
    test_seasons = parse_cols(args.test_seasons) if args.test_seasons else []
    if "season" in modes and not test_seasons:
        raise ValueError("--test-seasons is required when --split-modes includes season")

    df, source_csv, duplicate_check = load_source(args.dataset_dir)
    required = []
    if "strain" in modes:
        required.extend(strain_cols)
    if "serum" in modes:
        required.extend(serum_cols)
    if "season" in modes:
        required.append(args.season_col)
    require_columns(df, required)

    split_root = args.dataset_dir / "splited" / auto_split_id()
    context = (args, source_csv, duplicate_check)

    for mode in modes:
        if mode == "titer":
            for seed, frames in split_rows(
                df, args.seed, args.test_ratio, args.valid_ratio
            ).items():
                write_split(
                    split_root / mode / f"seed_{seed}",
                    f"{mode}/seed_{seed}",
                    mode,
                    *context,
                    frames,
                    seed=seed,
                )
        elif mode in {"strain", "serum"}:
            group_cols = strain_cols if mode == "strain" else serum_cols
            for seed, frames in split_by_group(
                df, group_cols, args.seed, args.test_ratio, args.valid_ratio, args.valid_split
            ).items():
                write_split(
                    split_root / mode / f"seed_{seed}",
                    f"{mode}/seed_{seed}",
                    mode,
                    *context,
                    frames,
                    group_cols,
                    seed=seed,
                )
        else:
            for test_season in test_seasons:
                frames, extra = split_by_season(df, test_season, args.season_col, args.valid_ratio)
                write_split(
                    split_root / "season" / test_season,
                    f"season/{test_season}",
                    mode,
                    *context,
                    frames,
                    [args.season_col],
                    extra,
                )

    print(f"Done. Split root: {split_root}")


if __name__ == "__main__":
    main()
