#!/usr/bin/env python3
"""
Build dataset splits with three modes used in the paper:
- titer  : row-level random split
- strain : group-level split by strain key
- serum  : group-level split by serum key
- season: rolling season holdout split by sheet prefix

Protocol-oriented output layout:
  <splits_root>/<protocol_version>/<dataset_version_id>/<mode>/<split_id>/
    - train.csv
    - valid.csv
    - test.csv
    - manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build titer/strain/serum/season dataset splits (standalone protocol tool)."
    )
    parser.add_argument(
        "--input-csv",
        default="",
        help="Input CSV path (legacy style). If omitted, use --raw-version-dir.",
    )
    parser.add_argument(
        "--raw-version-dir",
        default="",
        help="Raw dataset version directory (recommended), e.g. data/raw/r2026_03_27_mix_h1h3",
    )
    parser.add_argument(
        "--raw-csv-name",
        type=str,
        default="source.csv",
        help="CSV file name under --raw-version-dir.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="default_dataset",
        help="User-defined dataset name (logical group), e.g. hi_mix_h1h3.",
    )
    parser.add_argument(
        "--dataset-version-id",
        type=str,
        default="",
        help="Dataset version id. If empty and --raw-version-dir is set, use that directory name.",
    )
    parser.add_argument(
        "--dataset-description",
        type=str,
        default="",
        help="Optional human-readable dataset description recorded in metadata.",
    )
    parser.add_argument(
        "--dataset-meta-json",
        type=str,
        default="dataset_meta.json",
        help="Metadata file name under --raw-version-dir.",
    )
    parser.add_argument(
        "--splits-root",
        type=str,
        default="data/splits",
        help="Split root directory.",
    )
    parser.add_argument(
        "--protocol-version",
        type=str,
        default="v1",
        help="Split protocol version name.",
    )
    parser.add_argument(
        "--dataset-scoped-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write to <splits-root>/<protocol-version>/<split-id>/<mode>/ instead of "
            "<splits-root>/<protocol-version>/<dataset-version-id>/<split-id>/<mode>/."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument(
        "--group-valid",
        choices=("true", "false"),
        default="false",
        help="Whether validation should be split by group for strain/serum modes.",
    )
    parser.add_argument(
        "--strain-col",
        type=str,
        default="seq_id_c",
        help="Column name used for strain-group split.",
    )
    parser.add_argument(
        "--serum-col",
        type=str,
        default="seq_id_a",
        help="Column name used for serum-group split.",
    )
    parser.add_argument(
        "--pre-split-agg-cols",
        type=str,
        default="",
        help=(
            "Comma-separated columns used with serumPassCat/virusPassCat to aggregate labels "
            "before splitting test. Defaults to --serum-col,--strain-col."
        ),
    )
    parser.add_argument(
        "--train-agg-cols",
        type=str,
        default="",
        help=(
            "Comma-separated columns used with serumPassCat/virusPassCat to aggregate labels "
            "after test holdout and before validation split. Defaults to --serum-col,--strain-col."
        ),
    )
    parser.add_argument(
        "--split-modes",
        type=str,
        default="titer,strain,serum",
        help="Comma-separated split modes from {titer,strain,serum,season}.",
    )
    parser.add_argument(
        "--season-col",
        type=str,
        default="sheet",
        help="Column whose prefix before '-' defines the season key for season mode.",
    )
    parser.add_argument(
        "--test-seasons",
        type=str,
        default="",
        help="Comma-separated season keys to hold out one at a time in season mode.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="",
        help="Optional unique row id column for overlap checks (default: row index).",
    )
    parser.add_argument(
        "--split-id",
        type=str,
        default="",
        help="Optional manual split id. If empty, auto-generate.",
    )
    return parser.parse_args()


def _validate_ratios(test_ratio: float, valid_ratio: float) -> None:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("--test-ratio must be in (0, 1).")
    if not (0.0 <= valid_ratio < 1.0):
        raise ValueError("--valid-ratio must be in [0, 1).")
    if test_ratio + valid_ratio >= 1.0:
        raise ValueError("test_ratio + valid_ratio must be < 1.")


def _parse_modes(modes_arg: str) -> List[str]:
    valid = {"titer", "strain", "serum", "season"}
    modes = [m.strip().lower() for m in modes_arg.split(",") if m.strip()]
    if not modes:
        raise ValueError("--split-modes is empty.")
    unknown = [m for m in modes if m not in valid]
    if unknown:
        raise ValueError(f"Unknown modes in --split-modes: {unknown}. Valid: {sorted(valid)}")
    dedup = list(dict.fromkeys(modes))
    return dedup


def _parse_column_list(cols_arg: str) -> List[str]:
    return [col.strip() for col in cols_arg.split(",") if col.strip()]


def _parse_test_seasons(seasons_arg: str) -> List[str]:
    seasons = [season.strip() for season in seasons_arg.split(",") if season.strip()]
    return list(dict.fromkeys(seasons))


def _aggregation_subset_columns(cols_arg: str, fallback_cols_arg: str) -> List[str]:
    required = ["serumPassCat", "virusPassCat"]
    configured = _parse_column_list(cols_arg) or _parse_column_list(fallback_cols_arg)
    return list(dict.fromkeys([*required, *configured]))


def _aggregate_duplicates(df: pd.DataFrame, subset: List[str]) -> Tuple[pd.DataFrame, Dict]:
    before_count = len(df)
    agg_map = {col: "first" for col in df.columns if col not in subset}
    agg_map["label"] = "mean"
    grouped = df.groupby(subset, as_index=False, dropna=False, sort=False).agg(agg_map)
    grouped = grouped[[col for col in df.columns if col in grouped.columns]]
    duplicate_count = before_count - len(grouped)
    report = {
        "subset": subset,
        "before": int(before_count),
        "after": int(len(grouped)),
        "duplicates_aggregated": int(duplicate_count),
        "label_aggregation": "mean",
        "other_columns": "first",
    }
    return grouped, report


def _split_counts(n: int, test_ratio: float, valid_ratio: float) -> Tuple[int, int, int]:
    n_test = int(round(n * test_ratio))
    n_valid = int(round(n * valid_ratio))
    # Keep at least 1 sample for train whenever possible
    if n - n_test - n_valid <= 0 and n >= 3:
        n_valid = max(0, n_valid - 1)
    n_train = n - n_test - n_valid
    return n_train, n_valid, n_test


def _season_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _add_season_key(df: pd.DataFrame, season_col: str, key_col: str = "_season_key") -> pd.DataFrame:
    if season_col not in df.columns:
        raise KeyError(f"Required season column not found in CSV: {season_col!r}")
    keyed = df.copy()
    keyed[key_col] = keyed[season_col].fillna("").astype(str).str.split("-", n=1).str[0]
    if (keyed[key_col] == "").any():
        raise ValueError(f"Season column {season_col!r} contains empty values after prefix parsing.")
    return keyed


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_input_and_dataset_id(args: argparse.Namespace) -> Tuple[Path, str, Path | None]:
    raw_version_dir = Path(args.raw_version_dir).expanduser().resolve() if args.raw_version_dir else None

    if raw_version_dir is not None:
        input_csv = (raw_version_dir / args.raw_csv_name).resolve()
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        dataset_version_id = args.dataset_version_id.strip() or raw_version_dir.name
        return input_csv, dataset_version_id, raw_version_dir

    if not args.input_csv:
        raise ValueError("Either --input-csv or --raw-version-dir must be provided.")

    input_csv = Path(args.input_csv).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    dataset_version_id = args.dataset_version_id.strip()
    if not dataset_version_id:
        raise ValueError("When using --input-csv, --dataset-version-id is required.")
    return input_csv, dataset_version_id, None


def _auto_split_id(dataset_version_id: str, seed: int, test_ratio: float, valid_ratio: float) -> str:
    train_ratio = 1.0 - test_ratio - valid_ratio
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{dataset_version_id}__seed{seed}"
        f"__tr{train_ratio:.2f}_va{valid_ratio:.2f}_te{test_ratio:.2f}"
        f"__{stamp}"
    )


def _maybe_write_dataset_meta(
    raw_version_dir: Path | None,
    dataset_name: str,
    dataset_version_id: str,
    input_csv: Path,
    args: argparse.Namespace,
    row_count: int,
) -> Dict:
    meta = {
        "dataset_name": dataset_name,
        "dataset_version_id": dataset_version_id,
        "description": args.dataset_description,
        "raw_csv": str(input_csv),
        "raw_csv_sha256": _sha256(input_csv),
        "row_count": int(row_count),
        "created_at": _now_iso(),
    }
    if raw_version_dir is None:
        return meta

    meta_path = raw_version_dir / args.dataset_meta_json
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                existing.update({k: v for k, v in meta.items() if v not in ("", None)})
                meta = existing
        except json.JSONDecodeError:
            pass
    else:
        meta["raw_version_dir"] = str(raw_version_dir)

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _split_rows(
    df: pd.DataFrame,
    rng: np.random.Generator,
    test_ratio: float,
    valid_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    order = np.arange(n)
    rng.shuffle(order)
    n_train, n_valid, n_test = _split_counts(n, test_ratio, valid_ratio)
    i1 = n_train
    i2 = n_train + n_valid
    train_idx = order[:i1]
    valid_idx = order[i1:i2]
    test_idx = order[i2 : i2 + n_test]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[valid_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def _split_by_group(
    df: pd.DataFrame,
    group_cols: List[str],
    rng: np.random.Generator,
    test_ratio: float,
    valid_ratio: float,
    group_valid: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_values = df[group_cols].fillna("<NA>").astype(str).agg("|".join, axis=1)
    groups = group_values.unique().tolist()
    groups = np.array(groups, dtype=object)
    rng.shuffle(groups)

    if group_valid:
        n_train_g, n_valid_g, n_test_g = _split_counts(len(groups), test_ratio, valid_ratio)
        g1 = n_train_g
        g2 = n_train_g + n_valid_g
        train_groups = set(groups[:g1].tolist())
        valid_groups = set(groups[g1:g2].tolist())
        test_groups = set(groups[g2 : g2 + n_test_g].tolist())
        train_df = df[group_values.isin(train_groups)].copy().reset_index(drop=True)
        valid_df = df[group_values.isin(valid_groups)].copy().reset_index(drop=True)
        test_df = df[group_values.isin(test_groups)].copy().reset_index(drop=True)
    else:
        _, _, n_test_g = _split_counts(len(groups), test_ratio, 0.0)
        test_groups = set(groups[-n_test_g:].tolist()) if n_test_g else set()
        remaining_df = df[~group_values.isin(test_groups)].copy().reset_index(drop=True)
        train_df, valid_df, _ = _split_rows(
            remaining_df, rng, test_ratio=0.0, valid_ratio=valid_ratio / (1.0 - test_ratio)
        )
        test_df = df[group_values.isin(test_groups)].copy().reset_index(drop=True)
    return train_df, valid_df, test_df


def _split_train_pool_and_test(
    df: pd.DataFrame,
    mode: str,
    rng: np.random.Generator,
    test_ratio: float,
    strain_cols: List[str],
    serum_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str] | None]:
    if mode == "titer":
        train_pool_df, _, test_df = _split_rows(df, rng, test_ratio, valid_ratio=0.0)
        return train_pool_df, test_df, None
    if mode == "strain":
        train_pool_df, _, test_df = _split_by_group(
            df, strain_cols, rng, test_ratio, valid_ratio=0.0, group_valid=True
        )
        return train_pool_df, test_df, strain_cols
    if mode == "serum":
        train_pool_df, _, test_df = _split_by_group(
            df, serum_cols, rng, test_ratio, valid_ratio=0.0, group_valid=True
        )
        return train_pool_df, test_df, serum_cols
    raise ValueError(f"Unknown mode: {mode}")


def _split_train_and_valid(
    df: pd.DataFrame,
    rng: np.random.Generator,
    test_ratio: float,
    valid_ratio: float,
    group_cols: List[str] | None,
    group_valid: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    adjusted_valid_ratio = valid_ratio / (1.0 - test_ratio)
    if group_cols is not None and group_valid:
        train_df, valid_df, _ = _split_by_group(
            df,
            group_cols,
            rng,
            test_ratio=0.0,
            valid_ratio=adjusted_valid_ratio,
            group_valid=True,
        )
        return train_df, valid_df
    train_df, valid_df, _ = _split_rows(
        df,
        rng,
        test_ratio=0.0,
        valid_ratio=adjusted_valid_ratio,
    )
    return train_df, valid_df


def _choose_previous_valid_season(
    prior_seasons: List[str],
    valid_ratio: float,
) -> List[str]:
    if valid_ratio <= 0.0 or len(prior_seasons) < 2:
        return []
    return [prior_seasons[-1]]


def _run_one_season_split(
    df: pd.DataFrame,
    test_season: str,
    seed: int,
    test_ratio: float,
    valid_ratio: float,
    season_col: str,
    season_key_col: str,
    id_col: str,
    train_agg_subset: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    all_seasons = sorted(df[season_key_col].dropna().astype(str).unique().tolist(), key=_season_sort_key)
    if test_season not in all_seasons:
        raise ValueError(f"Requested test season {test_season!r} not found in {season_col!r}.")
    prior_seasons = [season for season in all_seasons if _season_sort_key(season) < _season_sort_key(test_season)]
    future_seasons = [season for season in all_seasons if _season_sort_key(season) > _season_sort_key(test_season)]
    if not prior_seasons:
        raise ValueError(f"Requested test season {test_season!r} has no earlier seasons for training.")

    train_pool = df[df[season_key_col].isin(prior_seasons)].copy().reset_index(drop=True)
    train_pool_before_agg = len(train_pool)
    train_pool, train_agg_report = _aggregate_duplicates(
        train_pool,
        list(dict.fromkeys([season_key_col, *train_agg_subset])),
    )
    valid_seasons = _choose_previous_valid_season(prior_seasons, valid_ratio)
    train_seasons = [season for season in prior_seasons if season not in set(valid_seasons)]
    train_df = train_pool[train_pool[season_key_col].isin(train_seasons)].copy().reset_index(drop=True)
    valid_df = train_pool[train_pool[season_key_col].isin(valid_seasons)].copy().reset_index(drop=True)
    test_df = df[df[season_key_col] == test_season].copy().reset_index(drop=True)

    overlap_stats = _check_no_overlap(train_df, valid_df, test_df, id_col)
    report = {
        "mode": "season",
        "seed": seed,
        "ratios": {
            "train_ratio": 1.0 - test_ratio - valid_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
        },
        "counts": {
            "all": int(len(train_df) + len(valid_df) + len(test_df)),
            "pre_train_pool_aggregation": int(train_pool_before_agg),
            "post_train_pool_aggregation": int(len(train_pool)),
            "train": int(len(train_df)),
            "valid": int(len(valid_df)),
            "test": int(len(test_df)),
            "unused": int(df[df[season_key_col].isin(future_seasons)].shape[0]),
        },
        "overlap": overlap_stats,
        "valid_split_strategy": "previous_season",
        "season_split_policy": {
            "train": "seasons earlier than the validation season",
            "valid": "immediate previous season before the test season",
            "test": "requested held-out season",
        },
        "train_pool_aggregation": train_agg_report,
        "season_col": season_col,
        "season_key_rule": "prefix_before_dash",
        "test_season": test_season,
        "train_seasons": train_seasons,
        "valid_seasons": valid_seasons,
        "unused_seasons": future_seasons,
        "season_counts": {
            season: int(count)
            for season, count in df.groupby(season_key_col).size().sort_index(key=lambda s: s.map(_season_sort_key)).items()
        },
        "group_columns": [season_key_col],
        "group_counts": {
            "all": int(len(all_seasons)),
            "train": int(len(train_seasons)),
            "valid": int(len(valid_seasons)),
            "test": 1,
            "unused": int(len(future_seasons)),
        },
        "group_leakage": _check_group_leakage(train_df, valid_df, test_df, [season_key_col]),
    }

    return (
        train_df.drop(columns=[season_key_col]).reset_index(drop=True),
        valid_df.drop(columns=[season_key_col]).reset_index(drop=True),
        test_df.drop(columns=[season_key_col]).reset_index(drop=True),
        report,
    )


def _check_no_overlap(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
) -> Dict[str, int]:
    if id_col and id_col in train_df.columns:
        a = set(train_df[id_col].tolist())
        b = set(valid_df[id_col].tolist())
        c = set(test_df[id_col].tolist())
    else:
        # Fallback: hash all row values
        a = set(pd.util.hash_pandas_object(train_df, index=False).tolist())
        b = set(pd.util.hash_pandas_object(valid_df, index=False).tolist())
        c = set(pd.util.hash_pandas_object(test_df, index=False).tolist())
    return {
        "train_valid_overlap": len(a & b),
        "train_test_overlap": len(a & c),
        "valid_test_overlap": len(b & c),
    }


def _check_group_leakage(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_cols: List[str],
) -> Dict[str, int]:
    ta = set(train_df[group_cols].fillna("<NA>").astype(str).agg("|".join, axis=1).tolist())
    va = set(valid_df[group_cols].fillna("<NA>").astype(str).agg("|".join, axis=1).tolist())
    sa = set(test_df[group_cols].fillna("<NA>").astype(str).agg("|".join, axis=1).tolist())
    return {
        "train_valid_group_overlap": len(ta & va),
        "train_test_group_overlap": len(ta & sa),
        "valid_test_group_overlap": len(va & sa),
    }


def _write_mode_outputs(
    mode: str,
    mode_dir: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report: Dict,
) -> None:
    mode_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(mode_dir / "train.csv", index=False)
    valid_df.to_csv(mode_dir / "valid.csv", index=False)
    test_df.to_csv(mode_dir / "test.csv", index=False)
    (mode_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_one_mode(
    df: pd.DataFrame,
    mode: str,
    seed: int,
    test_ratio: float,
    valid_ratio: float,
    strain_col: str,
    serum_col: str,
    id_col: str,
    group_valid: bool,
    train_agg_subset: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    rng = np.random.default_rng(seed)
    strain_cols = _parse_column_list(strain_col)
    serum_cols = _parse_column_list(serum_col)
    train_pool_df, test_df, group_cols = _split_train_pool_and_test(
        df=df,
        mode=mode,
        rng=rng,
        test_ratio=test_ratio,
        strain_cols=strain_cols,
        serum_cols=serum_cols,
    )
    train_pool_before_agg = len(train_pool_df)
    train_pool_df, train_agg_report = _aggregate_duplicates(train_pool_df, train_agg_subset)
    train_df, valid_df = _split_train_and_valid(
        train_pool_df,
        rng,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio,
        group_cols=group_cols,
        group_valid=group_valid,
    )

    overlap_stats = _check_no_overlap(train_df, valid_df, test_df, id_col)
    report = {
        "mode": mode,
        "seed": seed,
        "ratios": {
            "train_ratio": 1.0 - test_ratio - valid_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
        },
        "counts": {
            "all": int(len(train_df) + len(valid_df) + len(test_df)),
            "pre_train_pool_aggregation": int(train_pool_before_agg),
            "post_train_pool_aggregation": int(len(train_pool_df)),
            "train": int(len(train_df)),
            "valid": int(len(valid_df)),
            "test": int(len(test_df)),
        },
        "overlap": overlap_stats,
        "valid_split_strategy": "group" if group_cols is not None and group_valid else "random",
        "train_pool_aggregation": train_agg_report,
    }

    if group_cols is not None:
        report["group_columns"] = group_cols
        report["group_counts"] = {
            "all": int(df[group_cols].drop_duplicates().shape[0]),
            "train": int(train_df[group_cols].drop_duplicates().shape[0]),
            "valid": int(valid_df[group_cols].drop_duplicates().shape[0]),
            "test": int(test_df[group_cols].drop_duplicates().shape[0]),
        }
        report["group_leakage"] = _check_group_leakage(train_df, valid_df, test_df, group_cols)

    return train_df, valid_df, test_df, report


def main() -> None:
    args = parse_args()
    _validate_ratios(args.test_ratio, args.valid_ratio)
    split_modes = _parse_modes(args.split_modes)
    test_seasons = _parse_test_seasons(args.test_seasons)
    if "season" in split_modes and not test_seasons:
        raise ValueError("--test-seasons is required when --split-modes includes season.")
    input_csv, dataset_version_id, raw_version_dir = _resolve_input_and_dataset_id(args)
    splits_root = Path(args.splits_root).expanduser().resolve()
    dataset_name = args.dataset_name.strip()
    if not dataset_name:
        raise ValueError("--dataset-name cannot be empty.")

    raw_df = pd.read_csv(input_csv)
    if len(raw_df) == 0:
        raise ValueError("Input CSV is empty.")
    nan_label_count = int(raw_df["label"].isna().sum())
    labeled_df = raw_df.dropna(subset=["label"]).reset_index(drop=True)
    print(f"Dropped {nan_label_count} rows with NaN label.")
    if len(labeled_df) == 0:
        raise ValueError("No labeled rows remain after dropping NaN labels.")

    fallback_agg_cols = ",".join(_parse_column_list(args.serum_col) + _parse_column_list(args.strain_col))
    pre_split_agg_subset = _aggregation_subset_columns(args.pre_split_agg_cols, fallback_agg_cols)
    train_agg_subset = _aggregation_subset_columns(args.train_agg_cols, fallback_agg_cols)
    required_cols = list(
        dict.fromkeys(
            [
                *pre_split_agg_subset,
                *train_agg_subset,
                *_parse_column_list(args.strain_col),
                *_parse_column_list(args.serum_col),
                *([args.season_col] if "season" in split_modes else []),
            ]
        )
    )
    for col in required_cols:
        if col not in labeled_df.columns:
            raise KeyError(f"Required column not found in CSV: {col!r}")
    df, pre_split_agg_report = _aggregate_duplicates(labeled_df, pre_split_agg_subset)
    print(
        "Dropped "
        f"{pre_split_agg_report['duplicates_aggregated']} duplicate rows before splitting test "
        f"using subset: {', '.join(pre_split_agg_subset)}; aggregated duplicate labels by mean."
    )
    season_df = None
    season_pre_split_agg_report = None
    season_key_col = "_season_key"
    if "season" in split_modes:
        season_keyed_df = _add_season_key(labeled_df, args.season_col, season_key_col)
        season_df, season_pre_split_agg_report = _aggregate_duplicates(
            season_keyed_df,
            list(dict.fromkeys([season_key_col, *pre_split_agg_subset])),
        )
        print(
            "Dropped "
            f"{season_pre_split_agg_report['duplicates_aggregated']} duplicate rows before season splitting "
            f"using subset: {', '.join(season_pre_split_agg_report['subset'])}; "
            "aggregated duplicate labels by mean."
        )

    dataset_meta = _maybe_write_dataset_meta(
        raw_version_dir=raw_version_dir,
        dataset_name=dataset_name,
        dataset_version_id=dataset_version_id,
        input_csv=input_csv,
        args=args,
        row_count=len(df),
    )

    split_id = args.split_id.strip() or _auto_split_id(
        dataset_version_id=dataset_version_id,
        seed=args.seed,
        test_ratio=args.test_ratio,
        valid_ratio=args.valid_ratio,
    )

    if args.dataset_scoped_output:
        base_out = (splits_root / args.protocol_version).resolve()
    else:
        base_out = (splits_root / args.protocol_version / dataset_version_id).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    for mode in split_modes:
        if mode == "season":
            assert season_df is not None
            assert season_pre_split_agg_report is not None
            for test_season in test_seasons:
                train_df, valid_df, test_df, report = _run_one_season_split(
                    df=season_df,
                    test_season=test_season,
                    seed=args.seed,
                    test_ratio=args.test_ratio,
                    valid_ratio=args.valid_ratio,
                    season_col=args.season_col,
                    season_key_col=season_key_col,
                    id_col=args.id_col,
                    train_agg_subset=train_agg_subset,
                )
                mode_dir = base_out / split_id / "season" / test_season
                report.update(
                    {
                        "protocol_version": args.protocol_version,
                        "dataset_name": dataset_name,
                        "dataset_version_id": dataset_version_id,
                        "split_id": split_id,
                        "source": {
                            "input_csv": str(input_csv),
                            "input_csv_sha256": _sha256(input_csv),
                            "raw_version_dir": str(raw_version_dir) if raw_version_dir else "",
                        },
                        "dataset_meta": dataset_meta,
                        "pre_split_aggregation": season_pre_split_agg_report,
                        "paths": {
                            "train_csv": str((mode_dir / "train.csv").resolve()),
                            "valid_csv": str((mode_dir / "valid.csv").resolve()),
                            "test_csv": str((mode_dir / "test.csv").resolve()),
                        },
                        "created_at": _now_iso(),
                    }
                )
                _write_mode_outputs(mode, mode_dir, train_df, valid_df, test_df, report)
                print(
                    f"[season/{test_season}] all={report['counts']['all']} "
                    f"train={report['counts']['train']} "
                    f"valid={report['counts']['valid']} "
                    f"test={report['counts']['test']} -> {mode_dir}"
                )
            continue

        train_df, valid_df, test_df, report = _run_one_mode(
            df=df,
            mode=mode,
            seed=args.seed,
            test_ratio=args.test_ratio,
            valid_ratio=args.valid_ratio,
            strain_col=args.strain_col,
            serum_col=args.serum_col,
            id_col=args.id_col,
            group_valid=args.group_valid == "true",
            train_agg_subset=train_agg_subset,
        )
        mode_dir = base_out / split_id / mode
        report.update(
            {
                "protocol_version": args.protocol_version,
                "dataset_name": dataset_name,
                "dataset_version_id": dataset_version_id,
                "split_id": split_id,
                "source": {
                    "input_csv": str(input_csv),
                    "input_csv_sha256": _sha256(input_csv),
                    "raw_version_dir": str(raw_version_dir) if raw_version_dir else "",
                },
                "dataset_meta": dataset_meta,
                "pre_split_aggregation": pre_split_agg_report,
                "paths": {
                    "train_csv": str((mode_dir / "train.csv").resolve()),
                    "valid_csv": str((mode_dir / "valid.csv").resolve()),
                    "test_csv": str((mode_dir / "test.csv").resolve()),
                },
                "created_at": _now_iso(),
            }
        )
        _write_mode_outputs(mode, mode_dir, train_df, valid_df, test_df, report)
        print(
            f"[{mode}] all={report['counts']['all']} "
            f"train={report['counts']['train']} "
            f"valid={report['counts']['valid']} "
            f"test={report['counts']['test']} -> {mode_dir}"
        )

    print(f"Done. Split root: {base_out}")


if __name__ == "__main__":
    main()
