#!/usr/bin/env python3
"""
Build dataset splits with three modes used in the paper:
- titer  : row-level random split
- strain : group-level split by strain key
- serum  : group-level split by serum key

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
        description="Build titer/strain/serum dataset splits (standalone protocol tool)."
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split ratio.")
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
        "--split-modes",
        type=str,
        default="titer,strain,serum",
        help="Comma-separated split modes from {titer,strain,serum}.",
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
    valid = {"titer", "strain", "serum"}
    modes = [m.strip().lower() for m in modes_arg.split(",") if m.strip()]
    if not modes:
        raise ValueError("--split-modes is empty.")
    unknown = [m for m in modes if m not in valid]
    if unknown:
        raise ValueError(f"Unknown modes in --split-modes: {unknown}. Valid: {sorted(valid)}")
    dedup = list(dict.fromkeys(modes))
    return dedup


def _split_counts(n: int, test_ratio: float, valid_ratio: float) -> Tuple[int, int, int]:
    n_test = int(round(n * test_ratio))
    n_valid = int(round(n * valid_ratio))
    # Keep at least 1 sample for train whenever possible
    if n - n_test - n_valid <= 0 and n >= 3:
        n_valid = max(0, n_valid - 1)
    n_train = n - n_test - n_valid
    return n_train, n_valid, n_test


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
    group_col: str,
    rng: np.random.Generator,
    test_ratio: float,
    valid_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col!r}")
    groups = df[group_col].fillna("<NA>").astype(str).unique().tolist()
    groups = np.array(groups, dtype=object)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_train_g, n_valid_g, n_test_g = _split_counts(n_groups, test_ratio, valid_ratio)
    g1 = n_train_g
    g2 = n_train_g + n_valid_g
    train_groups = set(groups[:g1].tolist())
    valid_groups = set(groups[g1:g2].tolist())
    test_groups = set(groups[g2 : g2 + n_test_g].tolist())

    group_values = df[group_col].fillna("<NA>").astype(str)
    train_df = df[group_values.isin(train_groups)].copy().reset_index(drop=True)
    valid_df = df[group_values.isin(valid_groups)].copy().reset_index(drop=True)
    test_df = df[group_values.isin(test_groups)].copy().reset_index(drop=True)
    return train_df, valid_df, test_df


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
    group_col: str,
) -> Dict[str, int]:
    ta = set(train_df[group_col].fillna("<NA>").astype(str).tolist())
    va = set(valid_df[group_col].fillna("<NA>").astype(str).tolist())
    sa = set(test_df[group_col].fillna("<NA>").astype(str).tolist())
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    rng = np.random.default_rng(seed)

    if mode == "titer":
        train_df, valid_df, test_df = _split_rows(df, rng, test_ratio, valid_ratio)
        group_col = None
    elif mode == "strain":
        train_df, valid_df, test_df = _split_by_group(df, strain_col, rng, test_ratio, valid_ratio)
        group_col = strain_col
    elif mode == "serum":
        train_df, valid_df, test_df = _split_by_group(df, serum_col, rng, test_ratio, valid_ratio)
        group_col = serum_col
    else:
        raise ValueError(f"Unknown mode: {mode}")

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
            "all": int(len(df)),
            "train": int(len(train_df)),
            "valid": int(len(valid_df)),
            "test": int(len(test_df)),
        },
        "overlap": overlap_stats,
    }

    if group_col is not None:
        report["group_column"] = group_col
        report["group_counts"] = {
            "all": int(df[group_col].fillna("<NA>").astype(str).nunique()),
            "train": int(train_df[group_col].fillna("<NA>").astype(str).nunique()),
            "valid": int(valid_df[group_col].fillna("<NA>").astype(str).nunique()),
            "test": int(test_df[group_col].fillna("<NA>").astype(str).nunique()),
        }
        report["group_leakage"] = _check_group_leakage(train_df, valid_df, test_df, group_col)

    return train_df, valid_df, test_df, report


def main() -> None:
    args = parse_args()
    _validate_ratios(args.test_ratio, args.valid_ratio)
    split_modes = _parse_modes(args.split_modes)
    input_csv, dataset_version_id, raw_version_dir = _resolve_input_and_dataset_id(args)
    splits_root = Path(args.splits_root).expanduser().resolve()
    dataset_name = args.dataset_name.strip()
    if not dataset_name:
        raise ValueError("--dataset-name cannot be empty.")

    df = pd.read_csv(input_csv)
    if len(df) == 0:
        raise ValueError("Input CSV is empty.")
    dataset_meta = _maybe_write_dataset_meta(
        raw_version_dir=raw_version_dir,
        dataset_name=dataset_name,
        dataset_version_id=dataset_version_id,
        input_csv=input_csv,
        args=args,
        row_count=len(df),
    )

    required_cols: List[str] = [args.strain_col, args.serum_col]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column not found in CSV: {col!r}")

    split_id = args.split_id.strip() or _auto_split_id(
        dataset_version_id=dataset_version_id,
        seed=args.seed,
        test_ratio=args.test_ratio,
        valid_ratio=args.valid_ratio,
    )

    base_out = (splits_root / args.protocol_version / dataset_name / dataset_version_id).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    for mode in split_modes:
        train_df, valid_df, test_df, report = _run_one_mode(
            df=df,
            mode=mode,
            seed=args.seed,
            test_ratio=args.test_ratio,
            valid_ratio=args.valid_ratio,
            strain_col=args.strain_col,
            serum_col=args.serum_col,
            id_col=args.id_col,
        )
        mode_dir = base_out / mode / split_id
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
