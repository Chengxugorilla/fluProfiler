#!/usr/bin/env python3
"""
Train an AdaBoost pairwise antigenic-distance baseline from fixed train/test CSVs.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


SPLIT_FILENAMES = ("train.csv", "test.csv")
REQUIRED_COLUMNS = {
    "serumHA",
    "virusHA",
    "serumPassCat",
    "virusPassCat",
    "serumName",
    "virusName",
    "Type",
    "label",
}
META_COLUMNS = ("serumPassCat", "virusPassCat", "serumName", "virusName")


def resolve_split_data_dir(data_dir: Path, test_season: str | int | None = None) -> Path:
    resolved = Path(data_dir).expanduser()
    season = "" if test_season is None else str(test_season).strip()
    if not season:
        return resolved
    if resolved.name == season:
        return resolved
    return resolved / season


def sequence_mismatch_vector(serum_ha: Any, virus_ha: Any, start: int = 16, end: int = 345) -> np.ndarray:
    serum = "" if pd.isna(serum_ha) else str(serum_ha)
    virus = "" if pd.isna(virus_ha) else str(virus_ha)
    length = max(0, end - start)
    values = np.zeros(length, dtype=np.float32)
    for out_idx, pos in enumerate(range(start, end)):
        if pos >= len(serum) or pos >= len(virus):
            values[out_idx] = np.nan
        else:
            values[out_idx] = 0.0 if serum[pos] == virus[pos] else 1.0
    return values


class PairwiseFeatureBuilder:
    def __init__(self, sequence_start: int = 16, sequence_end: int = 345):
        self.sequence_start = sequence_start
        self.sequence_end = sequence_end
        self.encoder: OneHotEncoder | None = None

    def _sequence_features(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.empty((0, max(0, self.sequence_end - self.sequence_start)), dtype=np.float32)
        return np.vstack(
            [
                sequence_mismatch_vector(row.serumHA, row.virusHA, self.sequence_start, self.sequence_end)
                for row in frame.itertuples(index=False)
            ]
        )

    def _meta_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[:, META_COLUMNS].fillna("unknown").astype(str)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        meta_features = self.encoder.fit_transform(self._meta_frame(frame))
        return np.hstack([self._sequence_features(frame), meta_features]).astype(np.float32)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("PairwiseFeatureBuilder must be fitted before transform")
        if frame.empty:
            sequence_features = self._sequence_features(frame)
            meta_count = sum(len(categories) for categories in self.encoder.categories_)
            return np.empty((0, sequence_features.shape[1] + meta_count), dtype=np.float32)
        meta_features = self.encoder.transform(self._meta_frame(frame))
        return np.hstack([self._sequence_features(frame), meta_features]).astype(np.float32)


def load_fixed_split_frames(
    data_dir: Path,
    sample_limit: int | None = None,
    include_valid: bool = False,
) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    split_filenames = ("train.csv", "valid.csv", "test.csv") if include_valid else SPLIT_FILENAMES
    missing_files = [filename for filename in split_filenames if not (data_dir / filename).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required split file(s) under {data_dir}: {', '.join(missing_files)}"
        )

    frames = {name.removesuffix(".csv"): pd.read_csv(data_dir / name) for name in split_filenames}
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


def filter_frames_by_subtypes(
    frames: dict[str, pd.DataFrame],
    subtypes: list[str],
) -> dict[str, pd.DataFrame]:
    selected = set(subtypes)
    return {
        name: frame.loc[frame["Type"].astype(str).isin(selected)].reset_index(drop=True)
        for name, frame in frames.items()
    }


def validate_training_subtypes(
    frames: dict[str, pd.DataFrame],
    subtypes: list[str],
    merge_valid_into_train: bool,
) -> None:
    training_frames = [frames["train"]]
    if merge_valid_into_train:
        training_frames.append(frames["valid"])
    effective_train = pd.concat(training_frames, axis=0, ignore_index=True)
    available = sorted(effective_train["Type"].dropna().astype(str).unique().tolist())
    missing = [subtype for subtype in subtypes if subtype not in available]
    if missing:
        available_text = ", ".join(available) if available else "none"
        raise ValueError(
            f"Requested subtype(s) missing from effective training data: {', '.join(missing)}; "
            f"available: {available_text}"
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


def build_model(random_state: int, fast: bool = False) -> AdaBoostRegressor:
    if fast:
        return AdaBoostRegressor(n_estimators=50, random_state=random_state)
    return AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=7040, max_features=0.419171992638116),
        n_estimators=410,
        learning_rate=1.26852534318595,
        random_state=random_state,
    )


def finite_row_mask(features: np.ndarray) -> np.ndarray:
    return np.isfinite(features).all(axis=1)


def without_name_frame(frame: pd.DataFrame) -> pd.DataFrame:
    anonymized = frame.copy()
    anonymized.loc[:, "serumName"] = ""
    anonymized.loc[:, "virusName"] = ""
    return anonymized


def train_subtype_model(
    frames: dict[str, pd.DataFrame],
    subtype: str,
    sequence_start: int,
    sequence_end: int,
    random_state: int,
    fast: bool,
) -> dict[str, Any]:
    subtype_frames = {name: frame[frame["Type"] == subtype].reset_index(drop=True) for name, frame in frames.items()}
    train_frame = subtype_frames["train"]
    if train_frame.empty:
        raise ValueError(f"No training rows found for subtype {subtype}")

    builder = PairwiseFeatureBuilder(sequence_start=sequence_start, sequence_end=sequence_end)
    train_features = builder.fit_transform(train_frame)
    train_mask = finite_row_mask(train_features)
    train_features = train_features[train_mask]
    train_labels = train_frame.loc[train_mask, "label"].to_numpy(dtype=float)

    model = build_model(random_state=random_state, fast=fast)
    model.fit(train_features, train_labels)

    predictions: dict[str, pd.DataFrame] = {}
    split_metrics: dict[str, dict[str, float]] = {}
    row_counts: dict[str, int] = {}
    dropped_rows: dict[str, int] = {}

    for split_name, frame in subtype_frames.items():
        features = builder.transform(frame)
        mask = finite_row_mask(features)
        usable = frame.loc[mask].copy()
        pred = model.predict(features[mask]) if len(usable) else np.asarray([], dtype=float)
        usable["prediction"] = pred
        usable["reference"] = usable["label"].astype(float)
        usable["subtype_model"] = subtype
        if split_name == "test":
            without_name_features = builder.transform(without_name_frame(frame))
            without_name_pred = (
                model.predict(without_name_features[mask]) if len(usable) else np.asarray([], dtype=float)
            )
            usable["prediction_without_name"] = without_name_pred
        predictions[split_name] = usable
        split_metrics[split_name] = regression_metrics(
            usable["reference"].tolist(),
            usable["prediction"].tolist(),
        )
        if split_name == "test":
            split_metrics[split_name]["without_name"] = regression_metrics(
                usable["reference"].tolist(),
                usable["prediction_without_name"].tolist(),
            )
        row_counts[split_name] = int(len(usable))
        dropped_rows[split_name] = int(len(frame) - len(usable))

    return {
        "subtype": subtype,
        "model": model,
        "feature_builder": builder,
        "predictions": predictions,
        "metrics": split_metrics,
        "row_counts": row_counts,
        "dropped_rows": dropped_rows,
        "feature_count": int(train_features.shape[1]),
    }


def combine_predictions(subtype_results: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    combined: dict[str, list[pd.DataFrame]] = {}
    for result in subtype_results:
        for split_name, prediction_frame in result["predictions"].items():
            combined.setdefault(split_name, [])
            combined[split_name].append(prediction_frame)
    return {
        split_name: pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
        for split_name, frames in combined.items()
    }


def metrics_by_split_and_subtype(predictions: dict[str, pd.DataFrame]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split_name, frame in predictions.items():
        output[split_name] = {
            "overall": regression_metrics(frame["reference"].tolist(), frame["prediction"].tolist())
            if len(frame)
            else regression_metrics([], []),
            "by_subtype": {},
        }
        if split_name == "test" and "prediction_without_name" in frame.columns:
            output[split_name]["overall_without_name"] = (
                regression_metrics(frame["reference"].tolist(), frame["prediction_without_name"].tolist())
                if len(frame)
                else regression_metrics([], [])
            )
        for subtype, subtype_frame in frame.groupby("Type"):
            subtype_metrics = {
                "with_name": regression_metrics(
                    subtype_frame["reference"].tolist(),
                    subtype_frame["prediction"].tolist(),
                )
            }
            if split_name == "test" and "prediction_without_name" in subtype_frame.columns:
                subtype_metrics["without_name"] = regression_metrics(
                    subtype_frame["reference"].tolist(),
                    subtype_frame["prediction_without_name"].tolist(),
                )
            output[split_name]["by_subtype"][str(subtype)] = subtype_metrics
    return output


def prepare_output_dir(output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": output_dir,
        "config": output_dir / "run_config.json",
        "metrics": output_dir / "metrics.json",
        "predictions": output_dir / "predictions.csv",
        "models": output_dir / "models.pkl",
        "log": output_dir / "log.txt",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AdaBoost train/test pairwise baseline.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--test-season",
        type=str,
        default="",
        help="Optional season subdirectory under --data-dir, e.g. --data-dir .../season --test-season 26.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subtypes", nargs="+", default=[])
    parser.add_argument("--sample-limit", type=int, default=-1)
    parser.add_argument("--sequence-start", type=int, default=16)
    parser.add_argument("--sequence-end", type=int, default=345)
    parser.add_argument("--random-state", type=int, default=100)
    parser.add_argument("--fast", action="store_true", default=False)
    parser.add_argument(
        "--merge-valid-into-train",
        action="store_true",
        default=False,
        help="Train subtype models on train.csv plus valid.csv, and evaluate on test.csv.",
    )
    args = parser.parse_args(argv)
    if not args.subtypes:
        parser.error("--subtypes requires at least one subtype")
    return args


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_split_data_dir(args.data_dir, args.test_season)
    if args.sample_limit == 0 or args.sample_limit < -1:
        raise ValueError("--sample-limit must be -1 (all rows) or a positive integer")
    if args.sequence_end <= args.sequence_start:
        raise ValueError("--sequence-end must be greater than --sequence-start")

    sample_limit = None if args.sample_limit < 0 else int(args.sample_limit)
    frames = load_fixed_split_frames(
        args.data_dir,
        include_valid=args.merge_valid_into_train,
    )
    subtypes = list(dict.fromkeys(args.subtypes))
    validate_training_subtypes(frames, subtypes, args.merge_valid_into_train)
    frames = filter_frames_by_subtypes(frames, subtypes)
    if sample_limit is not None:
        frames = {name: frame.iloc[:sample_limit].copy() for name, frame in frames.items()}
    original_row_counts = {name: int(len(frame)) for name, frame in frames.items()}
    if args.merge_valid_into_train:
        frames["train"] = pd.concat([frames["train"], frames["valid"]], axis=0, ignore_index=True)
        frames.pop("valid")
    paths = prepare_output_dir(args.output_dir)
    subtype_results = [
        train_subtype_model(
            frames,
            subtype=subtype,
            sequence_start=args.sequence_start,
            sequence_end=args.sequence_end,
            random_state=args.random_state,
            fast=args.fast,
        )
        for subtype in subtypes
    ]
    predictions = combine_predictions(subtype_results)
    metrics = metrics_by_split_and_subtype(predictions)

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": "adaboost_pairwise_ha_mismatch",
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "output_dir": str(paths["root"]),
        "sample_limit": sample_limit,
        "sequence_start": args.sequence_start,
        "sequence_end": args.sequence_end,
        "random_state": args.random_state,
        "fast": args.fast,
        "merge_valid_into_train": bool(args.merge_valid_into_train),
        "test_season": args.test_season,
        "subtypes": subtypes,
        "input_row_counts": original_row_counts,
        "training_row_count": int(len(frames["train"])),
        "usable_row_counts": {name: int(len(frame)) for name, frame in predictions.items()},
        "subtype_row_counts": {result["subtype"]: result["row_counts"] for result in subtype_results},
        "subtype_dropped_rows": {result["subtype"]: result["dropped_rows"] for result in subtype_results},
        "feature_count_by_subtype": {result["subtype"]: result["feature_count"] for result in subtype_results},
    }
    result_payload = {"config": run_config, "metrics": metrics}

    paths["config"].write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["metrics"].write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.concat(
        [frame.assign(split=split_name) for split_name, frame in predictions.items()],
        axis=0,
        ignore_index=True,
    ).to_csv(paths["predictions"], index=False)
    with paths["models"].open("wb") as model_file:
        pickle.dump(
            {
                result["subtype"]: {
                    "model": result["model"],
                    "feature_builder": result["feature_builder"],
                }
                for result in subtype_results
            },
            model_file,
        )
    with paths["log"].open("w", encoding="utf-8") as log_file:
        log_file.write("===== RUN CONFIG START =====\n")
        json.dump(run_config, log_file, indent=2, ensure_ascii=False)
        log_file.write("\n===== RUN CONFIG END =====\n\n")
        json.dump(metrics, log_file, indent=2, ensure_ascii=False)
        log_file.write("\n")


if __name__ == "__main__":
    main()
