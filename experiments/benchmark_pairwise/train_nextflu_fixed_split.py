#!/usr/bin/env python3
"""
Train and evaluate the legacy Nextflu baseline on fixed split CSVs.

The underlying Nextflu implementation is kept in the historical notebook
`experiments/reverse_tests/train_Nextflu.ipynb`; this wrapper makes it usable
from the current fixed-split benchmark layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
NEXTFLU_NOTEBOOK = REPO_ROOT / "experiments" / "reverse_tests" / "train_Nextflu.ipynb"
REQUIRED_COLUMNS = {"serumHA", "virusHA", "serumName", "virusName", "Type", "label"}


def resolve_split_data_dir(data_dir: Path, test_season: str | int | None = None) -> Path:
    resolved = Path(data_dir).expanduser()
    season = "" if test_season is None else str(test_season).strip()
    if not season:
        return resolved
    if resolved.name == season:
        return resolved
    return resolved / season


def load_nextflu_namespace() -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "nextflu_legacy"}
    sys.path.insert(0, str(REPO_ROOT / "src"))
    notebook = nbformat.read(NEXTFLU_NOTEBOOK, as_version=4)
    exec("".join(notebook.cells[0].source), namespace)
    return namespace


def load_split_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir).expanduser().resolve()
    frames = {}
    for split in ("train", "valid", "test"):
        path = data_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required split file: {path}")
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        frames[split] = frame
    return frames


def regression_metrics(reference: pd.Series, prediction: pd.Series) -> dict[str, float]:
    y = reference.to_numpy(dtype=float)
    pred = prediction.to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    if len(y) == 0:
        return {"mae": 0.0, "mse": 0.0, "pearson": 0.0, "spearman": 0.0, "r2": 0.0, "rows": 0}
    diff = pred - y
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    r2_den = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if r2_den == 0.0 else float(1.0 - np.sum(diff**2) / r2_den)
    pearson = 0.0
    if len(y) > 1 and float(np.std(y)) > 0.0 and float(np.std(pred)) > 0.0:
        pearson = float(np.corrcoef(y, pred)[0, 1])
    spearman = 0.0
    if len(y) > 1:
        y_rank = pd.Series(y).rank(method="average").to_numpy()
        pred_rank = pd.Series(pred).rank(method="average").to_numpy()
        if float(np.std(y_rank)) > 0.0 and float(np.std(pred_rank)) > 0.0:
            spearman = float(np.corrcoef(y_rank, pred_rank)[0, 1])
    return {"mae": mae, "mse": mse, "pearson": pearson, "spearman": spearman, "r2": r2, "rows": int(len(y))}


def _jsonable_model(model: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        section: {str(key): float(value) for key, value in values.items()}
        for section, values in model.items()
    }


def run(
    data_dir: Path,
    output_dir: Path,
    subtypes: list[str] | None = None,
    overwrite: bool = False,
    merge_valid_into_train: bool = False,
    test_season: str = "",
) -> dict[str, Any]:
    data_dir = resolve_split_data_dir(data_dir, test_season)
    frames = load_split_frames(data_dir)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty; use --overwrite or choose another directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    namespace = load_nextflu_namespace()
    train_nextflu_model = namespace["train_Nextflu_model"]
    nextflu_predict = namespace["nextflu_predict"]

    train = frames["train"].copy()
    if merge_valid_into_train:
        train = pd.concat([train, frames["valid"]], axis=0, ignore_index=True)
    test = frames["test"].copy()
    subtypes = subtypes or sorted(train["Type"].dropna().astype(str).unique().tolist())
    models = {subtype: train_nextflu_model(train.copy(), subtype) for subtype in subtypes}

    test["pred_with_name"] = np.nan
    test["pred_without_name"] = np.nan
    for subtype, model in models.items():
        mask = test["Type"].astype(str) == subtype
        test.loc[mask, "pred_with_name"] = nextflu_predict(
            test.loc[mask],
            model["mutation_effects"],
            serum_potency=model["serum_potency"],
            virus_avidity=model["virus_avidity"],
        )

    anonymized_test = test.copy()
    anonymized_test["serumName"] = ""
    anonymized_test["virusName"] = ""
    for subtype, model in models.items():
        mask = anonymized_test["Type"].astype(str) == subtype
        test.loc[mask, "pred_without_name"] = nextflu_predict(
            anonymized_test.loc[mask],
            model["mutation_effects"],
            serum_potency=model["serum_potency"],
            virus_avidity=model["virus_avidity"],
        )

    test["reference"] = test["label"].astype(float)
    metrics = {
        "with_name": regression_metrics(test["reference"], test["pred_with_name"]),
        "without_name": regression_metrics(test["reference"], test["pred_without_name"]),
        "by_subtype": {},
    }
    for subtype, subtype_frame in test.groupby(test["Type"].astype(str)):
        metrics["by_subtype"][subtype] = {
            "with_name": regression_metrics(subtype_frame["reference"], subtype_frame["pred_with_name"]),
            "without_name": regression_metrics(subtype_frame["reference"], subtype_frame["pred_without_name"]),
        }

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": "nextflu_legacy_fixed_split",
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "output_dir": str(output_dir),
        "notebook_source": str(NEXTFLU_NOTEBOOK),
        "subtypes": subtypes,
        "merge_valid_into_train": bool(merge_valid_into_train),
        "test_season": test_season,
        "input_row_counts": {name: int(len(frame)) for name, frame in frames.items()},
        "training_row_count": int(len(train)),
    }
    payload = {"config": run_config, "metrics": metrics}

    test.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for subtype, model in models.items():
        (output_dir / f"{subtype}_model.json").write_text(
            json.dumps(_jsonable_model(model), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Nextflu baseline from fixed split CSVs.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--test-season",
        type=str,
        default="",
        help="Optional season subdirectory under --data-dir, e.g. --data-dir .../season --test-season 26.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subtypes", nargs="+", default=["H1N1", "H3N2"])
    parser.add_argument(
        "--merge-valid-into-train",
        action="store_true",
        default=False,
        help="Train on train.csv plus valid.csv, and evaluate only on test.csv.",
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        subtypes=args.subtypes,
        overwrite=args.overwrite,
        merge_valid_into_train=args.merge_valid_into_train,
        test_season=args.test_season,
    )


if __name__ == "__main__":
    main()
