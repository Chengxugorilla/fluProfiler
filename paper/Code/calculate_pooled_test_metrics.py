#!/usr/bin/env python3
"""Calculate H1N1+H3N2 pooled (micro) test metrics for benchmark results.

For every split, this script concatenates all available H1N1 and H3N2 test
predictions across runs (seed_0..seed_9 for titer/strain/serum and seasons
39..44 for season) before calculating regression metrics.  This is therefore
not an unweighted average of the two subtype metrics.

AdaBoost and Nextflu are evaluated with their without-name predictions;
SerumGate is evaluated with its regular ``mean`` prediction.

Example:
    python paper/Code/calculate_pooled_test_metrics.py \
      --results-root results/H1H3_HA1_v1.0/20260717_164256 \
      --output-csv paper/Code/pooled_test_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPLITS = ("titer", "strain", "serum", "season")
SUBTYPES = ("H1N1", "H3N2")


@dataclass(frozen=True)
class ModelSpec:
    directory: str
    prediction_filename: str
    reference_column: str
    prediction_column: str
    serumgate_layout: bool = False


MODEL_SPECS = {
    "Adaboost": ModelSpec(
        directory="Adaboost",
        prediction_filename="predictions.csv",
        reference_column="reference",
        prediction_column="prediction_without_name",
    ),
    "Nextflu": ModelSpec(
        directory="Nextflu",
        prediction_filename="predictions.csv",
        reference_column="reference",
        prediction_column="pred_without_name",
    ),
    "SerumGate": ModelSpec(
        directory="SerumGate-Minus-latent8",
        prediction_filename="predictions_test.csv",
        reference_column="label",
        prediction_column="mean",
        serumgate_layout=True,
    ),
}


def units_for_split(split: str) -> tuple[str, ...]:
    if split == "season":
        return tuple(str(season) for season in range(39, 45))
    return tuple(f"seed_{seed}" for seed in range(10))


def prediction_path(
    results_root: Path,
    spec: ModelSpec,
    split: str,
    unit: str,
    subtype: str,
) -> Path:
    base = results_root / spec.directory / split / unit
    if spec.serumgate_layout:
        base = base / "subtype"
    return base / subtype / spec.prediction_filename


def read_test_predictions(path: Path, spec: ModelSpec) -> tuple[list[float], list[float]]:
    """Read finite reference/prediction pairs from one prediction CSV.

    AdaBoost stores train and test predictions in one file.  When a ``split``
    column is present, only explicit test rows are used.  Other prediction
    files contain only test rows and are read in full.
    """

    references: list[float] = []
    predictions: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Prediction file is empty: {path}") from error
        try:
            reference_index = header.index(spec.reference_column)
            prediction_index = header.index(spec.prediction_column)
        except ValueError as error:
            raise ValueError(
                f"{path} must contain {spec.reference_column!r} and "
                f"{spec.prediction_column!r} columns"
            ) from error
        split_index = header.index("split") if "split" in header else None

        for row in reader:
            if split_index is not None and row[split_index] != "test":
                continue
            try:
                reference = float(row[reference_index])
                prediction = float(row[prediction_index])
            except (IndexError, ValueError):
                continue
            if math.isfinite(reference) and math.isfinite(prediction):
                references.append(reference)
                predictions.append(prediction)
    return references, predictions


def pearson(values_a: list[float], values_b: list[float]) -> float:
    count = len(values_a)
    mean_a = sum(values_a) / count
    mean_b = sum(values_b) / count
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    denominator = math.sqrt(
        sum(value * value for value in centered_a)
        * sum(value * value for value in centered_b)
    )
    return 0.0 if denominator == 0.0 else sum(
        value_a * value_b for value_a, value_b in zip(centered_a, centered_b)
    ) / denominator


def average_ranks(values: list[float]) -> list[float]:
    """Return 1-based average ranks, including the usual handling of ties."""

    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def calculate_metrics(references: list[float], predictions: list[float]) -> dict[str, float | int]:
    if not references:
        raise ValueError("No finite test prediction pairs were found")
    errors = [prediction - reference for reference, prediction in zip(references, predictions)]
    mean_reference = sum(references) / len(references)
    total_sum_squares = sum((reference - mean_reference) ** 2 for reference in references)
    residual_sum_squares = sum(error * error for error in errors)
    return {
        "rows": len(references),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mse": residual_sum_squares / len(errors),
        "pearson": pearson(references, predictions),
        "spearman": pearson(average_ranks(references), average_ranks(predictions)),
        "r2": 0.0 if total_sum_squares == 0.0 else 1.0 - residual_sum_squares / total_sum_squares,
    }


def pooled_metrics(results_root: Path, model_name: str, split: str) -> dict[str, float | int]:
    spec = MODEL_SPECS[model_name]
    all_references: list[float] = []
    all_predictions: list[float] = []
    missing_paths: list[Path] = []

    for unit in units_for_split(split):
        for subtype in SUBTYPES:
            path = prediction_path(results_root, spec, split, unit, subtype)
            if not path.is_file():
                missing_paths.append(path)
                continue
            references, predictions = read_test_predictions(path, spec)
            all_references.extend(references)
            all_predictions.extend(predictions)

    if missing_paths:
        missing_text = "\n  ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"Missing prediction files for {model_name}/{split}:\n  {missing_text}"
        )
    return calculate_metrics(all_references, all_predictions)


def write_csv(rows: Iterable[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("split", "model", "rows", "mae", "mse", "pearson", "spearman", "r2")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Calculate pooled H1N1+H3N2 micro test metrics."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project_root / "results/H1H3_HA1_v1.0/20260717_164256",
        help="Experiment result root containing Adaboost, Nextflu and SerumGate directories.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional CSV path for the pooled summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    rows: list[dict[str, float | int | str]] = []
    for split in SPLITS:
        for model_name in MODEL_SPECS:
            metrics = pooled_metrics(results_root, model_name, split)
            rows.append({"split": split, "model": model_name, **metrics})

    print("Pooled H1N1+H3N2 test metrics (all units and samples concatenated)")
    print("split    model       rows      MAE      MSE  Pearson Spearman       R2")
    for row in rows:
        print(
            f"{row['split']:<8} {row['model']:<10} {row['rows']:>7} "
            f"{row['mae']:>8.4f} {row['mse']:>8.4f} {row['pearson']:>8.4f} "
            f"{row['spearman']:>8.4f} {row['r2']:>8.4f}"
        )

    if args.output_csv is not None:
        output_path = args.output_csv.expanduser().resolve()
        write_csv(rows, output_path)
        print(f"\nWrote CSV: {output_path}")


if __name__ == "__main__":
    main()
