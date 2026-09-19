#!/usr/bin/env python3
"""Export H3N2 held-out-serum benchmark tables, including identity conditions."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


RESULTS_ROOT = Path("results/H1H3_HA1_v1.0/20260717_164256")
SPLIT_ROOT = Path("data/dataset/H1H3_HA1_v1.0/splited/20260717_164256/serum")
OUTPUT_DIR = RESULTS_ROOT / "SCMS-FiLM-H3" / "heldout_serum_benchmark"

SPECS = (
    ("fluProfiler", "without_name", "SCMS-FiLM-H3", "predictions_test.csv", "label", "mean", True),
    ("AdaBoost", "without_name", "Adaboost", "predictions.csv", "reference", "prediction_without_name", False),
    ("Nextflu", "without_name", "Nextflu", "predictions.csv", "reference", "pred_without_name", False),
    ("AdaBoost", "with_name", "Adaboost", "predictions.csv", "reference", "prediction", False),
    ("Nextflu", "with_name", "Nextflu", "predictions.csv", "reference", "pred_with_name", False),
)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for index in range(start, end):
            ranks[order[index]] = rank
        start = end
    return ranks


def pearson(values_a: list[float], values_b: list[float]) -> float:
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    denominator = math.sqrt(
        sum((value - mean_a) ** 2 for value in values_a)
        * sum((value - mean_b) ** 2 for value in values_b)
    )
    if denominator == 0:
        return 0.0
    return sum(
        (value_a - mean_a) * (value_b - mean_b)
        for value_a, value_b in zip(values_a, values_b)
    ) / denominator


def read_predictions(path: Path, reference_column: str, prediction_column: str) -> list[tuple[str, float, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") not in (None, "", "test"):
                continue
            try:
                reference = float(row[reference_column])
                prediction = float(row[prediction_column])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(reference) and math.isfinite(prediction)):
                continue
            serum_id = row.get("task_key") or "||".join(
                (row.get("serumHA", ""), row.get("serumPassCat", ""), row.get("serumName", ""))
            )
            rows.append((serum_id, reference, prediction))
    return rows


def h3_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["serumType"] == "H3N2"]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    metadata = []
    prediction_rows = []

    for seed in range(10):
        split_id = str(seed)
        unit = f"seed_{seed}"
        split_dir = SPLIT_ROOT / unit
        test_rows = h3_rows(split_dir / "test.csv")
        train_rows = h3_rows(split_dir / "train.csv")
        valid_rows = h3_rows(split_dir / "valid.csv")
        heldout_groups = {
            (row["seq_a"], row["serumPassCat"], row["serumName"])
            for row in test_rows
        }
        metadata.append(
            {
                "split_id": split_id,
                "n_heldout_sera": len(heldout_groups),
                "n_train": len(train_rows) + len(valid_rows),
                "n_train_before_merge": len(train_rows),
                "n_valid": len(valid_rows),
                "n_test": len(test_rows),
            }
        )

        for model, identity_setting, directory, filename, reference_column, prediction_column, is_fluprofiler in SPECS:
            prediction_path = RESULTS_ROOT / directory / "serum" / unit
            subtype_dir = "subtype/H3N2_seed42" if is_fluprofiler else "H3N2"
            values = read_predictions(
                prediction_path / subtype_dir / filename,
                reference_column,
                prediction_column,
            )
            if len(values) != len(test_rows):
                raise ValueError(f"{prediction_path}: {len(values)} predictions, expected {len(test_rows)}")
            references = [value[1] for value in values]
            predictions = [value[2] for value in values]
            mae = sum(abs(prediction - reference) for reference, prediction in zip(references, predictions)) / len(values)
            summary.append(
                {
                    "split_id": split_id,
                    "model": model,
                    "identity_setting": identity_setting,
                    "n_test": len(values),
                    "MAE": f"{mae:.6f}",
                    "Pearson": f"{pearson(references, predictions):.6f}",
                    "Spearman": f"{pearson(average_ranks(references), average_ranks(predictions)):.6f}",
                }
            )
            prediction_rows.extend(
                {
                    "split_id": split_id,
                    "model": model,
                    "identity_setting": identity_setting,
                    "serum_id": serum_id,
                    "y_true": f"{reference:.10g}",
                    "y_pred": f"{prediction:.10g}",
                }
                for serum_id, reference, prediction in values
            )

    write_csv(
        OUTPUT_DIR / "benchmark_summary.csv",
        summary,
        ["split_id", "model", "identity_setting", "n_test", "MAE", "Pearson", "Spearman"],
    )
    write_csv(
        OUTPUT_DIR / "split_metadata.csv",
        metadata,
        ["split_id", "n_heldout_sera", "n_train", "n_train_before_merge", "n_valid", "n_test"],
    )
    write_csv(
        OUTPUT_DIR / "prediction_level_results.csv",
        prediction_rows,
        ["split_id", "model", "identity_setting", "serum_id", "y_true", "y_pred"],
    )
    print(f"Wrote {len(summary)} summary rows and {len(prediction_rows)} prediction rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
