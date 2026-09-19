"""Plot H3N2 seasonal ranks with name-aware versions of all three models."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import plot_h3n2_seasonal_performance_without_name_baselines as base


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results/H1H3_HA1_v1.0/20260717_164256"
OUTPUT_PNG = PROJECT_ROOT / "paper/data/Fig_H3N2_seasonal_performance_with_name.png"
OUTPUT_CSV = PROJECT_ROOT / "paper/data/Fig_H3N2_seasonal_performance_with_name.csv"
SEASONS = ("39", "40", "41", "42", "43", "44")
MODELS = ("SCMS-FiLM-H3+SVBias", "Adaboost", "Nextflu")


def load_nextflu(season: str) -> dict[str, float]:
    values = json.loads((RESULTS_ROOT / "Nextflu/season" / season / "H3N2/metrics.json").read_text())["metrics"]["with_name"]
    return {"MAE": values["mae"], "MSE": values["mse"], "Pearson": values["pearson"], "Spearman": values["spearman"], "R2": values["r2"]}


def load_adaboost(season: str) -> dict[str, float]:
    result_dir = RESULTS_ROOT / "Adaboost/season" / season / "H3N2"
    values = json.loads((result_dir / "metrics.json").read_text())["metrics"]["test"]["by_subtype"]["H3N2"]["with_name"]
    frame = pd.read_csv(result_dir / "predictions.csv")
    frame = frame.loc[(frame["Type"] == "H3N2") & (frame["split"] == "test")]
    return {"MAE": values["mae"], "MSE": values["mse"], "Pearson": values["pearson"], "Spearman": float(spearmanr(frame["label"], frame["prediction"]).statistic), "R2": values["r2"]}


def load_scms_with_name(season: str) -> dict[str, float]:
    result_dir = RESULTS_ROOT / "SCMS-FiLM-H3+SVBias/season" / season / "subtype/H3N2_seed42"
    with (result_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        final = list(csv.DictReader(handle))[-1]
    frame = pd.read_csv(result_dir / "predictions_test.csv")
    labels = frame["label"].to_numpy(dtype=float)
    predictions = frame["mean"].to_numpy(dtype=float)
    r2 = 1.0 - np.square(labels - predictions).sum() / np.square(labels - labels.mean()).sum()
    return {"MAE": float(final["test_pooled_mae"]), "MSE": float(final["test_pooled_mse"]), "Pearson": float(final["test_pooled_pearson"]), "Spearman": float(final["test_pooled_spearman"]), "R2": float(r2)}


def main() -> None:
    loaders = {"SCMS-FiLM-H3+SVBias": load_scms_with_name, "Adaboost": load_adaboost, "Nextflu": load_nextflu}
    rows = []
    for season in SEASONS:
        for model in MODELS:
            for metric, value in loaders[model](season).items():
                rows.append({"season": season, "model": model, "metric": metric, "value": value})
    values = pd.DataFrame(rows)
    values["rank"] = np.nan
    for (_, metric), index in values.groupby(["season", "metric"]).groups.items():
        values.loc[index, "rank"] = values.loc[index, "value"].rank(method="min", ascending=metric not in base.HIGHER_BETTER)
    values["rank"] = values["rank"].astype(int)
    values.to_csv(OUTPUT_CSV, index=False)

    base.MODELS = MODELS
    base.DISPLAY_NAMES = {"SCMS-FiLM-H3+SVBias": "SCMS-FiLM-H3+SVBias", "Adaboost": "AdaBoost", "Nextflu": "Nextflu"}
    base.MODEL_COLORS = {"SCMS-FiLM-H3+SVBias": "#aec2d3", "Adaboost": "#bcc8b8", "Nextflu": "#dcd6a5"}
    base.OUTPUT_PNG = OUTPUT_PNG
    base.plot_rank_matrix(values)
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
