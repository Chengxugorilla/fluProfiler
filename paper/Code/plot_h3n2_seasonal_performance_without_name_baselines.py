"""Plot H3N2 seasonal performance ranks for SCMS-FiLM-H3 and no-name baselines."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results/H1H3_HA1_v1.0/20260717_164256"
OUTPUT_PNG = PROJECT_ROOT / "paper/data/Fig_H3N2_seasonal_performance_without_name_baselines.png"
OUTPUT_CSV = PROJECT_ROOT / "paper/data/Fig_H3N2_seasonal_performance_without_name_baselines.csv"
SEASONS = ("39", "40", "41", "42", "43", "44")
MODELS = ("SCMS-FiLM-H3", "Adaboost", "Nextflu")
METRICS = ("MAE", "MSE", "Pearson", "Spearman", "R2")
HIGHER_BETTER = {"Pearson", "Spearman", "R2"}
DISPLAY_NAMES = {"SCMS-FiLM-H3": "SCMS-FiLM-H3", "Adaboost": "AdaBoost", "Nextflu": "Nextflu"}
MODEL_COLORS = {"SCMS-FiLM-H3": "#aec2d3", "Adaboost": "#bcc8b8", "Nextflu": "#dcd6a5"}


def load_nextflu(season: str) -> dict[str, float]:
    path = RESULTS_ROOT / "Nextflu/season" / season / "H3N2/metrics.json"
    values = json.loads(path.read_text(encoding="utf-8"))["metrics"]["without_name"]
    return {"MAE": values["mae"], "MSE": values["mse"], "Pearson": values["pearson"], "Spearman": values["spearman"], "R2": values["r2"]}


def load_adaboost(season: str) -> dict[str, float]:
    result_dir = RESULTS_ROOT / "Adaboost/season" / season / "H3N2"
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    values = metrics["metrics"]["test"]["by_subtype"]["H3N2"]["without_name"]
    frame = pd.read_csv(result_dir / "predictions.csv")
    frame = frame.loc[(frame["Type"] == "H3N2") & (frame["split"] == "test")]
    return {"MAE": values["mae"], "MSE": values["mse"], "Pearson": values["pearson"], "Spearman": float(spearmanr(frame["label"], frame["prediction_without_name"]).statistic), "R2": values["r2"]}


def load_scms(season: str) -> dict[str, float]:
    result_dir = RESULTS_ROOT / "SCMS-FiLM-H3/season" / season / "subtype/H3N2_seed42"
    with (result_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        final = list(csv.DictReader(handle))[-1]
    frame = pd.read_csv(result_dir / "predictions_test.csv")
    labels = frame["label"].to_numpy(dtype=float)
    predictions = frame["mean"].to_numpy(dtype=float)
    r2 = 1.0 - np.square(labels - predictions).sum() / np.square(labels - labels.mean()).sum()
    return {"MAE": float(final["test_pooled_mae"]), "MSE": float(final["test_pooled_mse"]), "Pearson": float(final["test_pooled_pearson"]), "Spearman": float(final["test_pooled_spearman"]), "R2": float(r2)}


def load_values() -> pd.DataFrame:
    loaders = {"SCMS-FiLM-H3": load_scms, "Adaboost": load_adaboost, "Nextflu": load_nextflu}
    rows = []
    for season in SEASONS:
        for model in MODELS:
            for metric, value in loaders[model](season).items():
                rows.append({"season": season, "model": model, "metric": metric, "value": value})
    values = pd.DataFrame(rows)
    values["rank"] = values.groupby(["season", "metric"], group_keys=False).apply(
        lambda group: group["value"].rank(method="min", ascending=group.name[1] not in HIGHER_BETTER).astype(int),
        include_groups=False,
    ).to_numpy()
    return values


def plot_rank_matrix(values: pd.DataFrame) -> None:
    cell_w, cell_h, left, bottom = 0.42, 0.78, 2.45, 0.75
    fig, ax = plt.subplots(figsize=(15, 3.6), dpi=300)
    rank_colors = {1: "#ffffff", 2: "#eeeeee", 3: "#dddddd"}
    for row_index, model in enumerate(MODELS):
        y = bottom + (len(MODELS) - 1 - row_index) * cell_h
        ax.add_patch(Rectangle((left - 2.1, y), 2.0, cell_h, facecolor="#f6f6f6", edgecolor="#333333", linewidth=0.8))
        ax.add_patch(Rectangle((left - 1.92, y + 0.17), 0.22, 0.44, facecolor=MODEL_COLORS[model], edgecolor="none"))
        ax.text(left - 1.58, y + cell_h / 2, DISPLAY_NAMES[model], ha="left", va="center", fontsize=10, fontweight="bold" if model == "SCMS-FiLM-H3" else None)
        for season_index, season in enumerate(SEASONS):
            for metric_index, metric in enumerate(METRICS):
                x = left + (season_index * len(METRICS) + metric_index) * cell_w
                rank = int(values.query("season == @season and metric == @metric and model == @model")["rank"].iloc[0])
                ax.add_patch(Rectangle((x, y), cell_w, cell_h, facecolor=rank_colors[rank], edgecolor="#333333", linewidth=0.6))
                ax.text(x + cell_w / 2, y + cell_h / 2, str(rank), ha="center", va="center", fontsize=11)
    for season_index, season in enumerate(SEASONS):
        start_x = left + season_index * len(METRICS) * cell_w
        ax.text(start_x + len(METRICS) * cell_w / 2, bottom + len(MODELS) * cell_h + 0.48, f"Season {season}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        for metric_index, metric in enumerate(METRICS):
            ax.text(start_x + (metric_index + 0.5) * cell_w, bottom + len(MODELS) * cell_h + 0.07, metric, ha="center", va="bottom", fontsize=7, rotation=45)
    summary_x = left + len(SEASONS) * len(METRICS) * cell_w + 0.35
    ax.text(summary_x + 0.45, bottom + len(MODELS) * cell_h + 0.18, "Mean rank", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for row_index, model in enumerate(MODELS):
        y = bottom + (len(MODELS) - 1 - row_index) * cell_h
        mean_rank = values.loc[values["model"] == model, "rank"].mean()
        ax.add_patch(Rectangle((summary_x, y), 0.9, cell_h, facecolor="#f6f6f6", edgecolor="#333333", linewidth=0.8))
        ax.text(summary_x + 0.45, y + cell_h / 2, f"{mean_rank:.2f}", ha="center", va="center", fontsize=11)
    ax.set_title("H3N2 seasonal extrapolation ranking (baselines without name)", fontsize=13, pad=16)
    ax.set_xlim(left - 2.2, summary_x + 1.05)
    ax.set_ylim(bottom - 0.2, bottom + len(MODELS) * cell_h + 0.9)
    ax.axis("off")
    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    values = load_values()
    values.to_csv(OUTPUT_CSV, index=False)
    plot_rank_matrix(values)
    print(values.pivot(index=["season", "metric"], columns="model", values=["value", "rank"]).round(4))
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
