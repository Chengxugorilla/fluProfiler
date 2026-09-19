"""Plot H3N2 serum-split label-versus-prediction scatters for three models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results/H1H3_HA1_v1.0/20260717_164256"
OUTPUT_DIR = PROJECT_ROOT / "paper/data/H3N2_serum_prediction_scatters_without_name"
MODELS = ("SCMS-FiLM-H3", "Nextflu", "Adaboost")
COLORS = {"SCMS-FiLM-H3": "#4C78A8", "Nextflu": "#B79A20", "Adaboost": "#59A14F"}


def load_predictions(model: str, seed: int) -> pd.DataFrame:
    if model == "Nextflu":
        path = RESULTS_ROOT / "Nextflu/serum" / f"seed_{seed}" / "H3N2/predictions.csv"
        frame = pd.read_csv(path)
        frame = frame.loc[
            (frame["Type"] == "H3N2") & frame["pred_without_name"].notna(),
            ["label", "pred_without_name"],
        ].rename(columns={"pred_without_name": "prediction"})
    elif model == "Adaboost":
        path = RESULTS_ROOT / "Adaboost/serum" / f"seed_{seed}" / "H3N2/predictions.csv"
        frame = pd.read_csv(path)
        frame = frame.loc[
            (frame["Type"] == "H3N2")
            & (frame["split"] == "test")
            & frame["prediction_without_name"].notna(),
            ["label", "prediction_without_name"],
        ].rename(columns={"prediction_without_name": "prediction"})
    elif model == "SCMS-FiLM-H3":
        path = (
            RESULTS_ROOT
            / "SCMS-FiLM-H3-Epoch50/serum"
            / f"seed_{seed}"
            / "subtype/H3N2_seed42/predictions_test.csv"
        )
        frame = pd.read_csv(path)
        frame = frame.loc[frame["Type"] == "H3N2", ["label", "mean"]].rename(
            columns={"mean": "prediction"}
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    return frame.dropna(subset=["label", "prediction"])


def plot_model(model: str) -> None:
    frame = pd.concat([load_predictions(model, seed) for seed in range(10)], ignore_index=True)
    labels = frame["label"].to_numpy(dtype=float)
    predictions = frame["prediction"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(labels, predictions, 1)
    pearson_r = float(np.corrcoef(labels, predictions)[0, 1])
    mae = float(np.abs(predictions - labels).mean())

    low = float(min(labels.min(), predictions.min()))
    high = float(max(labels.max(), predictions.max()))
    padding = (high - low) * 0.04
    limits = (low - padding, high + padding)
    line_x = np.linspace(*limits, 200)

    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=300)
    ax.scatter(labels, predictions, s=5, alpha=0.08, color=COLORS[model], linewidths=0, rasterized=True)
    ax.plot(line_x, line_x, "--", color="#777777", linewidth=1.2, label="identity")
    ax.plot(line_x, slope * line_x + intercept, color="black", linewidth=1.6, label="linear fit")
    ax.set(xlim=limits, ylim=limits, xlabel="Observed HI label", ylabel="Predicted HI label")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"H3N2 serum split: {model}")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.04,
        f"n = {len(frame):,}\nslope = {slope:.3f}\nPearson r = {pearson_r:.3f}\nMAE = {mae:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{model.lower().replace('-', '_')}_scatter.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"{model}: slope={slope:.4f}, Pearson r={pearson_r:.4f}, MAE={mae:.4f}")
    print(f"Saved: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        plot_model(model)


if __name__ == "__main__":
    main()
