"""Create Figure 2D: seasonal H3N2 ranking from identity-free predictions.

The five displayed metrics are MAE, MSE, Pearson, Spearman, and R-squared.
Within every season and metric, lower MAE and higher correlations receive the
better rank.  All values are recalculated from the saved test predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results/H3_HA1_v1.0/20260827_112805"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "Fig2_seasonal_rank"

SEASONS = {
    "39": "2023NH",
    "40": "2023SH",
    "41": "2024NH",
    "42": "2024SH",
    "43": "2025NH",
    "44": "2025SH",
}
MODELS = ("fluProfiler", "AdaBoost", "Nextflu")
DISPLAY_NAMES = {
    "fluProfiler": "fluProfiler",
    "AdaBoost": "AdaBoost",
    "Nextflu": "Nextflu",
}
MODEL_COLORS = {
    "fluProfiler": "#0012E6",
    "AdaBoost": "#003642",
    "Nextflu": "#8D5103",
}
METRICS = ("MAE", "MSE", "Pearson", "Spearman", "R2")
METRIC_LABELS = {"MAE": "L1", "MSE": "L2", "Pearson": "r", "Spearman": "ρ", "R2": "R²"}
HIGHER_BETTER = {"Pearson", "Spearman", "R2"}
RANK_FILLS = {
    1: "#70A6D8",
    2: "#B7D2EF",
    3: "#EAF2FA",
}
FONT_CANDIDATES = ("Arial", "Liberation Sans", "DejaVu Sans")


def configure_fonts() -> None:
    """Use the same clean sans-serif styling as Figure 2 panels A--C."""
    available = {font.name for font in plt.matplotlib.font_manager.fontManager.ttflist}
    font = next(candidate for candidate in FONT_CANDIDATES if candidate in available)
    plt.rcParams.update({"font.family": font, "svg.fonttype": "none", "pdf.fonttype": 42})


def prediction_file(results_root: Path, model: str, season: str) -> tuple[Path, str, str | None]:
    if model == "fluProfiler":
        return (
            results_root / f"SCMS-FiLM-H3/season/{season}/subtype/H3N2_seed42/predictions_test.csv",
            "mean",
            None,
        )
    if model == "AdaBoost":
        return (
            results_root / f"Adaboost/season/{season}/H3N2/predictions.csv",
            "prediction_without_name",
            "split == 'test'",
        )
    return (
        results_root / f"Nextflu/season/{season}/H3N2/predictions.csv",
        "pred_without_name",
        None,
    )


def compute_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    # fluProfiler serializes labels to four decimal places in predictions_test.csv.
    # Apply that same precision to every model for a common labelled test panel.
    y_true = frame["label"].round(4).to_numpy(dtype=float)
    y_pred = frame[prediction_column].to_numpy(dtype=float)
    residual = y_true - y_pred
    total = y_true - y_true.mean()
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "MSE": float(np.mean(np.square(residual))),
        "Pearson": float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
        "R2": float(1.0 - np.square(residual).sum() / np.square(total).sum()),
    }


def collect_seasonal_metrics(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season, season_label in SEASONS.items():
        for model in MODELS:
            path, prediction_column, query = prediction_file(results_root, model, season)
            if not path.exists():
                raise FileNotFoundError(f"Missing {model} predictions for {season_label}: {path}")
            frame = pd.read_csv(path)
            if query:
                frame = frame.query(query).copy()
            required = {"label", prediction_column}
            if missing := required.difference(frame.columns):
                raise ValueError(f"{path} lacks columns: {sorted(missing)}")
            values = compute_metrics(frame, prediction_column)
            for metric, value in values.items():
                rows.append(
                    {
                        "season_id": season,
                        "season": season_label,
                        "model": model,
                        "metric": metric,
                        "value": value,
                        "n_test_pairs": len(frame),
                    }
                )

    values = pd.DataFrame(rows)
    # Preserve original row indices while ranking: groupby.apply can reorder groups.
    values["rank"] = np.nan
    for metric in METRICS:
        metric_mask = values["metric"].eq(metric)
        values.loc[metric_mask, "rank"] = values.loc[metric_mask].groupby("season")["value"].rank(
            method="min", ascending=metric not in HIGHER_BETTER
        )
    values["rank"] = values["rank"].astype(int)
    return values


def lighten(color: str, amount: float) -> tuple[float, float, float, float]:
    base = np.asarray(to_rgba(color))
    return tuple(base * (1.0 - amount) + np.asarray(to_rgba("#FFFFFF")) * amount)


def draw_rank_matrix(values: pd.DataFrame, output_dir: Path) -> None:
    configure_fonts()
    fig, ax = plt.subplots(figsize=(11.45, 2.55), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cell_w, cell_h = 0.53, 0.72
    label_w, header_h = 1.85, 0.82
    left, bottom = 0.15, 0.20
    season_gap, summary_gap = 0.025, 0.08
    metrics_width = len(METRICS) * cell_w
    table_width = len(SEASONS) * metrics_width + (len(SEASONS) - 1) * season_gap
    table_height = len(MODELS) * cell_h
    table_x = left + label_w
    header_y = bottom + table_height

    # Model labels: the user-specified model hues, softened to match panels A--C.
    for row_index, model in enumerate(MODELS):
        y = bottom + (len(MODELS) - 1 - row_index) * cell_h
        ax.add_patch(
            FancyBboxPatch(
                (left, y), label_w, cell_h,
                boxstyle="round,pad=0.01,rounding_size=0.045",
                facecolor=lighten(MODEL_COLORS[model], 0.91),
                edgecolor="#8AA2B8", linewidth=0.8,
            )
        )
        ax.text(
            left + label_w / 2, y + cell_h / 2, DISPLAY_NAMES[model],
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color=MODEL_COLORS[model], linespacing=0.9,
        )

    # Headers and rank cells.  Blue intensity encodes rank universally.
    for season_index, (season_id, season_label) in enumerate(SEASONS.items()):
        season_x = table_x + season_index * (metrics_width + season_gap)
        ax.add_patch(
            FancyBboxPatch(
                (season_x, header_y), metrics_width, header_h,
                boxstyle="round,pad=0.008,rounding_size=0.045",
                facecolor="#E6F0FA", edgecolor="#7C9DBA", linewidth=0.75,
            )
        )
        ax.text(
            season_x + metrics_width / 2, header_y + header_h * 0.69, season_label,
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="#071A64",
        )
        for metric_index, metric in enumerate(METRICS):
            x = season_x + metric_index * cell_w
            ax.text(
                x + cell_w / 2, header_y + header_h * 0.25, METRIC_LABELS[metric],
                ha="center", va="center", fontsize=10.5, fontweight="bold", color="#071A64",
            )
            if metric_index:
                ax.plot([x, x], [header_y, header_y + header_h * 0.48], color="#AFC3D7", lw=0.6)
            for row_index, model in enumerate(MODELS):
                y = bottom + (len(MODELS) - 1 - row_index) * cell_h
                rank = int(values.loc[
                    (values["season_id"] == season_id)
                    & (values["model"] == model)
                    & (values["metric"] == metric), "rank"
                ].iloc[0])
                ax.add_patch(Rectangle(
                    (x, y), cell_w, cell_h, facecolor=RANK_FILLS[rank],
                    edgecolor="#D4E2EF", linewidth=0.75,
                ))
                ax.text(
                    x + cell_w / 2, y + cell_h / 2, str(rank), ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#071A64",
                )

    # Summary columns retain the sparse, high-contrast style of the reference.
    summary_x = table_x + table_width + summary_gap
    total_rankings = len(SEASONS) * len(METRICS)
    summary_specs = (("Mean\nrank", 0.98), (f"Top-2/\n{total_rankings}", 1.00))
    for summary_index, (title, width) in enumerate(summary_specs):
        x = summary_x + sum(item[1] for item in summary_specs[:summary_index])
        ax.add_patch(FancyBboxPatch(
            (x, header_y), width, header_h,
            boxstyle="round,pad=0.008,rounding_size=0.045",
            facecolor="#E6F0FA", edgecolor="#7C9DBA", linewidth=0.75,
        ))
        ax.text(x + width / 2, header_y + header_h / 2, title, ha="center", va="center",
                fontsize=10.8, fontweight="bold", color="#071A64", linespacing=0.9)
        for row_index, model in enumerate(MODELS):
            y = bottom + (len(MODELS) - 1 - row_index) * cell_h
            ax.add_patch(Rectangle((x, y), width, cell_h, facecolor="#FFFFFF",
                                   edgecolor="#8AA2B8", linewidth=0.75))
            ranks = values.loc[values["model"] == model, "rank"]
            text = f"{ranks.mean():.1f}" if summary_index == 0 else f"{(ranks <= 2).sum()}/{total_rankings}"
            ax.text(x + width / 2, y + cell_h / 2, text, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#071A64")

    max_x = summary_x + sum(width for _, width in summary_specs)
    ax.set_xlim(left - 0.03, max_x + 0.05)
    ax.set_ylim(bottom - 0.03, header_y + header_h + 0.03)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Fig2D_SeasonalRank"
    for extension in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    values = collect_seasonal_metrics(args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values.to_csv(args.output_dir / "Fig2D_seasonal_metric_ranks.csv", index=False)
    draw_rank_matrix(values, args.output_dir)
    print(f"Wrote Figure 2D ranks and panels to {args.output_dir}")


if __name__ == "__main__":
    main()
