#!/usr/bin/env python3
"""Build Figure 2 held-out-serum panels from the real split predictions.

The script compares identity-free predictions from fluProfiler, AdaBoost and
Nextflu over the ten matched H3N2 held-out-serum splits.  It writes an audited
long-form metric table plus independent MAE, Pearson and Spearman panels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


MODEL_ORDER = ("fluProfiler", "AdaBoost", "Nextflu")
MODEL_LABELS = {
    "fluProfiler": "fluProfiler",
    "AdaBoost": "AdaBoost",
    "Nextflu": "Nextflu",
}
COLORS = {
    "fluProfiler": "#0012E6",
    "AdaBoost": "#003642",
    "Nextflu": "#8D5103",
}
PAIR_KEY_COLUMNS = ("seq_id_a", "seq_id_b", "seq_id_c", "seq_id_d")
LABELLED_PAIR_KEY_COLUMNS = (*PAIR_KEY_COLUMNS, "label")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "results/H3_HA1_v1.0/20260827_112805",
        help="Root containing model serum/seed_* outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Fig2_heldout_serum_panels",
        help="Directory for the audited CSV and independent figure panels.",
    )
    return parser.parse_args()


def prediction_path(results_root: Path, model: str, seed: int) -> Path:
    if model == "fluProfiler":
        return results_root / (
            f"SCMS-FiLM-H3/serum/seed_{seed}/subtype/H3N2_seed42/"
            "predictions_test.csv"
        )
    result_directory = {"AdaBoost": "Adaboost", "Nextflu": "Nextflu"}[model]
    return results_root / f"{result_directory}/serum/seed_{seed}/H3N2/predictions.csv"


def read_test_predictions(path: Path, model: str) -> pd.DataFrame:
    if model == "fluProfiler":
        prediction_column = "mean"
    elif model == "AdaBoost":
        prediction_column = "prediction_without_name"
    elif model == "Nextflu":
        prediction_column = "pred_without_name"
    else:
        raise ValueError(f"Unsupported model: {model}")

    usecols = [*PAIR_KEY_COLUMNS, "label", prediction_column]
    frame = pd.read_csv(path, usecols=lambda name: name in set(usecols) | {"split"})
    if "split" in frame.columns:
        frame = frame.loc[frame["split"].eq("test")].copy()
        frame = frame.drop(columns="split")
    if frame.empty:
        raise ValueError(f"{path}: no test predictions found")
    if frame[prediction_column].isna().any():
        raise ValueError(f"{path}: identity-free prediction column contains missing values")
    if frame["label"].isna().any():
        raise ValueError(f"{path}: label column contains missing values")
    return frame.rename(columns={prediction_column: "prediction"})


def pair_signature(frame: pd.DataFrame, columns: tuple[str, ...] = PAIR_KEY_COLUMNS) -> np.ndarray:
    """Order-invariant hash of a pair multiset, retaining duplicate measurements."""
    hashes = pd.util.hash_pandas_object(frame.loc[:, columns], index=False)
    return np.sort(hashes.to_numpy())


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame["label"].astype(float)
    prediction = frame["prediction"].astype(float)
    return {
        "MAE": float((truth - prediction).abs().mean()),
        "Pearson": float(truth.corr(prediction, method="pearson")),
        "Spearman": float(truth.corr(prediction, method="spearman")),
        "n_test_pairs": int(len(frame)),
    }


def collect_metrics(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float | int | str]] = []
    audit_rows: list[dict[str, int | str]] = []
    for seed in range(10):
        frames: dict[str, pd.DataFrame] = {}
        for model in MODEL_ORDER:
            path = prediction_path(results_root, model, seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            frames[model] = read_test_predictions(path, model)

        reference_signature = pair_signature(frames["fluProfiler"])
        reference_labelled_signature = pair_signature(
            frames["fluProfiler"], LABELLED_PAIR_KEY_COLUMNS
        )
        for model, frame in frames.items():
            pair_multiset_matches = np.array_equal(pair_signature(frame), reference_signature)
            if not pair_multiset_matches:
                raise ValueError(f"seed {seed}: {model} test sequence pairs do not match fluProfiler")
            labelled_pair_multiset_matches = np.array_equal(
                pair_signature(frame, LABELLED_PAIR_KEY_COLUMNS),
                reference_labelled_signature,
            )
            metric_rows.append({"split": seed, "model": model, **compute_metrics(frame)})
            audit_rows.append({
                "split": seed,
                "model": model,
                "n_test_pairs": len(frame),
                "sequence_pair_multiset_matches_fluProfiler": pair_multiset_matches,
                "labelled_pair_multiset_matches_fluProfiler": labelled_pair_multiset_matches,
            })

    metrics = pd.DataFrame(metric_rows).sort_values(["split", "model"]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows).sort_values(["split", "model"]).reset_index(drop=True)
    expected = {(seed, model) for seed in range(10) for model in MODEL_ORDER}
    observed = set(zip(metrics["split"], metrics["model"]))
    if observed != expected:
        raise ValueError("Expected exactly one metric record for each seed and model")
    return metrics, audit


def draw_metric_panel(
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_stem: str,
    output_dir: Path,
) -> None:
    values_by_model = {
        model: data.loc[data["model"].eq(model), metric].dropna().astype(float).to_numpy()
        for model in MODEL_ORDER
    }
    if any(len(values) != 10 for values in values_by_model.values()):
        raise ValueError(f"{metric}: each model must have exactly 10 split values")

    data_min = min(values.min() for values in values_by_model.values())
    data_max = max(values.max() for values in values_by_model.values())
    if metric == "MAE":
        lower, upper = 0.0, max(2.0, data_max * 1.12)
        tick_step = 0.5
    else:
        lower, upper = min(0.0, data_min - 0.05), max(1.0, data_max + 0.05)
        tick_step = 0.2
    if data_min < lower or data_max > upper:
        raise ValueError(f"{metric}: requested axis limits would clip real split values")

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    positions = np.arange(len(MODEL_ORDER))
    rng = np.random.default_rng(20260907)
    y_range = upper - lower
    for position, model in zip(positions, MODEL_ORDER):
        values = values_by_model[model]
        color = COLORS[model]
        ax.boxplot(
            values,
            positions=[position],
            widths=0.68,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            boxprops={"facecolor": color, "edgecolor": color, "linewidth": 1.15, "alpha": 0.28},
            medianprops={"color": "#111111", "linewidth": 1.05},
            whiskerprops={"color": color, "linewidth": 1.25},
            capprops={"color": color, "linewidth": 1.25},
        )
        ax.scatter(
            np.full(len(values), position) + rng.normal(0, 0.045, len(values)),
            values,
            s=28,
            color=color,
            alpha=0.94,
            edgecolors="#F7F7F7",
            linewidths=0.9,
            zorder=3,
        )
        mean, sd = values.mean(), values.std(ddof=1)
        text_y = min(max(values) + 0.065 * y_range, upper - 0.055 * y_range)
        ax.text(
            position,
            text_y,
            f"{mean:.3f} \N{PLUS-MINUS SIGN} {sd:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold" if model == "fluProfiler" else "normal",
            color=color,
        )

    ticks = np.arange(lower, upper + tick_step * 0.5, tick_step)
    ax.set_xlim(-0.55, len(MODEL_ORDER) - 0.45)
    ax.set_ylim(lower, upper)
    ax.set_yticks(ticks)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], fontsize=9)
    for tick, model in zip(ax.get_xticklabels(), MODEL_ORDER):
        tick.set_color(COLORS[model])
        if model == "fluProfiler":
            tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("#111111")
    ax.tick_params(axis="both", which="major", width=1.1, length=5, color="#111111")
    ax.grid(False)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.93)
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(output_dir / f"{output_stem}.{suffix}", bbox_inches="tight", transparent=True, **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next(
        (name for name in ("Arial", "Liberation Sans", "DejaVu Sans") if name in available_fonts),
        "sans-serif",
    )
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.transparent": True,
    })
    metrics, audit = collect_metrics(args.results_root)
    metrics.to_csv(args.output_dir / "heldout_serum_results.csv", index=False)
    audit.to_csv(args.output_dir / "heldout_serum_input_audit.csv", index=False)
    draw_metric_panel(metrics, "MAE", "MAE", "Fig2A_HeldoutSerum_MAE", args.output_dir)
    draw_metric_panel(metrics, "Pearson", "Pearson correlation", "Fig2B_HeldoutSerum_Pearson", args.output_dir)
    draw_metric_panel(metrics, "Spearman", "Spearman correlation", "Fig2C_HeldoutSerum_Spearman", args.output_dir)
    print(f"Wrote audited metrics and panels to {args.output_dir}")


if __name__ == "__main__":
    main()
