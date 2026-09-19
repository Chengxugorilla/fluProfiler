#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch


JOB_DIR = Path("/home/chenyh/workspace/fluProfiler/temp job")
DEFAULT_INPUT = JOB_DIR / "recurrent_vaccine_predictions.matrix.sorted.csv"
DEFAULT_OUTPUT = JOB_DIR / "recurrent_vaccine_predictions.top20_heatmap.png"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
BOLD_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


COLORS = [
    "#b9f5b6",  # <0
    "#5ebd63",  # 0-0.5
    "#48a9e6",  # 0.5-1
    "#ffd23f",  # 1-1.5
    "#ff8a00",  # 1.5-2
    "#f05252",  # 2-2.5
    "#a779e9",  # >2.5
]
LEGEND_LABELS = ["<0", "0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5", ">2.5"]


def build_heatmap(input_csv: Path, output_png: Path, top_n: int = 20) -> Path:
    df = pd.read_csv(input_csv).head(top_n)
    id_col = df.columns[0]
    value_df = df.drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce")
    values = value_df.to_numpy(dtype=float)

    lower = min(-0.01, float(np.nanmin(values)) - 0.01)
    upper = max(2.51, float(np.nanmax(values)) + 0.01)
    bounds = [lower, 0, 0.5, 1, 1.5, 2, 2.5, upper]
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(bounds, cmap.N)

    font = FontProperties(fname=str(FONT_PATH))
    bold_font = FontProperties(fname=str(BOLD_FONT_PATH))
    fig = plt.figure(figsize=(16, 9), dpi=180, facecolor="white")

    fig.text(
        0.05,
        0.94,
        "fluProfiler预测：H1N1天然株（Egg）与历年疫苗株之间的抗原距离",
        fontproperties=bold_font,
        fontsize=26,
        ha="left",
    )
    fig.add_artist(plt.Line2D([0.05, 0.93], [0.915, 0.915], transform=fig.transFigure, color="black", linewidth=2))

    fig.text(
        0.05,
        0.88,
        f"H1N1天然株（血清株）：\n• GISAID recurrent HA氨基酸序列\n• passage：EGG\n• 显示排序后 TOP{top_n} / 共979条",
        fontproperties=font,
        fontsize=15,
        va="top",
    )
    fig.text(
        0.55,
        0.88,
        f"历年疫苗株（病毒株）：\n• vaccine HA氨基酸序列\n• passage：EGG\n• 共{value_df.shape[1]}列",
        fontproperties=font,
        fontsize=15,
        va="top",
    )

    ax = fig.add_axes([0.08, 0.16, 0.76, 0.56])
    ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(f"天然株（Egg）TOP{top_n}与疫苗株的抗原距离预测值", fontproperties=font, fontsize=14, pad=10)
    ax.set_xticks(np.arange(value_df.shape[1]))
    ax.set_xticklabels(value_df.columns, rotation=45, ha="right", rotation_mode="anchor", fontproperties=font, fontsize=9)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df[id_col], fontproperties=bold_font, fontsize=8)
    ax.tick_params(length=0)

    ax.set_xticks(np.arange(-0.5, value_df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.2, color="#202020")

    fig.text(0.025, 0.43, "天然株", fontproperties=font, fontsize=16, rotation=90, va="center", ha="center")
    fig.text(0.025, 0.67, "↑", fontproperties=font, fontsize=18, va="center", ha="center")

    legend_ax = fig.add_axes([0.86, 0.35, 0.10, 0.24])
    legend_ax.axis("off")
    legend_ax.text(0, 1.05, "抗原距离\n范围", fontproperties=font, fontsize=10, va="bottom")
    handles = [Patch(facecolor=color, edgecolor="white", label=label) for color, label in zip(COLORS, LEGEND_LABELS)]
    legend_ax.legend(handles=handles, loc="upper left", frameon=False, prop=font, handlelength=1.2, handleheight=1.2)

    fig.text(
        0.86,
        0.25,
        "排序方式：\n1.  抗原距离>2\n    的个数：从少到多\n2.  抗原距离最\n    大值：从小到大",
        fontproperties=bold_font,
        fontsize=14,
        va="top",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot top-N recurrent-vaccine antigenic distance heatmap.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = build_heatmap(args.input_csv, args.output_png, args.top_n)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
