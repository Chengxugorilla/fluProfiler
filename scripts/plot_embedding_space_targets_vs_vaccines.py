#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


JOB_DIR = Path("/home/chenyh/workspace/fluProfiler/temp job")
EMBEDDING_DIR = JOB_DIR / "embedding"
TARGET_FASTA = JOB_DIR / "strain76vsvaccine26.fasta"
VACCINE_FASTA = JOB_DIR / "H1N1_vaccine_origin.fasta"
OUT_PNG = JOB_DIR / "strain76_vaccine26_vs_vaccines_embedding_pca.png"
OUT_CSV = JOB_DIR / "strain76_vaccine26_vs_vaccines_embedding_distances.csv"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
BOLD_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def accession_candidates(record_id: str) -> list[str]:
    parts = [part.strip() for part in record_id.split("|")]
    return [part for part in reversed(parts) if part.startswith("EPI")]


def embedding_id(record_id: str) -> str:
    candidates = accession_candidates(record_id)
    for candidate in candidates:
        if (EMBEDDING_DIR / f"matrix_{candidate}.pt").is_file():
            return candidate
    return candidates[0] if candidates else record_id.split("|")[0].strip()


def load_mean_vector(seq_id: str) -> np.ndarray:
    path = EMBEDDING_DIR / f"matrix_{seq_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    matrix = torch.load(path, map_location="cpu", weights_only=False)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix for {seq_id}, got shape {matrix.shape}")
    if matrix.shape[0] > 2:
        matrix = matrix[1:-1]
    return matrix.mean(axis=0)


def target_rows() -> list[dict[str, str]]:
    rows = []
    labels = ["1976 New Jersey", "2025 Missouri / 2026 vaccine"]
    for label, record in zip(labels, SeqIO.parse(TARGET_FASTA, "fasta")):
        rows.append(
            {
                "seq_id": embedding_id(record.description),
                "plot_label": label,
                "group": "target",
                "record_id": record.description,
            }
        )
    return rows


def vaccine_rows() -> list[dict[str, str]]:
    rows = []
    for record in SeqIO.parse(VACCINE_FASTA, "fasta"):
        seq_id = embedding_id(record.description)
        rows.append(
            {
                "seq_id": seq_id,
                "plot_label": seq_id.replace("EPI_ISL_", "ISL_"),
                "group": "vaccine",
                "record_id": record.description,
            }
        )
    return rows


def main() -> int:
    rows = target_rows() + vaccine_rows()
    seen = set()
    unique_rows = []
    for row in rows:
        key = row["seq_id"]
        if key not in seen:
            unique_rows.append(row)
            seen.add(key)

    vectors = np.vstack([load_mean_vector(row["seq_id"]) for row in unique_rows])
    coords = PCA(n_components=2, random_state=42).fit_transform(vectors)
    pca = PCA(n_components=2, random_state=42).fit(vectors)
    explained = pca.explained_variance_ratio_ * 100
    coords = pca.transform(vectors)

    df = pd.DataFrame(unique_rows)
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]

    target_ids = [row["seq_id"] for row in target_rows()]
    target_vectors = {seq_id: vectors[df.index[df["seq_id"] == seq_id][0]] for seq_id in target_ids}
    records = []
    for _, row in df.iterrows():
        if row["seq_id"] in target_ids:
            continue
        idx = df.index[df["seq_id"] == row["seq_id"]][0]
        vec = vectors[idx : idx + 1]
        rec = {"seq_id": row["seq_id"], "plot_label": row["plot_label"]}
        for target_id, target_vec in target_vectors.items():
            rec[f"cosine_to_{target_id}"] = float(cosine_distances(vec, target_vec.reshape(1, -1))[0, 0])
            rec[f"euclidean_to_{target_id}"] = float(euclidean_distances(vec, target_vec.reshape(1, -1))[0, 0])
        records.append(rec)
    pd.DataFrame(records).to_csv(OUT_CSV, index=False)

    font = FontProperties(fname=str(FONT_PATH))
    bold_font = FontProperties(fname=str(BOLD_FONT_PATH))
    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)

    vaccine = df[df["group"] == "vaccine"]
    ax.scatter(vaccine["PC1"], vaccine["PC2"], s=80, c="#4c78a8", alpha=0.85, label="历年疫苗株")
    for _, row in vaccine.iterrows():
        ax.text(row["PC1"], row["PC2"], row["plot_label"], fontsize=7, fontproperties=font, ha="left", va="bottom")

    targets = df[df["group"] == "target"]
    colors = ["#e45756", "#f58518"]
    for color, (_, row) in zip(colors, targets.iterrows()):
        ax.scatter(row["PC1"], row["PC2"], s=180, marker="*", c=color, edgecolor="black", linewidth=0.8, label=row["plot_label"])
        ax.text(row["PC1"], row["PC2"], row["plot_label"], fontsize=10, fontproperties=bold_font, ha="left", va="top")

    if len(targets) == 2:
        x = targets["PC1"].to_numpy()
        y = targets["PC2"].to_numpy()
        ax.plot(x, y, color="#444444", linewidth=1.2, linestyle="--", alpha=0.8)

    ax.set_title("两条目标序列与历年疫苗株的 LucaVirus embedding PCA 分布", fontproperties=bold_font, fontsize=16)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)", fontproperties=font)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)", fontproperties=font)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend(prop=font, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(OUT_PNG)
    print(OUT_CSV)
    print(df[["seq_id", "plot_label", "group", "PC1", "PC2"]].to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
