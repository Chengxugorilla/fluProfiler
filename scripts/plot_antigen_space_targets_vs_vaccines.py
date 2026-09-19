#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fluprofiler")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


JOB_DIR = Path("/home/chenyh/workspace/fluProfiler/temp job")
EMBEDDING_DIR = JOB_DIR / "embedding"
TARGET_FASTA = JOB_DIR / "strain76vsvaccine26.fasta"
VACCINE_FASTA = JOB_DIR / "H1N1_vaccine_origin.fasta"
MODEL_PATH = Path("/home/chenyh/workspace/2025-08-19_17-43-32.pth")
PIPELINE_SRC_DIR = Path("/home/chenyh/workspace/fluProfiler_pipeline/src")
OUT_PNG = JOB_DIR / "strain76_vaccine26_vs_vaccines_antigen_space_pca.png"
OUT_COORDS_CSV = JOB_DIR / "strain76_vaccine26_vs_vaccines_antigen_space_coords.csv"
OUT_DISTANCES_CSV = JOB_DIR / "strain76_vaccine26_vs_vaccines_antigen_space_distances.csv"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
BOLD_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
TARGET_LABELS = ("1976 New Jersey", "2025 Missouri / 2026 vaccine")


class Emb2Vec(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.matrix_dropout = model.matrix_dropout
        self.matrix_pooler = model.matrix_pooler
        self.linear = model.linear[0][0]

    def forward(self, embedding):
        embedding = self.matrix_dropout(embedding)
        seq_vector = self.matrix_pooler(embedding)
        return self.linear(seq_vector)


def import_model_classes(pipeline_src_dir: Path) -> None:
    src_dir = str(pipeline_src_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import fluProfiler_models  # noqa: F401


def accession_candidates(record_id: str) -> list[str]:
    parts = [part.strip() for part in record_id.split("|")]
    return [part for part in reversed(parts) if part.startswith("EPI")]


def embedding_id(record_id: str, embedding_dir: Path = EMBEDDING_DIR) -> str:
    candidates = accession_candidates(record_id)
    for candidate in candidates:
        if (embedding_dir / f"matrix_{candidate}.pt").is_file():
            return candidate
    return candidates[0] if candidates else record_id.split("|")[0].strip()


def target_rows(
    target_fasta: Path = TARGET_FASTA,
    embedding_dir: Path = EMBEDDING_DIR,
    labels: tuple[str, ...] = TARGET_LABELS,
) -> list[dict[str, str]]:
    rows = []
    for label, record in zip(labels, SeqIO.parse(target_fasta, "fasta")):
        rows.append(
            {
                "seq_id": embedding_id(record.description, embedding_dir),
                "plot_label": label,
                "group": "target",
                "record_id": record.description,
            }
        )
    return rows


def vaccine_rows(vaccine_fasta: Path = VACCINE_FASTA, embedding_dir: Path = EMBEDDING_DIR) -> list[dict[str, str]]:
    rows = []
    for record in SeqIO.parse(vaccine_fasta, "fasta"):
        seq_id = embedding_id(record.description, embedding_dir)
        rows.append(
            {
                "seq_id": seq_id,
                "plot_label": seq_id.replace("EPI_ISL_", "ISL_"),
                "group": "vaccine",
                "record_id": record.description,
            }
        )
    return rows


def unique_rows_by_seq_id(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    unique_rows = []
    for row in rows:
        seq_id = row["seq_id"]
        if seq_id in seen:
            continue
        unique_rows.append(row)
        seen.add(seq_id)
    return unique_rows


def build_plot_rows(target_fasta: Path, vaccine_fasta: Path, embedding_dir: Path) -> list[dict[str, str]]:
    rows = target_rows(target_fasta, embedding_dir) + vaccine_rows(vaccine_fasta, embedding_dir)
    return unique_rows_by_seq_id(rows)


def load_one_pt(pt_path: Path) -> torch.Tensor:
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        for value in obj.values():
            if torch.is_tensor(value):
                return value
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
    raise ValueError(f"Unrecognized pt format: {pt_path}, type={type(obj)}")


def load_antigen_vector(seq_id: str, embedding_dir: Path, emb2vec: nn.Module) -> np.ndarray:
    pt_path = embedding_dir / f"matrix_{seq_id}.pt"
    if not pt_path.is_file():
        raise FileNotFoundError(pt_path)

    emb = load_one_pt(pt_path)
    if emb.dim() == 2:
        emb = emb.unsqueeze(0)

    with torch.no_grad():
        vector = emb2vec(emb)[0].cpu().float().numpy()
    return vector


def load_antigen_vectors(rows: list[dict[str, str]], embedding_dir: Path, emb2vec: nn.Module) -> np.ndarray:
    return np.vstack([load_antigen_vector(row["seq_id"], embedding_dir, emb2vec) for row in rows])


def fit_pca(vectors: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(vectors)
    return coords, pca.explained_variance_ratio_ * 100


def distance_rows(df: pd.DataFrame, vectors: np.ndarray) -> list[dict[str, float | str]]:
    target_df = df[df["group"] == "target"]
    target_vectors = {
        row["seq_id"]: vectors[df.index[df["seq_id"] == row["seq_id"]][0]]
        for _, row in target_df.iterrows()
    }

    records = []
    for _, row in df.iterrows():
        if row["seq_id"] in target_vectors:
            continue
        idx = df.index[df["seq_id"] == row["seq_id"]][0]
        vec = vectors[idx : idx + 1]
        rec: dict[str, float | str] = {"seq_id": row["seq_id"], "plot_label": row["plot_label"]}
        for target_id, target_vec in target_vectors.items():
            target_vec = target_vec.reshape(1, -1)
            rec[f"cosine_to_{target_id}"] = float(cosine_distances(vec, target_vec)[0, 0])
            rec[f"euclidean_to_{target_id}"] = float(euclidean_distances(vec, target_vec)[0, 0])
        records.append(rec)
    return records


def font(path: Path) -> FontProperties | None:
    if path.is_file():
        return FontProperties(fname=str(path))
    return None


def plot_antigen_space(df: pd.DataFrame, explained: np.ndarray, out_png: Path) -> None:
    regular_font = font(FONT_PATH)
    bold_font = font(BOLD_FONT_PATH) or regular_font

    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)

    vaccine = df[df["group"] == "vaccine"]
    ax.scatter(vaccine["PC1"], vaccine["PC2"], s=80, c="#4c78a8", alpha=0.85, label="历年疫苗株")
    for _, row in vaccine.iterrows():
        ax.text(row["PC1"], row["PC2"], row["plot_label"], fontsize=7, fontproperties=regular_font, ha="left", va="bottom")

    targets = df[df["group"] == "target"]
    colors = ["#e45756", "#f58518"]
    for color, (_, row) in zip(colors, targets.iterrows()):
        ax.scatter(
            row["PC1"],
            row["PC2"],
            s=180,
            marker="*",
            c=color,
            edgecolor="black",
            linewidth=0.8,
            label=row["plot_label"],
        )
        ax.text(row["PC1"], row["PC2"], row["plot_label"], fontsize=10, fontproperties=bold_font, ha="left", va="top")

    if len(targets) == 2:
        ax.plot(
            targets["PC1"].to_numpy(),
            targets["PC2"].to_numpy(),
            color="#444444",
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
        )

    ax.set_title("两条目标序列与历年疫苗株的 fluProfiler 抗原性空间 PCA 分布", fontproperties=bold_font, fontsize=16)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)", fontproperties=regular_font)
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)", fontproperties=regular_font)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend(prop=regular_font, loc="best", frameon=True)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def load_emb2vec(model_path: Path, pipeline_src_dir: Path) -> Emb2Vec:
    import_model_classes(pipeline_src_dir)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    return Emb2Vec(model).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 1976/2026 vaccine targets and historical vaccines in fluProfiler antigen space.")
    parser.add_argument("--target-fasta", type=Path, default=TARGET_FASTA)
    parser.add_argument("--vaccine-fasta", type=Path, default=VACCINE_FASTA)
    parser.add_argument("--embedding-dir", type=Path, default=EMBEDDING_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--pipeline-src-dir", type=Path, default=PIPELINE_SRC_DIR)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    parser.add_argument("--out-coords-csv", type=Path, default=OUT_COORDS_CSV)
    parser.add_argument("--out-distances-csv", type=Path, default=OUT_DISTANCES_CSV)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_plot_rows(args.target_fasta, args.vaccine_fasta, args.embedding_dir)
    emb2vec = load_emb2vec(args.model_path, args.pipeline_src_dir)
    vectors = load_antigen_vectors(rows, args.embedding_dir, emb2vec)
    coords, explained = fit_pca(vectors, args.seed)

    df = pd.DataFrame(rows)
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]
    df.to_csv(args.out_coords_csv, index=False)
    pd.DataFrame(distance_rows(df, vectors)).to_csv(args.out_distances_csv, index=False)
    plot_antigen_space(df, explained, args.out_png)

    print(args.out_png)
    print(args.out_coords_csv)
    print(args.out_distances_csv)
    print(df[["seq_id", "plot_label", "group", "PC1", "PC2"]].to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
