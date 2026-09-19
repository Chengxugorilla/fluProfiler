#!/usr/bin/env python3
"""Reproduce the Figure 4 vaccine-ranking workflow with SCMS-FiLM checkpoints.

This is deliberately independent from ``Fig.4 (Vaccine_selection).ipynb``.
For each target season, it scores every candidate serum against every virus in
that season, then ranks candidates by their mean predicted antigenic distance.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parent
SUPPLEMENTARY_EMBEDDING_DIR = ANALYSIS_DIR / "assets/embeddings"
DARWIN_ID = "DARWIN6_2021_CELL_HA1"
sys.path.insert(0, str(REPO_ROOT))

from experiments.serum_gate import train_serum_mutation_set as scms  # noqa: E402
from fluprofiler.models.serum_mutation_set_model import (  # noqa: E402
    SerumMutationSetConfig,
    SerumMutationSetMinusModel,
)


SEASONS = {
    "39": "2023Feb", "40": "2023Sep", "41": "2024Feb",
    "42": "2024Sep", "43": "2025Feb", "44": "2025Sep",
}
WHO = {
    "39": ["A/DARWIN/9/2021 EGG", "A/DARWIN/6/2021 CELL"],
    "40": ["A/THAILAND/8/2022 EGG", "A/MASSACHUSETTS/18/2022 CELL"],
    "41": ["A/THAILAND/8/2022 EGG", "A/MASSACHUSETTS/18/2022 CELL"],
    "42": ["A/CROATIA/10136RV/2023 EGG", "A/DISTRICTOFCOLUMBIA/27/2023 CELL"],
    "43": ["A/CROATIA/10136RV/2023 EGG", "A/DISTRICTOFCOLUMBIA/27/2023 CELL"],
    "44": ["A/SINGAPORE/GP20238/2024 EGG", "A/SYDNEY/1359/2024 CELL"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="39,40,41,42,43,44")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpu-cache-gb", type=float, default=0)
    p.add_argument("--max-queries-per-task", type=int, default=32)
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    return p.parse_args()


def make_ha_lookup(source: pd.DataFrame) -> dict[str, str]:
    """Map either HA sequence ID to its aligned HA sequence."""
    serum = source[["seq_id_a", "serumHA"]].rename(columns={"seq_id_a": "id", "serumHA": "ha"})
    virus = source[["seq_id_c", "virusHA"]].rename(columns={"seq_id_c": "id", "virusHA": "ha"})
    merged = pd.concat([serum, virus], ignore_index=True).dropna().drop_duplicates("id")
    return dict(zip(merged["id"].astype(str), merged["ha"].astype(str)))


def supplement_candidates(source: pd.DataFrame, lookup: dict[str, str]) -> pd.DataFrame:
    raw = json.loads((REPO_ROOT / "paper/data/supplementary_vaccines.json").read_text())
    # Darwin/6/2021 CELL is the Figure 4A supplementary vaccine.  Its exact
    # HA1 sequence is residues 17--345 of the legacy full-HA record.
    darwin = raw["A/DARWIN/6/2021"].copy()
    darwin["seq_id_a"] = DARWIN_ID
    darwin["serumHA"] = str(darwin["seq_a"])[16:345]
    darwin["Type"] = "H3N2"
    darwin_frame = pd.DataFrame([darwin])

    extra = pd.DataFrame(raw.values()).copy()
    extra["serumHA"] = extra["seq_id_a"].astype(str).map(lookup)
    extra = pd.concat([extra, darwin_frame], ignore_index=True, sort=False)
    supported = extra["serumHA"].notna() & extra["seq_id_a"].map(
        lambda seq_id: any((directory / f"matrix_{seq_id}.pt").is_file() for directory in (
            REPO_ROOT / "data/embedding/files", SUPPLEMENTARY_EMBEDDING_DIR
        ))
    )
    skipped = extra.loc[~supported, ["serumName", "seq_id_a"]]
    if not skipped.empty:
        print(
            "Skipping supplementary candidates without HA1 alignment/embedding: "
            + ", ".join(skipped["serumName"].tolist()),
            file=sys.stderr,
        )
    extra = extra.loc[supported].copy()
    extra["Type"] = "H3N2"
    return extra


def candidate_and_virus_tables(source: pd.DataFrame, season: str, extra: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_key = source["sheet"].astype(str).str.split("-", n=1).str[0]
    # This intentionally mirrors prepare_tables() in the legacy Figure 4
    # notebook: a recommendation panel evaluates every serum available up to
    # and including its target season, against viruses from that target season.
    prior = source.loc[season_key.astype(int) <= int(season)].copy()
    target = source.loc[season_key == season].copy()
    prior = prior.loc[prior["Type"].eq("H3N2")]
    target = target.loc[target["Type"].eq("H3N2")]
    serum = pd.concat([prior, extra], ignore_index=True, sort=False)
    # Figure 4 candidates are uniquely identified by their displayed vaccine
    # name and passage category.  Do not let alternate sequence identifiers
    # duplicate (and therefore reweight) the same named vaccine.
    serum = serum.sort_values(["serumName", "serumPassCat", "seq_id_a"]).drop_duplicates(
        ["serumName", "serumPassCat"], keep="first"
    )
    serum = serum[["seq_id_a", "serumHA", "serumPassCat", "serumName", "serumIslID"]]
    # Likewise, retain one record per named circulating virus and passage
    # category; sequence IDs are not used as a separate weighting factor.
    virus = target.sort_values(["virusName", "virusPassCat", "seq_id_c"]).drop_duplicates(
        ["virusName", "virusPassCat"], keep="first"
    )
    virus = virus[["seq_id_c", "virusHA", "virusPassCat", "virusName", "virusDate", "virusIslID"]]
    if serum.empty or virus.empty:
        raise ValueError(f"season {season}: empty candidate serum or target-virus table")
    return serum.reset_index(drop=True), virus.reset_index(drop=True)


def legacy_supplement_candidates(source: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Recover legacy supplementary sera using embeddings available to SCMS-FiLM.

    The legacy JSON contains full-HA/NA records whose original embedding store is
    no longer present.  Prefer an exact named serum from the HA1 source table.
    If the name is absent, an exact HA1-sequence and passage match may reuse the
    corresponding source embedding: the vaccine retains its own name and full
    HA/NA grouping sequences, while SCMS-FiLM receives the correct HA1 asset.
    Darwin/6 CELL uses the dedicated HA1 asset bundled with this analysis.
    """
    raw = json.loads((REPO_ROOT / "paper/data/supplementary_vaccines.json").read_text())
    rows: list[dict] = []
    missing: list[str] = []
    for name, record in raw.items():
        passage = record["serumPassCat"]
        matched = source.loc[
            source["serumName"].eq(name) & source["serumPassCat"].eq(passage)
        ].sort_values("seq_id_a")
        if not matched.empty:
            row = matched.iloc[0].to_dict()
        else:
            # Supplementary records contain full HA.  The HA1 dataset uses
            # residues 17--345 (zero-based slice 16:345), as for Darwin below.
            ha1 = str(record["seq_a"])[16:345]
            sequence_match = source.loc[
                source["serumHA"].eq(ha1) & source["serumPassCat"].eq(passage)
            ].sort_values("seq_id_a")
            if not sequence_match.empty:
                # Start from the supplementary record so seq_a/seq_b remain
                # its legacy identifiers; only the HA1 embedding key is shared.
                row = dict(record)
                row["seq_id_a"] = sequence_match.iloc[0]["seq_id_a"]
                row["serumHA"] = ha1
            elif name == "A/DARWIN/6/2021" and (
                SUPPLEMENTARY_EMBEDDING_DIR / f"matrix_{DARWIN_ID}.pt"
            ).is_file():
                row = dict(record)
                row["seq_id_a"] = DARWIN_ID
                row["serumHA"] = ha1
            else:
                missing.append(f"{name} {passage.strip('<>')}")
                continue
        # Preserve the legacy grouping sequences from the supplementary JSON.
        row["seq_a"] = record.get("seq_a", row.get("seq_a"))
        row["seq_b"] = record.get("seq_b", row.get("seq_b"))
        row["Type"] = "H3N2"
        rows.append(row)
    return pd.DataFrame(rows), missing


def legacy_candidate_and_virus_tables(
    source: pd.DataFrame,
    season: str,
    extra: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce ``prepare_tables`` from the legacy Figure 4 notebook."""
    season_key = source["sheet"].astype(str).str.split("-", n=1).str[0]
    candidates = source.loc[season_key.astype(int) <= int(season)]
    target = source.loc[season_key.eq(season)]
    candidates = candidates.loc[candidates["Type"].eq("H3N2")]
    target = target.loc[target["Type"].eq("H3N2")]

    # A Figure 4 candidate denotes a named vaccine virus in a particular
    # passage category, not every legacy sequence record for that vaccine.
    # Combine supplementary records before deduplicating so they cannot add a
    # second copy of an existing name/passage candidate.
    serum_source = candidates.copy()
    if extra is not None and not extra.empty:
        serum_source = pd.concat([serum_source, extra], ignore_index=True, sort=False)

    # Keep the original values for plotting, but compare normalized keys so
    # whitespace/capitalization variants do not become separate candidates.
    serum_source["_serum_name_key"] = (
        serum_source["serumName"].astype("string").str.strip().str.casefold()
    )
    serum_source["_serum_pass_key"] = (
        serum_source["serumPassCat"].astype("string").str.strip().str.casefold()
    )
    candidate_keys = ["_serum_name_key", "_serum_pass_key"]

    sequence_counts = (
        serum_source.drop_duplicates(candidate_keys + ["seq_a", "seq_b"])
        .groupby(candidate_keys, dropna=False)
        .size()
    )
    conflicts = sequence_counts[sequence_counts.gt(1)]
    if not conflicts.empty:
        warnings.warn(
            f"{len(conflicts)} serumName/serumPassCat candidate(s) have "
            "multiple legacy sequences; using the lowest seq_id_a/seq_id_b "
            "representative for each.",
            stacklevel=2,
        )

    serum = (
        serum_source.sort_values(
            candidate_keys + ["seq_id_a", "seq_id_b"], kind="stable"
        )
        .groupby(candidate_keys, as_index=False, sort=False, dropna=False)
        .agg(
            seq_a=("seq_a", "first"),
            seq_b=("seq_b", "first"),
            seq_id_a=("seq_id_a", "first"),
            seq_id_b=("seq_id_b", "first"),
            serumHA=("serumHA", "first"),
            serumPassCat=("serumPassCat", "first"),
            label=("label", "mean"),
            serumName=("serumName", "first"),
            serumIslID=("serumIslID", "first"),
        )
        .reset_index()
        .drop(columns=candidate_keys)
    )
    virus = (
        target.groupby(["seq_c", "seq_d", "virusPassCat"])
        .agg(
            seq_id_c=("seq_id_c", "first"),
            seq_id_d=("seq_id_d", "first"),
            virusHA=("virusHA", "first"),
            label=("label", "mean"),
            virusName=("virusName", "first"),
            virusDate=("virusDate", "first"),
            virusIslID=("virusIslID", "first"),
        )
        .reset_index()
    )
    return serum, virus


def legacy_future_who_candidates(source: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Match the special 2023Sep WHO-candidate construction in the old notebook."""
    season_key = source["sheet"].astype(str).str.split("-", n=1).str[0]
    frame = source.loc[season_key.eq("41") & source["serumName"].isin(names)]
    return (
        frame.groupby(["seq_a", "seq_b", "serumPassCat"])
        .agg(
            seq_id_a=("seq_id_a", "first"), seq_id_b=("seq_id_b", "first"),
            serumHA=("serumHA", "first"), label=("label", "mean"),
            serumName=("serumName", "first"), serumIslID=("serumIslID", "first"),
        )
        .reset_index()
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SerumMutationSetMinusModel, scms.MutationSetVocabs]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = SerumMutationSetConfig(**checkpoint["model_config"])
    distance = scms.load_ha_distance_matrix(REPO_ROOT / "ha1_distance_no_bias_329.npy")
    model = SerumMutationSetMinusModel(config, distance)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    vocabs = scms.MutationSetVocabs(
        passage_to_id=checkpoint["passage_to_id"],
        subtype_to_id=checkpoint["subtype_to_id"],
        serum_name_to_id=checkpoint.get("serum_name_to_id", {"": 0}),
        query_virus_to_id=checkpoint.get("query_virus_to_id", {"": 0}),
    )
    return model, vocabs


def predict_pairs(frame: pd.DataFrame, model: SerumMutationSetMinusModel, vocabs: scms.MutationSetVocabs,
                  device: torch.device, max_queries: int, cache_gb: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["label"] = 0.0
    frame["Type"] = "H3N2"
    files = scms.required_ha_embedding_files({"pairs": frame})
    embeddings = {}
    for filename in files:
        path = next((directory / filename for directory in (
            REPO_ROOT / "data/embedding/files", SUPPLEMENTARY_EMBEDDING_DIR
        ) if (directory / filename).is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing embedding: {filename}")
        # Existing assets may be saved as Tensor, while the one-off LucaVirus
        # supplementary asset can be saved as ndarray.
        embeddings[filename.removesuffix(".pt")] = torch.as_tensor(
            torch.load(path, map_location="cpu", weights_only=False)
        ).float()
    embedding_source = embeddings
    if cache_gb > 0:
        embedding_source = scms.GpuEmbeddingCache(embeddings, device, int(cache_gb * 1024**3))
    loader = scms.build_loader(
        frame, vocabs, embedding_source, batch_size=1, shuffle=False,
        max_queries_per_task=max_queries,
        task_cols=["seq_id_a", "serumPassCat", "serumName"],
        aligned_cache=scms.AlignedEmbeddingCache(),
    )
    _, predictions = scms.evaluate_model(model, loader, device, full_task_bias_loss=True)
    return predictions


def rank_candidates(predictions: pd.DataFrame) -> pd.DataFrame:
    # Figure 4's build_vaccine_table(..., clip_negative=True) clips individual
    # predicted antigenic distances before calculating each vaccine mean.
    predictions = predictions.copy()
    predictions["mean"] = predictions["mean"].clip(lower=0)
    grouped = predictions.groupby(
        ["serumName", "serumPassCat"], as_index=False
    ).agg(seq_id_a=("seq_id_a", "first"), serumIslID=("serumIslID", "first"),
          predicted_distance=("mean", "mean"), virus_count=("virusName", "nunique"),
          predicted_distance_median=("mean", "median"))
    grouped["coverage_score"] = -grouped["predicted_distance"]
    grouped["vaccine"] = grouped["serumName"] + " " + grouped["serumPassCat"].str.strip("<>")
    return grouped.sort_values("predicted_distance", ascending=True).reset_index(drop=True)


def legacy_rank_candidates(
    predictions: pd.DataFrame,
    *,
    include_serum_name: bool = False,
    weight_by_collection_date: bool = False,
) -> pd.DataFrame:
    """Reproduce the old notebook's ``build_vaccine_table`` calculation."""
    frame = predictions.copy()
    frame["mean"] = frame["mean"].clip(lower=0)
    group_cols = ["seq_a", "seq_b", "serumPassCat"]
    if include_serum_name:
        group_cols.append("serumName")

    if weight_by_collection_date:
        dates = pd.to_datetime(frame["virusDate"], errors="coerce")
        ordinals = dates.map(lambda value: value.toordinal() if pd.notna(value) else float("nan"))
        if ordinals.notna().any() and ordinals.max() != ordinals.min():
            weights = ordinals.rank(method="first", ascending=True)
            frame["_date_weight"] = weights / weights.sum()
        else:
            frame["_date_weight"] = 1.0
        frame["_date_weight"] = frame["_date_weight"].fillna(1.0)

        def summarize(group: pd.DataFrame) -> pd.Series:
            return pd.Series({
                "serumName": group["serumName"].iloc[0],
                "serumIslID": group["serumIslID"].iloc[0],
                "seq_id_a": group["seq_id_a"].iloc[0],
                "predicted_distance": float(
                    (group["mean"] * group["_date_weight"]).sum()
                    / group["_date_weight"].sum()
                ),
                "virus_count": len(group),
            })

        grouped = frame.groupby(group_cols, as_index=False).apply(
            summarize, include_groups=False
        ).reset_index(drop=True)
    else:
        aggregations = {
            "predicted_distance": ("mean", "mean"),
            "serumIslID": ("serumIslID", "first"),
            "seq_id_a": ("seq_id_a", "first"),
            "virus_count": ("seq_c", "size"),
        }
        if not include_serum_name:
            aggregations["serumName"] = ("serumName", "first")
        grouped = frame.groupby(group_cols, as_index=False).agg(**aggregations)
    grouped["coverage_score"] = -grouped["predicted_distance"] + 2.0
    grouped["vaccine"] = grouped["serumName"] + " " + grouped["serumPassCat"].str.strip("<>")
    return grouped.sort_values("coverage_score", ascending=False).reset_index(drop=True)


def plot_ranking(
    ranking: pd.DataFrame,
    season: str,
    output: Path,
    top_n: int,
    highlights: list[str] | None = None,
    candidate_count: int | None = None,
) -> None:
    """Draw the ranking with the *exact* legacy Figure 4 bar-plot grammar.

    Keep this deliberately separate from the SCMS-FiLM scoring code: the
    reference notebook distinguishes WHO strains only with a star in their
    tick label--it does not recolour their bars or add a panel title.
    """
    highlights = highlights if highlights is not None else WHO[season]
    top = ranking.sort_values("coverage_score", ascending=False).head(top_n).copy()
    top["vaccine_label"] = (
        top["vaccine"].str.replace(r"^A/", "", regex=True)
        .str.replace(" EGG", " (EGG)", regex=False)
        .str.replace(" CELL", " (CELL)", regex=False)
    )
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 7, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3, "ytick.major.size": 3, "figure.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(4, 3.4))
    sns.barplot(data=top, y="vaccine_label", x="coverage_score", color="#2E6F8E", edgecolor="none", ax=ax)
    for patch, vaccine_name in zip(ax.patches, top["vaccine"]):
        if vaccine_name in highlights:
            patch.set_facecolor("#C44E52")
    ax.axvline(0, color="0.2", lw=0.8)
    ax.xaxis.grid(True, which="major", color="0.9", lw=0.6)
    ax.yaxis.grid(False)
    sns.despine(ax=ax, left=True, bottom=False)
    ax.set_xlabel("Predicted coverage score")
    ax.set_ylabel("")
    # Do not add a title: the source notebook calls plot_vaccine_top_bar()
    # without one, and panel letters/titles are composed in the manuscript.
    ax.set_title(None, pad=6)
    if candidate_count is not None:
        ax.text(
            0.98, 0.98, f"Candidate sera: n={candidate_count}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
        )
    ax.tick_params(axis="y", pad=2)
    labels = [f"★ {label}" if name in highlights else label for label, name in zip(top["vaccine_label"], top["vaccine"])]
    ax.set_yticklabels(labels)
    xmin, xmax = top["coverage_score"].min(), top["coverage_score"].max()
    pad = (xmax - xmin) * 0.06 if xmax > xmin else 0.1
    ax.set_xlim(xmin - pad, xmax + pad)
    fig.tight_layout()
    fig.savefig(output, dpi=300, transparent=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    requested = [s.strip() for s in args.seasons.split(",") if s.strip()]
    unknown = sorted(set(requested) - set(SEASONS))
    if unknown:
        raise ValueError(f"Unknown season(s): {unknown}")
    device = torch.device(args.device)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(REPO_ROOT / "data/dataset/H3_HA1_v1.0/processed/source.csv")
    # The legacy notebook has four supplementary full-HA candidates.  They are
    # included only if equivalent HA1-aligned sequences and HA1 embeddings are
    # present; this prevents mixing a full-HA asset into the HA1 model.
    extra = supplement_candidates(source, make_ha_lookup(source))
    for season in requested:
        run_root = REPO_ROOT / "results/H3_HA1_v1.0/20260902_161615/SCMS-FiLM-H3/season" / season / "subtype/H3N2_seed42"
        checkpoint = run_root / "checkpoints/best_model.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        season_output = output_root / f"season_{season}_{SEASONS[season]}"
        season_output.mkdir(exist_ok=True)
        serum, virus = candidate_and_virus_tables(source, season, extra)
        pairs = serum.merge(virus, how="cross")
        model, vocabs = load_model(checkpoint, device)
        predictions = predict_pairs(pairs, model, vocabs, device, args.max_queries_per_task, args.gpu_cache_gb)
        ranking = rank_candidates(predictions)
        predictions.to_csv(season_output / "pair_predictions.csv", index=False)
        ranking.to_csv(season_output / "vaccine_ranking.csv", index=False)
        plot_ranking(ranking, season, season_output / f"Fig4_{season}_{SEASONS[season]}.svg", args.top_n)
        (season_output / "run_metadata.json").write_text(json.dumps({
            "season": season, "figure_panel": SEASONS[season], "checkpoint": str(checkpoint),
            "candidate_serum_count": len(serum), "target_virus_count": len(virus),
            "pair_count": len(pairs), "max_queries_per_task": args.max_queries_per_task,
            "score_definition": "coverage_score = - mean predicted antigenic distance",
        }, indent=2), encoding="utf-8")
        print(f"completed season {season}: {len(serum)} candidates × {len(virus)} viruses")


if __name__ == "__main__":
    main()
