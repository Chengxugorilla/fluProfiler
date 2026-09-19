#!/usr/bin/env python3
"""
CPU smoke test for the SerumGate trainer.
"""

from __future__ import annotations

import tempfile
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from experiments.serum_gate.train_zero_shot import run_training


def _row(task_idx: int, query_idx: int, split_offset: int = 0) -> dict:
    ref_idx = task_idx + split_offset
    return {
        "seq_id_a": f"ref_ha_{ref_idx}",
        "seq_id_b": f"ref_na_{ref_idx}",
        "seq_id_c": f"test_ha_{ref_idx}_{query_idx}",
        "seq_id_d": f"test_na_{ref_idx}_{query_idx}",
        "seq_a": "AAAA",
        "seq_b": "ANVTAAANST",
        "seq_c": "AAAT",
        "seq_d": "ANVTAAAAAA",
        "serumPassCat": "<EGG>" if task_idx % 2 == 0 else "<CELL>",
        "virusPassCat": "<CELL>",
        "serumName": f"serum_{ref_idx}",
        "virusName": f"virus_{ref_idx}_{query_idx}",
        "label": float(query_idx) + 0.25 * float(task_idx),
        "serumDate": "2020-01-01",
        "Type": "H3N2",
        "virusDate": "2021-01-01",
        "serumIslID": f"ref_{ref_idx}",
        "virusIslID": f"test_{ref_idx}_{query_idx}",
        "sheet": "smoke",
        "serumHA": "A" * 20,
        "virusHA": "A" * 19 + "T",
    }


def _write_split(data_dir: Path) -> pd.DataFrame:
    rows = []
    for task_idx in range(9):
        for query_idx in range(3):
            rows.append(_row(task_idx, query_idx))
    frame = pd.DataFrame(rows)
    frame.iloc[:15].to_csv(data_dir / "train.csv", index=False)
    frame.iloc[15:21].to_csv(data_dir / "valid.csv", index=False)
    frame.iloc[21:].to_csv(data_dir / "test.csv", index=False)
    return frame


def _write_embeddings(frame: pd.DataFrame, embedding_dir: Path) -> None:
    ids = sorted(set(frame["seq_id_a"].tolist()) | set(frame["seq_id_c"].tolist()))
    for idx, seq_id in enumerate(ids):
        torch.save(torch.full((4, 6), float(idx) / 10.0), embedding_dir / f"matrix_{seq_id}.pt")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "data"
        embedding_dir = root / "embeddings"
        output_dir = root / "out"
        data_dir.mkdir()
        embedding_dir.mkdir()
        frame = _write_split(data_dir)
        _write_embeddings(frame, embedding_dir)
        run_training(
            Namespace(
                data_dir=data_dir,
                embedding_dir=embedding_dir,
                output_dir=output_dir,
                serum_task_cols="seq_id_a,seq_id_b,serumPassCat",
                strict_group_resplit=True,
                preserve_test_split=True,
                refit_train_valid=False,
                allow_task_overlap=False,
                train_ratio=0.6,
                valid_ratio=0.2,
                sample_limit=-1,
                batch_size=1,
                max_queries_per_task=2,
                epochs=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                latent_dim=4,
                theta_dim=4,
                predictor_arch="calibrated_metric",
                predictor_hidden_dim=8,
                distance_hidden_dim=5,
                calibration_hidden_dim=6,
                residual_hidden_dim=4,
                residual_scale=0.25,
                na_branch="none",
                na_pooling="mean",
                na_latent_dim=4,
                na_hidden_dim=4,
                na_effect_init=0.1,
                ha_pooling="lowrank_attention",
                ha_pair_mode="independent",
                ha_attention_dim=4,
                ha_attention_heads=2,
                ha_attention_dropout=0.1,
                ha_mean_gate_init=0.25,
                label_weight_thresholds="2,4,6",
                label_weight_threshold_mode="fixed",
                label_weight_quantiles="0.35,0.75,0.95",
                label_weight_values="1,1.3,1.8,2.5",
                rank_loss_weight=0.1,
                rank_loss_margin=0.1,
                rank_loss_min_label_delta=0.0,
                loss="nll",
                device="cpu",
                gpu_cache_gb=0.0,
                seed=3,
                progress=False,
            )
        )
        assert (output_dir / "checkpoints" / "best_model.pth").is_file()
        assert (output_dir / "metrics.csv").is_file()
        assert not (output_dir / "metrics.jsonl").exists()


if __name__ == "__main__":
    main()
