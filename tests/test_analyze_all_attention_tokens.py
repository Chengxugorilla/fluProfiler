from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
MODULE_PATH = REPO_ROOT / "experiments" / "serum_gate" / "analyze_all_attention_tokens.py"
SPEC = importlib.util.spec_from_file_location("analyze_all_attention_tokens", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

from fluprofiler.models.serum_gate_minus_all_attention_model import SerumGateMinusAllAttentionModel  # noqa: E402
from fluprofiler.models.serum_gate_minus_model import SerumGateMinusConfig  # noqa: E402


def test_collect_unique_virus_sequences_filters_and_counts(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "Type": ["H3N2", "H3N2", "H1N1", "H3N2"],
            "seq_id_c": ["HA_1", "HA_1", "HA_2", "HA_3"],
            "seq_c": ["A" * 329, "A" * 329, "B" * 329, "C" * 328],
            "virusHA": ["AAAA", "AAAA", "BBBB", "CCCC"],
            "virusName": ["v1", "v1-repeat", "v2", "v3"],
            "virusDate": ["2020", "2021", "2019", "2022"],
        }
    )
    csv_path = tmp_path / "whole.csv"
    frame.to_csv(csv_path, index=False)

    records = analysis.collect_unique_virus_sequences(csv_path, "H3N2")

    assert records["seq_id_c"].tolist() == ["HA_1", "HA_3"]
    assert records["virusHA"].tolist() == ["AAAA", "CCCC"]
    assert records["occurrence_count"].tolist() == [2, 1]


def test_collect_unique_virus_sequences_rejects_inconsistent_mapping(tmp_path: Path) -> None:
    csv_path = tmp_path / "whole.csv"
    pd.DataFrame(
        {
            "Type": ["H3N2", "H3N2"],
            "seq_id_c": ["HA_1", "HA_1"],
            "seq_c": ["A" * 329, "A" * 329],
            "virusHA": ["AAAA", "AAAT"],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="maps to multiple virusHA sequences"):
        analysis.collect_unique_virus_sequences(csv_path, "H3N2")


def test_summarize_attention_preserves_all_tokens() -> None:
    attention = np.asarray(
        [
            [0.05, 0.10, 0.15, 0.20, 0.40, 0.10],
            [0.10, 0.20, 0.10, 0.30, 0.20, 0.10],
        ],
        dtype=np.float64,
    )

    summary, ranked = analysis.summarize_attention(attention, token_count=6)

    assert summary["token_label"].tolist() == ["BOS", "AA_1", "AA_2", "AA_3", "AA_4", "EOS"]
    assert summary["token_type"].tolist() == ["special", "amino_acid", "amino_acid", "amino_acid", "amino_acid", "special"]
    assert summary["count"].tolist() == [2] * 6
    assert np.allclose(summary["mean"], attention.mean(axis=0))
    assert summary.loc[4, "top1_count"] == 1
    assert summary.loc[3, "top1_count"] == 1
    assert ranked.iloc[0]["mean"] >= ranked.iloc[-1]["mean"]
    assert ranked["rank"].tolist() == list(range(1, 7))


def test_validate_attention_checks_probability_rows() -> None:
    attention = np.asarray([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]], dtype=np.float64)

    checks = analysis.validate_attention(attention, expected_rows=2, token_count=3)

    assert checks["row_count"] == 2
    assert checks["token_count"] == 3
    assert checks["minimum_attention"] == pytest.approx(0.1)
    assert checks["maximum_attention"] == pytest.approx(0.5)
    assert checks["max_row_sum_error"] < 1e-12

    with pytest.raises(ValueError, match="sum to one"):
        analysis.validate_attention(attention * 0.5, expected_rows=2, token_count=3)
    with pytest.raises(ValueError, match="negative"):
        analysis.validate_attention(np.asarray([[1.1, -0.1, 0.0]]), expected_rows=1, token_count=3)


def test_run_analysis_writes_complete_outputs(tmp_path: Path) -> None:
    config = SerumGateMinusConfig(
        hidden_size=4,
        latent_dim=2,
        theta_dim=4,
        passage_vocab_size=2,
        passage_pair_vocab_size=4,
        subtype_vocab_size=1,
        subtype_dim=0,
        predictor_hidden_dim=8,
        ha_pooling="lowrank_attention_only",
        ha_attention_dim=4,
        ha_attention_heads=1,
        ha_attention_dropout=0.0,
        na_branch="none",
    )
    model = SerumGateMinusAllAttentionModel(config)
    checkpoint_path = tmp_path / "best_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config.__dict__,
            "epoch": 3,
        },
        checkpoint_path,
    )

    data_csv = tmp_path / "whole.csv"
    pd.DataFrame(
        {
            "Type": ["H3N2", "H3N2", "H3N2"],
            "seq_id_c": ["HA_1", "HA_1", "HA_2"],
            "virusHA": ["A" * 329, "A" * 329, "C" * 329],
            "virusName": ["virus-1", "virus-1-repeat", "virus-2"],
            "virusDate": ["2020", "2021", "2022"],
        }
    ).to_csv(data_csv, index=False)
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()
    torch.save(torch.randn(331, 4), embedding_dir / "matrix_HA_1.pt")
    torch.save(torch.randn(331, 4), embedding_dir / "matrix_HA_2.pt")
    output_dir = tmp_path / "analysis"

    result = analysis.run_analysis(
        argparse.Namespace(
            checkpoint=checkpoint_path,
            data_csv=data_csv,
            embedding_dir=embedding_dir,
            output_dir=output_dir,
            type_filter="H3N2",
            device="cpu",
            batch_size=2,
            token_count=331,
            expected_count=2,
            top_k=5,
        )
    )

    expected_files = {
        "attention_by_sequence.csv",
        "token_summary.csv",
        "top_tokens.csv",
        "mean_attention_profile.png",
        "top_tokens.png",
        "attention_heatmap.png",
        "analysis_config.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    assert all((output_dir / name).stat().st_size > 0 for name in expected_files)
    matrix_frame = pd.read_csv(output_dir / "attention_by_sequence.csv")
    token_columns = [f"token_{idx}" for idx in range(331)]
    assert matrix_frame.shape[0] == 2
    assert np.allclose(matrix_frame[token_columns].sum(axis=1), 1.0, atol=1e-6)
    assert pd.read_csv(output_dir / "token_summary.csv").shape[0] == 331
    assert result["sequence_count"] == 2
    assert result["token_count"] == 331
