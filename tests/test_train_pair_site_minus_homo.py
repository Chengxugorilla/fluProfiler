from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
TRAIN_SCRIPT = (
    REPO_ROOT / "experiments" / "serum_gate" / "train_pair_site_minus_homo.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "train_pair_site_minus_homo",
        TRAIN_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def aligned_frame(**overrides) -> pd.DataFrame:
    row = {
        "seq_id_a": "HA_REF",
        "seq_id_c": "HA_QUERY",
        "seq_a": "ABC",
        "seq_c": "ADC",
        "serumHA": "AB-C",
        "virusHA": "A-DC",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def aligned_embeddings(*, reference_length: int = 3) -> dict[str, torch.Tensor]:
    return {
        "matrix_HA_REF": torch.ones(reference_length, 6),
        "matrix_HA_QUERY": torch.ones(3, 6),
    }


def test_prepare_pair_site_embeddings_removes_leading_and_trailing_special_tokens():
    module = load_module()
    serum_matrix = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    virus_matrix = torch.arange(30, 60, dtype=torch.float32).reshape(5, 6)
    embeddings = {
        "matrix_HA_REF": serum_matrix,
        "matrix_HA_QUERY": virus_matrix,
    }

    prepared = module.prepare_pair_site_embeddings(
        {"train": aligned_frame()},
        embeddings,
    )

    assert torch.equal(prepared["matrix_HA_REF"], serum_matrix[1:-1])
    assert torch.equal(prepared["matrix_HA_QUERY"], virus_matrix[1:-1])
    assert embeddings["matrix_HA_REF"].shape == torch.Size([5, 6])


def test_validate_pair_site_alignment_accepts_post_embedding_gap_mapping():
    module = load_module()

    max_length = module.validate_pair_site_alignment(
        {"train": aligned_frame()},
        aligned_embeddings(),
    )

    assert max_length == 4


def test_validate_pair_site_alignment_rejects_embedding_length_mismatch():
    module = load_module()

    with pytest.raises(ValueError, match=r"HA_REF.*expected 3 embedding rows, got 2"):
        module.validate_pair_site_alignment(
            {"train": aligned_frame()},
            aligned_embeddings(reference_length=2),
        )


def test_validate_pair_site_alignment_rejects_different_pair_lengths():
    module = load_module()
    frame = aligned_frame(virusHA="ADC")

    with pytest.raises(ValueError, match=r"train row 0.*alignment lengths differ"):
        module.validate_pair_site_alignment({"train": frame}, aligned_embeddings())


def test_validate_pair_site_alignment_rejects_alignment_that_changes_sequence():
    module = load_module()
    frame = aligned_frame(serumHA="AX-C")

    with pytest.raises(ValueError, match=r"serum alignment does not match HA_REF"):
        module.validate_pair_site_alignment({"train": frame}, aligned_embeddings())


def test_validate_pair_site_alignment_rejects_multiple_reference_alignments():
    module = load_module()
    frame = pd.concat(
        [
            aligned_frame(),
            aligned_frame(serumHA="A-BC", virusHA="AD-C"),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match=r"HA_REF.*multiple serumHA alignments"):
        module.validate_pair_site_alignment({"train": frame}, aligned_embeddings())


def test_parse_args_exposes_pair_site_defaults():
    module = load_module()

    args = module.parse_args(
        [
            "--data-dir",
            "data",
            "--embedding-dir",
            "embeddings",
            "--output-dir",
            "out",
        ]
    )

    assert args.site_proj_dim == 128
    assert args.d_model == 256
    assert args.num_layers == 2
    assert args.num_heads == 4
    assert args.transformer_ff_dim == 1024
    assert args.site_attention_dim == 128
    assert args.dropout == pytest.approx(0.1)


def test_parse_args_does_not_expose_legacy_pooling_options():
    module = load_module()

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--data-dir",
                "data",
                "--embedding-dir",
                "embeddings",
                "--output-dir",
                "out",
                "--ha-pooling",
                "mean",
            ]
        )


def test_run_training_smoke_writes_reconstructable_pair_site_checkpoint():
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "splits"
        embedding_dir = root / "embeddings"
        output_dir = root / "out"
        data_dir.mkdir()
        embedding_dir.mkdir()

        rows = [
            {
                "seq_id_a": "ref",
                "seq_id_b": "na_ref",
                "seq_id_c": "query_a",
                "seq_id_d": "na_query_a",
                "seq_a": "ABC",
                "seq_b": "AAAA",
                "seq_c": "ADC",
                "seq_d": "AAAA",
                "serumHA": "AB-C",
                "virusHA": "A-DC",
                "serumPassCat": "cell",
                "virusPassCat": "cell",
                "Type": "H1N1",
                "label": 6.0,
                "homo_label": 8.0,
                "diff_label": 2.0,
            },
            {
                "seq_id_a": "ref",
                "seq_id_b": "na_ref",
                "seq_id_c": "query_b",
                "seq_id_d": "na_query_b",
                "seq_a": "ABC",
                "seq_b": "AAAA",
                "seq_c": "AEC",
                "seq_d": "AAANST",
                "serumHA": "AB-C",
                "virusHA": "A-EC",
                "serumPassCat": "cell",
                "virusPassCat": "egg",
                "Type": "H1N1",
                "label": 5.0,
                "homo_label": 8.0,
                "diff_label": 3.0,
            },
        ]
        pd.DataFrame(rows).to_csv(data_dir / "train.csv", index=False)
        pd.DataFrame(rows[:1]).to_csv(data_dir / "valid.csv", index=False)
        pd.DataFrame(rows[1:]).to_csv(data_dir / "test.csv", index=False)
        for seq_id in ("ref", "query_a", "query_b"):
            torch.save(torch.ones(5, 6), embedding_dir / f"matrix_{seq_id}.pt")

        args = module.parse_args(
            [
                "--data-dir",
                str(data_dir),
                "--embedding-dir",
                str(embedding_dir),
                "--output-dir",
                str(output_dir),
                "--type",
                "H1N1",
                "--epochs",
                "1",
                "--site-proj-dim",
                "4",
                "--d-model",
                "8",
                "--num-layers",
                "2",
                "--num-heads",
                "2",
                "--transformer-ff-dim",
                "16",
                "--site-attention-dim",
                "4",
                "--predictor-hidden-dim",
                "8",
                "--dropout",
                "0",
                "--skip-test-eval",
                "--no-progress",
            ]
        )

        result = module.run_training(args)

        metrics = pd.read_csv(result["paths"]["metrics"])
        assert "valid_diff_mse" in metrics.columns
        assert "valid_homo_mse" in metrics.columns
        assert "valid_query_mse" in metrics.columns
        predictions = pd.read_csv(output_dir / "predictions_valid.csv")
        for column in (
            "label",
            "homo_label",
            "diff_label",
            "mean",
            "self_score",
            "query_score",
        ):
            assert column in predictions.columns

        run_config = json.loads((output_dir / "run_config.json").read_text())
        assert run_config["model"] == "SerumGate-Minus-PairSiteAttn-Homo"
        assert run_config["model_config"]["max_site_length"] == 4
        assert run_config["model_config"]["site_proj_dim"] == 4
        assert run_config["model_config"]["num_layers"] == 2

        checkpoint = torch.load(
            output_dir / "checkpoints" / "best_model.pth",
            map_location="cpu",
            weights_only=False,
        )
        reconstructed = module.PairSiteAttentionMinusModel(checkpoint["model_config"])
        reconstructed.load_state_dict(checkpoint["model_state_dict"])
