from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PATH = ROOT / "experiments" / "serum_gate" / "run_gradient_x_input.py"
SPEC = importlib.util.spec_from_file_location("run_gradient_x_input", PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

from fluprofiler.models.serum_gate_minus_all_attention_model import SerumGateMinusAllAttentionModel  # noqa: E402
from fluprofiler.models.serum_gate_minus_model import SerumGateMinusConfig  # noqa: E402


def save_checkpoint(path: Path) -> None:
    config = SerumGateMinusConfig(
        hidden_size=4, latent_dim=2, theta_dim=4, passage_vocab_size=4,
        passage_pair_vocab_size=16, subtype_vocab_size=1, subtype_dim=0,
        predictor_hidden_dim=8, ha_pooling="lowrank_attention_only",
        ha_attention_dim=4, ha_attention_heads=1, ha_attention_dropout=0.0,
        na_branch="none",
    )
    model = SerumGateMinusAllAttentionModel(config)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.config.__dict__,
        "passage_to_id": {"unknown": 0, "both": 1, "cell": 2, "egg": 3},
        "subtype_to_id": {"constant": 0},
        "epoch": 2,
    }, path)


def test_filter_priority_lengths_and_deduplication(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({
        "serumType": ["H3N2", "H3N2", "H1N1", "H3N2"],
        "Type": ["H1N1", "H1N1", "H3N2", "H3N2"],
        "seq_id_a": ["a1", "a1-copy", "a2", "a3"],
        "seq_id_c": ["c1", "c1-copy", "c2", "c3"],
        "seq_a": ["A" * 329, "A" * 329, "B" * 329, "C" * 328],
        "seq_c": ["D" * 329, "D" * 329, "E" * 329, "F" * 329],
        "serumPassCat": ["<EGG>", "<EGG>", "<CELL>", "<CELL>"],
        "virusPassCat": ["<CELL>", "<CELL>", "<EGG>", "<CELL>"],
    }).to_csv(csv_path, index=False)
    rows = module.collect_samples(csv_path)
    assert rows["seq_id_a"].tolist() == ["a1"]
    assert rows["sample_index"].tolist() == [0]


def test_attribution_and_atomic_outputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    save_checkpoint(checkpoint)
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    for seq_id in ("a1", "c1"):
        torch.save(torch.randn(331, 4), embeddings / f"matrix_{seq_id}.pt")
    data_csv = tmp_path / "data.csv"
    pd.DataFrame({
        "serumType": ["H3N2"], "Type": ["H3N2"],
        "seq_id_a": ["a1"], "seq_id_c": ["c1"],
        "seq_a": ["A" * 329], "seq_c": ["C" * 329],
        "serumPassCat": ["<EGG>"], "virusPassCat": ["<CELL>"], "label": [2.0],
    }).to_csv(data_csv, index=False)
    output = tmp_path / "output"
    result = module.run(argparse.Namespace(
        checkpoint=checkpoint, data_csv=data_csv, embedding_dir=embeddings,
        output_dir=output, type_filter="H3N2", device="cpu", batch_size=1,
        token_count=331, expected_count=1,
    ))
    assert result["sample_count"] == 1
    assert {path.name for path in output.iterdir()} == {
        "attribution_by_sample.csv", "site_summary.csv",
        "attribution_arrays.npz", "analysis_config.json",
    }
    samples = pd.read_csv(output / "attribution_by_sample.csv")
    assert np.allclose(samples["mean"], samples["self_score"] - samples["query_score"], atol=1e-6)
    assert len([name for name in samples if name.startswith("reference_token_")]) == 331
    assert len([name for name in samples if name.startswith("query_token_")]) == 331
    summary = pd.read_csv(output / "site_summary.csv")
    assert summary.shape[0] == 993
    assert summary["side"].drop_duplicates().tolist() == ["reference", "query", "combined"]
