from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "serum_gate" / "infer_minus_checkpoint.py"


def load_module():
    spec = importlib.util.spec_from_file_location("infer_minus_checkpoint", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_to_requested_checkpoint():
    module = load_module()

    args = module.parse_args([])

    assert args.checkpoint == module.DEFAULT_CHECKPOINT
    assert str(args.checkpoint).endswith(
        "results/H1H3_HA1/SerumGate-Minus-latent8/serum/subtype/H3N2/checkpoints/best_model.pth"
    )


def test_prepare_inference_frame_accepts_unlabeled_rows():
    module = load_module()
    frame = pd.DataFrame(
        [
            {
                "seq_id_a": "HA_ref",
                "seq_id_b": "NA_ref",
                "seq_id_c": "HA_query",
                "seq_id_d": "NA_query",
                "seq_b": "AAAA",
                "seq_d": "AAAA",
                "serumPassCat": "<EGG>",
                "virusPassCat": "<CELL>",
                "Type": "H3N2",
            }
        ]
    )

    prepared, has_labels = module.prepare_inference_frame(frame, source="memory", type_filter="")

    assert has_labels is False
    assert prepared["label"].tolist() == [0.0]
