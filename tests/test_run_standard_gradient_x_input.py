from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "experiments" / "serum_gate"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
PATH = SCRIPT_DIR / "run_standard_gradient_x_input.py"
SPEC = importlib.util.spec_from_file_location("run_standard_gradient_x_input", PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

from fluprofiler.models.serum_gate_minus_model import SerumGateMinusConfig, SerumGateMinusModel  # noqa: E402


def test_loads_standard_checkpoint_strictly(tmp_path: Path) -> None:
    config = SerumGateMinusConfig(
        hidden_size=4, latent_dim=2, theta_dim=4, passage_vocab_size=4,
        passage_pair_vocab_size=16, subtype_vocab_size=1, subtype_dim=0,
        predictor_hidden_dim=8, ha_pooling="lowrank_attention",
        ha_attention_dim=4, ha_attention_heads=1, ha_attention_dropout=0.0,
        na_branch="none",
    )
    expected = SerumGateMinusModel(config)
    path = tmp_path / "standard.pth"
    torch.save({
        "model_state_dict": expected.state_dict(),
        "model_config": expected.config.__dict__,
        "passage_to_id": {"unknown": 0, "both": 1, "cell": 2, "egg": 3},
        "subtype_to_id": {"constant": 0},
    }, path)
    loaded, _ = module.load_standard_model(path, "cpu")
    assert type(loaded) is SerumGateMinusModel
    for name, value in expected.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
