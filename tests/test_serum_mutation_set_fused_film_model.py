from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fluprofiler.models.serum_mutation_set_fused_film_model import (  # noqa: E402
    SerumMutationSetMinusFusedFiLMModel,
)
from test_serum_mutation_set_model import (  # noqa: E402
    make_batch,
    tiny_config,
)
from fluprofiler.models.serum_mutation_set_model import (  # noqa: E402
    SerumMutationSetMinusModel,
)


class SerumMutationSetFusedFiLMModelTests(unittest.TestCase):
    def test_fusing_unfused_condition_and_film_preserves_predictions(self):
        distance = torch.eye(6)
        torch.manual_seed(11)
        original = SerumMutationSetMinusModel(tiny_config(), distance)
        original.eval()

        fused = SerumMutationSetMinusFusedFiLMModel.from_unfused_model(original)
        fused.eval()

        batch = make_batch()
        original_out = original(batch)
        fused_out = fused(batch)

        self.assertEqual(len(fused.serum_condition_encoder), 2)
        self.assertIsInstance(fused.serum_condition_encoder[1], torch.nn.ReLU)
        torch.testing.assert_close(fused_out["mean"], original_out["mean"])
        torch.testing.assert_close(fused_out["query_score"], original_out["query_score"])
        torch.testing.assert_close(fused_out["self_score"], original_out["self_score"])
        torch.testing.assert_close(fused_out["huber_loss"], original_out["huber_loss"])


class FusedFiLMTrainingEntrypointTests(unittest.TestCase):
    def test_help_exposes_original_training_cli(self):
        script = (
            REPO_ROOT
            / "experiments"
            / "serum_gate"
            / "train_serum_mutation_set_fused_film.py"
        )

        completed = subprocess.run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "fluProfiler",
                "python",
                str(script),
                "--help",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--no-use-passage-pair-feature", completed.stdout)


if __name__ == "__main__":
    unittest.main()
