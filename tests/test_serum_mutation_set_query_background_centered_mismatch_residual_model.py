from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_serum_mutation_set_model import make_batch, tiny_config  # noqa: E402
from fluprofiler.models.serum_mutation_set_query_background_centered_mismatch_residual_model import (  # noqa: E402
    SerumMutationSetMinusQueryBackgroundCenteredMismatchResidualModel,
)


class QueryBackgroundCenteredMismatchResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = SerumMutationSetMinusQueryBackgroundCenteredMismatchResidualModel(
            tiny_config(), torch.eye(6)
        )
        self.model.eval()

    def test_zero_initialization_preserves_query_background_prediction(self):
        out = self.model(make_batch())
        torch.testing.assert_close(out["mean"], out["sequence_mean"])
        self.assertEqual(float(out["mismatch_residual"].detach().abs().sum()), 0.0)

    def test_uniform_position_weight_has_exactly_zero_effect(self):
        with torch.no_grad():
            self.model.mismatch_position_weight.fill_(0.7)
        out = self.model(make_batch())
        torch.testing.assert_close(
            out["mismatch_residual"],
            torch.zeros_like(out["mismatch_residual"]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_centered_residual_receives_gradient_from_training_loss(self):
        self.model.train()
        out = self.model(make_batch())
        out["huber_loss"].backward()
        gradient = self.model.mismatch_position_weight.grad
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertAlmostEqual(float(gradient.sum()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
