from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_serum_mutation_set_model import make_batch, tiny_config  # noqa: E402
from fluprofiler.models.serum_mutation_set_query_background_mismatch_residual_model import (  # noqa: E402
    SerumMutationSetMinusQueryBackgroundMismatchResidualModel,
)


class QueryBackgroundMismatchResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = SerumMutationSetMinusQueryBackgroundMismatchResidualModel(
            tiny_config(), torch.eye(6)
        )
        self.model.eval()

    def test_zero_initialization_preserves_query_background_prediction(self):
        out = self.model(make_batch())
        torch.testing.assert_close(out["mean"], out["sequence_mean"])
        self.assertEqual(float(out["mismatch_residual"].detach().abs().sum()), 0.0)

    def test_residual_sums_weights_only_at_valid_mismatch_positions(self):
        batch = make_batch()
        with torch.no_grad():
            self.model.mismatch_position_weight.copy_(
                torch.tensor([0.4, 0.3, -0.2, 0.1, 0.8, -0.7])
            )
        out = self.model(batch)
        expected = torch.tensor([[0.2, 0.3, 0.0]])
        torch.testing.assert_close(out["mismatch_residual"], expected)
        torch.testing.assert_close(out["mean"], out["sequence_mean"] + expected)

    def test_training_loss_and_gradient_use_residual_prediction(self):
        batch = make_batch()
        self.model.train()
        out = self.model(batch)
        out["huber_loss"].backward()
        self.assertGreater(
            float(self.model.mismatch_position_weight.grad.abs().sum()), 0.0
        )
        expected = F.smooth_l1_loss(
            out["mean"], batch.labels, beta=0.5, reduction="none"
        )
        self.assertTrue(torch.isfinite(expected).all())


if __name__ == "__main__":
    unittest.main()
