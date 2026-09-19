from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_serum_mutation_set_model import make_batch, tiny_config  # noqa: E402
from fluprofiler.models.serum_mutation_set_query_background_model import (  # noqa: E402
    SerumMutationSetMinusQueryBackgroundModel,
)


class QueryBackgroundTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = SerumMutationSetMinusQueryBackgroundModel(tiny_config(), torch.eye(6))
        self.model.eval()

    def test_model_removes_inherited_mutation_self_attention(self):
        batch = make_batch()
        out = self.model(batch)
        self.assertEqual(len(self.model.mutation_blocks), 0)
        self.assertNotIn("mutation_self_attention", out)

    def test_invalid_query_sites_receive_no_attention(self):
        batch = make_batch()
        batch.query_embedding_mask[0, 0, 1] = 0.0
        out = self.model(batch)
        self.assertEqual(float(out["query_background_attention"][0, 0, :, :, 1].detach().abs().sum()), 0.0)

    def test_cross_attention_parameters_receive_gradients(self):
        batch = make_batch()
        self.model.train()
        self.model(batch)["huber_loss"].backward()
        self.assertFalse(hasattr(self.model, "query_context_gate"))
        self.assertGreater(float(self.model.query_site_projection[1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(self.model.query_background_attention.in_proj_weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
