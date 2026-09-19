from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.fluprofiler.models_v2 import BatchInput, fluProfiler_HA_only_distance_v2


class MeanPooler(nn.Module):
    def forward(self, x, mask=None, save_attention_path=None):
        if mask is None:
            return x.mean(dim=1)
        weights = mask.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-12)


class HAOnlyDistanceTests(unittest.TestCase):
    def test_predicts_scaled_distance_plus_passage_pair_bias(self):
        config = SimpleNamespace(hidden_size=2, matrix_fc_size=None, loss_reduction="mean")
        args = SimpleNamespace(output_mode="regression", ignore_index=-100)
        model = fluProfiler_HA_only_distance_v2(config, args)
        model.matrix_pooler = MeanPooler()
        model.eval()

        with torch.no_grad():
            model.distance_scale_logit.fill_(2.0)
            model.global_bias.fill_(0.5)
            model.passage_pair_bias.weight.zero_()
            model.passage_pair_bias.weight[1 * model.passage_vocab_size + 2, 0] = 0.25

        batch = BatchInput(
            matrices={
                "serum_HA": torch.tensor([[[0.0, 0.0], [2.0, 0.0]]]),
                "virus_HA": torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]),
            },
            matrix_masks={
                "serum_HA": torch.tensor([[1.0, 1.0]]),
                "virus_HA": torch.tensor([[1.0, 1.0]]),
            },
            passage_tokens=torch.tensor([[1, 2]]),
            labels=torch.tensor([3.0]),
        )

        output = model(batch)

        expected_distance = torch.tensor([2.0])
        expected = expected_distance * torch.nn.functional.softplus(torch.tensor(2.0)) + 0.75
        self.assertTrue(torch.allclose(output.extras["distance"], expected_distance))
        self.assertTrue(torch.allclose(output.pred, expected.view(-1)))
        self.assertIn("serum_antigen_vector", output.extras)
        self.assertIn("virus_antigen_vector", output.extras)
        self.assertIn("passage_bias", output.extras)
        self.assertIsNotNone(output.loss)

    def test_optional_name_bias_can_be_enabled_or_disabled_per_batch(self):
        config = SimpleNamespace(hidden_size=2, matrix_fc_size=None, loss_reduction="mean")
        args = SimpleNamespace(output_mode="regression", ignore_index=-100)
        model = fluProfiler_HA_only_distance_v2(
            config,
            args,
            serum_name_vocab_size=3,
            virus_name_vocab_size=3,
        )
        model.matrix_pooler = MeanPooler()
        model.eval()

        with torch.no_grad():
            model.distance_scale_logit.fill_(0.0)
            model.global_bias.zero_()
            model.passage_pair_bias.weight.zero_()
            model.serum_name_bias.weight.zero_()
            model.virus_name_bias.weight.zero_()
            model.serum_name_bias.weight[1, 0] = 0.4
            model.virus_name_bias.weight[2, 0] = 0.6

        batch = BatchInput(
            matrices={
                "serum_HA": torch.tensor([[[0.0, 0.0]]]),
                "virus_HA": torch.tensor([[[0.0, 0.0]]]),
            },
            passage_tokens=torch.tensor([[1, 2]]),
            meta={
                "serum_name_ids": torch.tensor([1]),
                "virus_name_ids": torch.tensor([2]),
            },
        )

        with_name = model(batch)
        without_name = model(
            BatchInput(
                matrices=batch.matrices,
                passage_tokens=batch.passage_tokens,
                meta={
                    "serum_name_ids": torch.tensor([1]),
                    "virus_name_ids": torch.tensor([2]),
                    "use_name_bias": False,
                },
            )
        )

        self.assertTrue(torch.allclose(with_name.pred, torch.tensor([1.0])))
        self.assertTrue(torch.allclose(without_name.pred, torch.tensor([0.0])))
        self.assertIn("name_bias", with_name.extras)


if __name__ == "__main__":
    unittest.main()
