from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from src.fluprofiler.models.metric_antigenic_model import (
    MetricAntigenicBatch,
    MetricHAAntigenicModel,
    MetricHAAntigenicModelConfig,
)


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class MeanPooler(nn.Module):
    def forward(self, x, mask=None, save_attention_path=None):
        if mask is None:
            return x.mean(dim=1)
        weights = mask.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-12)


class MetricHAAntigenicModelTests(unittest.TestCase):
    def test_forward_decomposes_prediction_components(self):
        config = MetricHAAntigenicModelConfig(
            hidden_size=2,
            latent_dim=2,
            passage_vocab_size=3,
            passage_pair_vocab_size=9,
            subtype_vocab_size=2,
            na_lambda_max=0.25,
            use_serum_scale=True,
            use_assay_bias=True,
            use_na_glycan_residual=True,
        )
        model = MetricHAAntigenicModel(config)
        model.ha_pooler = MeanPooler()
        model.ha_dropout = nn.Identity()
        model.eval()

        with torch.no_grad():
            model.ha_projection.weight.copy_(torch.eye(2))
            model.ha_projection.bias.zero_()
            model.ha_dim_weight_logit.fill_(inverse_softplus(1.0))
            model.ha_distance_scale_logit.fill_(inverse_softplus(2.0))
            model.rho_intercept.fill_(inverse_softplus(0.5))
            model.rho_from_z.weight.zero_()
            model.rho_passage_effect.weight.zero_()
            model.passage_pair_bias.weight.zero_()
            model.subtype_na_glycan_logit.weight.zero_()

        batch = MetricAntigenicBatch(
            serum_ha=torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
            virus_ha=torch.tensor([[[3.0, 0.0], [3.0, 0.0]]]),
            serum_ha_mask=torch.tensor([[1.0, 1.0]]),
            virus_ha_mask=torch.tensor([[1.0, 1.0]]),
            serum_passage=torch.tensor([1]),
            test_passage=torch.tensor([2]),
            passage_pair=torch.tensor([5]),
            subtype=torch.tensor([1]),
            s_nagly=torch.tensor([0.4]),
            labels=torch.tensor([3.05]),
        )

        out = model(batch)

        self.assertAlmostEqual(out["d_ha"].item(), 6.0, places=5)
        self.assertAlmostEqual(out["rho_ha"].item(), 0.5, places=5)
        self.assertAlmostEqual(out["b_assay"].item(), 0.0, places=5)
        self.assertAlmostEqual(out["lambda_nagly"].item(), 0.125, places=5)
        self.assertAlmostEqual(out["r_na"].item(), 0.05, places=5)
        self.assertAlmostEqual(out["pred"].item(), 3.05, places=5)
        self.assertIn("rho_ha_times_d_ha", out)
        self.assertIsNotNone(out["loss"])

    def test_assay_bias_uses_passage_pair_fixed_effect(self):
        config = MetricHAAntigenicModelConfig(
            hidden_size=2,
            latent_dim=2,
            passage_pair_vocab_size=6,
            use_serum_scale=False,
            use_assay_bias=True,
            use_na_glycan_residual=False,
        )
        model = MetricHAAntigenicModel(config)
        reference = torch.zeros(2)

        with torch.no_grad():
            model.passage_pair_bias.weight.zero_()
            model.passage_pair_bias.weight[4, 0] = 1.25

        out = model.compute_assay_bias(
            MetricAntigenicBatch(
                serum_ha=torch.zeros(2, 1, 2),
                virus_ha=torch.zeros(2, 1, 2),
                passage_pair=torch.tensor([4, 0]),
            ),
            reference,
        )

        self.assertTrue(torch.allclose(out, torch.tensor([1.25, 0.0])))

    def test_ha_distance_is_symmetric_and_zero_for_identical_inputs(self):
        config = MetricHAAntigenicModelConfig(
            hidden_size=2,
            latent_dim=2,
            use_serum_scale=False,
            use_assay_bias=False,
            use_na_glycan_residual=False,
        )
        model = MetricHAAntigenicModel(config)
        model.ha_pooler = MeanPooler()
        model.ha_dropout = nn.Identity()
        model.eval()

        with torch.no_grad():
            model.ha_projection.weight.copy_(torch.eye(2))
            model.ha_projection.bias.zero_()
            model.ha_dim_weight_logit.fill_(inverse_softplus(1.0))
            model.ha_distance_scale_logit.fill_(inverse_softplus(1.0))

        a = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        b = torch.tensor([[[3.0, 4.0], [3.0, 4.0]]])

        d_ab = model.compute_ha_distance(model.encode_ha(a), model.encode_ha(b))
        d_ba = model.compute_ha_distance(model.encode_ha(b), model.encode_ha(a))
        d_aa = model.compute_ha_distance(model.encode_ha(a), model.encode_ha(a))

        self.assertTrue(torch.allclose(d_ab, d_ba))
        self.assertTrue(torch.allclose(d_aa, torch.zeros_like(d_aa), atol=1e-6))

    def test_na_glycan_residual_is_subtype_specific(self):
        config = MetricHAAntigenicModelConfig(
            hidden_size=2,
            latent_dim=2,
            subtype_vocab_size=2,
            na_lambda_max=0.25,
            use_serum_scale=False,
            use_assay_bias=False,
            use_na_glycan_residual=True,
        )
        model = MetricHAAntigenicModel(config)
        model.ha_pooler = MeanPooler()
        model.ha_dropout = nn.Identity()
        model.eval()

        self.assertEqual(model.subtype_na_glycan_logit.weight.numel(), 2)

        with torch.no_grad():
            model.ha_projection.weight.copy_(torch.eye(2))
            model.ha_projection.bias.zero_()
            model.ha_dim_weight_logit.fill_(inverse_softplus(1.0))
            model.ha_distance_scale_logit.fill_(inverse_softplus(1.0))
            model.subtype_na_glycan_logit.weight[0, 0] = -10.0
            model.subtype_na_glycan_logit.weight[1, 0] = 10.0

        batch = {
            "serum_ha": torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]]),
            "virus_ha": torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]]),
            "s_nagly": torch.tensor([1.0, 1.0]),
            "subtype": torch.tensor([0, 1]),
        }

        out = model(batch)

        self.assertLess(out["lambda_nagly"][0].item(), 0.001)
        self.assertGreater(out["lambda_nagly"][1].item(), 0.249)

    def test_optional_name_bias_adds_fixed_effects_and_can_be_disabled(self):
        config = MetricHAAntigenicModelConfig(
            hidden_size=2,
            latent_dim=2,
            use_serum_scale=False,
            use_assay_bias=False,
            use_na_glycan_residual=False,
            serum_name_vocab_size=3,
            virus_name_vocab_size=3,
        )
        model = MetricHAAntigenicModel(config)
        model.ha_pooler = MeanPooler()
        model.ha_dropout = nn.Identity()
        model.eval()

        with torch.no_grad():
            model.ha_projection.weight.copy_(torch.eye(2))
            model.ha_projection.bias.zero_()
            model.ha_dim_weight_logit.fill_(inverse_softplus(1.0))
            model.ha_distance_scale_logit.fill_(inverse_softplus(1.0))
            model.serum_name_bias.weight.zero_()
            model.virus_name_bias.weight.zero_()
            model.serum_name_bias.weight[1, 0] = 0.4
            model.virus_name_bias.weight[2, 0] = 0.6

        batch = MetricAntigenicBatch(
            serum_ha=torch.tensor([[[0.0, 0.0]]]),
            virus_ha=torch.tensor([[[0.0, 0.0]]]),
            serum_name=torch.tensor([1]),
            virus_name=torch.tensor([2]),
        )

        with_name = model(batch)
        batch.use_name_bias = False
        without_name = model(batch)

        self.assertTrue(torch.allclose(with_name["pred"], torch.tensor([1.0])))
        self.assertTrue(torch.allclose(without_name["pred"], torch.tensor([0.0])))
        self.assertIn("name_bias", with_name)

    def test_ha_mismatch_residual_adds_pairwise_sequence_signal(self):
        config = MetricHAAntigenicModelConfig(
            hidden_size=2,
            latent_dim=2,
            use_serum_scale=False,
            use_assay_bias=False,
            use_na_glycan_residual=False,
            ha_mismatch_dim=3,
        )
        model = MetricHAAntigenicModel(config)
        model.ha_pooler = MeanPooler()
        model.ha_dropout = nn.Identity()
        model.eval()

        with torch.no_grad():
            model.ha_projection.weight.copy_(torch.eye(2))
            model.ha_projection.bias.zero_()
            model.ha_dim_weight_logit.fill_(inverse_softplus(1.0))
            model.ha_distance_scale_logit.fill_(inverse_softplus(1.0))
            model.ha_mismatch_residual.weight.copy_(torch.tensor([[0.5, 1.0, -0.25]]))
            model.ha_mismatch_residual.bias.fill_(0.1)

        out = model(
            MetricAntigenicBatch(
                serum_ha=torch.tensor([[[0.0, 0.0]]]),
                virus_ha=torch.tensor([[[0.0, 0.0]]]),
                ha_mismatch=torch.tensor([[1.0, 0.0, 1.0]]),
            )
        )

        self.assertTrue(torch.allclose(out["ha_mismatch_residual"], torch.tensor([0.35])))
        self.assertTrue(torch.allclose(out["pred"], torch.tensor([0.35])))


if __name__ == "__main__":
    unittest.main()
