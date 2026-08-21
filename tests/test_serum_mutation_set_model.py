from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fluprofiler.models.serum_mutation_set_model import (  # noqa: E402
    DistanceBiasedSelfAttention,
    MutationSetPool,
    SerumMutationSetBatch,
    SerumMutationSetConfig,
    SerumMutationSetMinusModel,
)


def tiny_config() -> SerumMutationSetConfig:
    return SerumMutationSetConfig(
        hidden_size=4,
        site_dim=4,
        background_dim=8,
        mutation_dim=8,
        position_dim=4,
        amino_acid_dim=3,
        presence_dim=2,
        max_position_embeddings=6,
        amino_acid_vocab_size=23,
        passage_vocab_size=3,
        passage_pair_vocab_size=9,
        passage_dim=2,
        subtype_vocab_size=2,
        subtype_dim=2,
        theta_dim=8,
        mutation_attention_heads=2,
        mutation_attention_layers=1,
        mutation_ffn_dim=16,
        predictor_hidden_dim=12,
        attention_dropout=0.0,
        predictor_dropout=0.0,
        label_weight_thresholds=(2.0, 4.0, 6.0),
        label_weight_values=(1.0, 1.3, 1.8, 2.5),
    )


def make_batch() -> SerumMutationSetBatch:
    torch.manual_seed(7)
    reference_ha = torch.randn(1, 4, 4)
    query_ha = reference_ha[:, None].repeat(1, 3, 1, 1)
    query_ha[0, 0, 0] += 0.7
    query_ha[0, 0, 2] -= 0.4
    query_ha[0, 1, 1] = 0.0

    reference_aa = torch.tensor([[1, 2, 3, 4]])
    query_aa = torch.tensor(
        [[
            [5, 2, 6, 4],
            [1, 22, 3, 4],
            [1, 2, 3, 4],
        ]]
    )
    reference_embedding_mask = torch.ones(1, 4)
    query_embedding_mask = torch.ones(1, 3, 4)
    query_embedding_mask[0, 1, 1] = 0.0

    return SerumMutationSetBatch(
        reference_ha=reference_ha,
        query_ha=query_ha,
        reference_aa=reference_aa,
        query_aa=query_aa,
        reference_aligned_mask=torch.ones(1, 4),
        query_aligned_mask=torch.ones(1, 3, 4),
        reference_embedding_mask=reference_embedding_mask,
        query_embedding_mask=query_embedding_mask,
        serum_passage=torch.tensor([1]),
        query_passage=torch.tensor([[1, 2, 1]]),
        passage_pair=torch.tensor([[4, 5, 4]]),
        subtype=torch.tensor([1]),
        labels=torch.tensor([[1.0, 2.5, 0.0]]),
        query_mask=torch.ones(1, 3),
    )


class DistanceBiasTests(unittest.TestCase):
    def test_missing_distance_and_diagonal_have_exactly_zero_bias(self):
        attention = DistanceBiasedSelfAttention(
            dim=8,
            heads=2,
            dropout=0.0,
            alpha_init=0.1,
            tau_init=4.0,
        )
        distance = torch.tensor(
            [[[0.0, 2.0, float("nan")], [2.0, 0.0, 10.0], [float("nan"), 10.0, 0.0]]]
        )

        bias = attention.distance_bias(distance)

        self.assertTrue(torch.equal(bias[:, :, 0, 0], torch.zeros(1, 2)))
        self.assertTrue(torch.equal(bias[:, :, 0, 2], torch.zeros(1, 2)))
        self.assertTrue(torch.equal(bias[:, :, 2, 0], torch.zeros(1, 2)))

    def test_near_distance_has_greater_bias_than_far_distance(self):
        attention = DistanceBiasedSelfAttention(
            dim=8,
            heads=2,
            dropout=0.0,
            alpha_init=0.1,
            tau_init=4.0,
        )
        distance = torch.tensor(
            [[[0.0, 2.0, 10.0], [2.0, 0.0, 5.0], [10.0, 5.0, 0.0]]]
        )

        bias = attention.distance_bias(distance)

        self.assertTrue(torch.all(bias[:, :, 0, 1] > bias[:, :, 0, 2]))
        self.assertTrue(torch.all(bias[:, :, 0, 2] > 0))


class MutationPoolTests(unittest.TestCase):
    def test_padded_token_values_do_not_change_pooled_representation(self):
        torch.manual_seed(2)
        pool = MutationSetPool(dim=8)
        pool.eval()
        tokens = torch.randn(1, 3, 8)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        changed = tokens.clone()
        changed[:, 2] = 10000.0

        first, first_weights = pool(tokens, mask, torch.tensor([2.0]))
        second, second_weights = pool(changed, mask, torch.tensor([2.0]))

        torch.testing.assert_close(first, second)
        self.assertEqual(float(first_weights[0, 2].detach()), 0.0)
        self.assertEqual(float(second_weights[0, 2].detach()), 0.0)

    def test_count_can_be_removed_from_pool_without_changing_output_shape(self):
        pool = MutationSetPool(dim=8, use_mutation_count=False)
        tokens = torch.randn(2, 3, 8)
        mask = torch.ones(2, 3)

        pooled, weights = pool(tokens, mask, torch.tensor([1.0, 3.0]))

        self.assertEqual(tuple(pooled.shape), (2, 8))
        self.assertEqual(tuple(weights.shape), (2, 3))
        self.assertEqual(pool.projection[0].in_features, 16)

    def test_attention_pool_can_be_removed_while_retaining_mean_and_count(self):
        pool = MutationSetPool(dim=8, use_attention_pool=False)
        tokens = torch.randn(2, 3, 8)
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

        pooled, weights = pool(tokens, mask, torch.tensor([2.0, 3.0]))

        self.assertIsNone(pool.score)
        self.assertEqual(pool.projection[0].in_features, 9)
        self.assertEqual(tuple(pooled.shape), (2, 8))
        torch.testing.assert_close(weights, torch.zeros_like(mask))


class SerumMutationSetModelTests(unittest.TestCase):
    def setUp(self):
        distance = torch.tensor(
            [
                [0.0, 2.0, 8.0, float("nan"), 7.0, 6.0],
                [2.0, 0.0, 3.0, 9.0, 7.0, 6.0],
                [8.0, 3.0, 0.0, 4.0, 7.0, 6.0],
                [float("nan"), 9.0, 4.0, 0.0, 7.0, 6.0],
                [7.0, 7.0, 7.0, 7.0, 0.0, 2.0],
                [6.0, 6.0, 6.0, 6.0, 2.0, 0.0],
            ]
        )
        torch.manual_seed(11)
        self.model = SerumMutationSetMinusModel(tiny_config(), distance)
        self.model.eval()

    def test_zero_init_film_starts_as_identity_conditioning(self):
        config = tiny_config()
        config.zero_init_film = True
        model = SerumMutationSetMinusModel(config, self.model.ha_distance_matrix)

        self.assertTrue(torch.equal(model.film.weight, torch.zeros_like(model.film.weight)))
        self.assertTrue(torch.equal(model.film.bias, torch.zeros_like(model.film.bias)))

        batch = make_batch()
        encoded = model.encode_query_mutations(batch)
        theta, _ = model._condition(
            encoded["z_background"],
            batch.serum_passage,
            batch.subtype,
        )
        gamma, beta = model.film(theta).chunk(2, dim=-1)
        torch.testing.assert_close(gamma, torch.zeros_like(gamma))
        torch.testing.assert_close(beta, torch.zeros_like(beta))

    def test_gamma_only_film_ignores_beta_parameters(self):
        config = tiny_config()
        config.use_film_beta = False
        model = SerumMutationSetMinusModel(config, self.model.ha_distance_matrix)
        model.eval()
        batch = make_batch()

        with torch.no_grad():
            model.film.weight[config.mutation_dim:].fill_(3.0)
            model.film.bias[config.mutation_dim:].fill_(5.0)
        first = model(batch)["mean"]

        with torch.no_grad():
            model.film.weight[config.mutation_dim:].fill_(-7.0)
            model.film.bias[config.mutation_dim:].fill_(-11.0)
        second = model(batch)["mean"]

        torch.testing.assert_close(first, second)

    def test_direct_background_uses_mean_pooled_site_projection(self):
        config = tiny_config()
        config.background_dim = config.site_dim
        config.direct_background = True
        model = SerumMutationSetMinusModel(config, self.model.ha_distance_matrix)
        batch = make_batch()

        projected, background = model._reference_background(batch)

        expected = (
            projected * batch.reference_embedding_mask.unsqueeze(-1)
        ).sum(dim=1) / batch.reference_embedding_mask.sum(dim=1, keepdim=True)
        self.assertIsInstance(model.background_encoder, torch.nn.Identity)
        torch.testing.assert_close(background, expected)

    def test_direct_background_requires_matching_dimensions(self):
        config = tiny_config()
        config.direct_background = True

        with self.assertRaisesRegex(ValueError, "background_dim to equal site_dim"):
            SerumMutationSetMinusModel(config, self.model.ha_distance_matrix)

    def test_low_rank_site_adapter_keeps_site_output_dimension(self):
        config = tiny_config()
        config.site_bottleneck_dim = 2
        model = SerumMutationSetMinusModel(config, self.model.ha_distance_matrix)

        self.assertEqual(model.site_projection[1].in_features, 4)
        self.assertEqual(model.site_projection[1].out_features, 2)
        self.assertEqual(model.site_projection[2].in_features, 2)
        self.assertEqual(model.site_projection[2].out_features, 4)
        encoded = model.encode_query_mutations(make_batch())
        self.assertEqual(tuple(encoded["z_background"].shape), (1, 8))

    def test_transformer_can_be_bypassed_without_changing_shared_initialization(self):
        distance = self.model.ha_distance_matrix
        torch.manual_seed(29)
        original = SerumMutationSetMinusModel(tiny_config(), distance)
        bypass_config = tiny_config()
        bypass_config.bypass_mutation_transformer = True
        torch.manual_seed(29)
        bypassed = SerumMutationSetMinusModel(bypass_config, distance)

        torch.testing.assert_close(
            original.predictor[0].weight,
            bypassed.predictor[0].weight,
        )
        out = bypassed(make_batch())
        self.assertEqual(tuple(out["mean"].shape), (1, 3))
        self.assertTrue(torch.equal(
            out["mutation_self_attention"],
            torch.zeros_like(out["mutation_self_attention"]),
        ))

    def test_background_token_bias_can_be_bypassed_without_changing_shared_initialization(self):
        distance = self.model.ha_distance_matrix
        torch.manual_seed(31)
        original = SerumMutationSetMinusModel(tiny_config(), distance)
        bypass_config = tiny_config()
        bypass_config.use_background_to_mutation = False
        torch.manual_seed(31)
        bypassed = SerumMutationSetMinusModel(bypass_config, distance)

        torch.testing.assert_close(
            original.predictor[0].weight,
            bypassed.predictor[0].weight,
        )
        out = bypassed(make_batch())
        self.assertEqual(tuple(out["mean"].shape), (1, 3))
        self.assertTrue(torch.isfinite(out["huber_loss"]))

    def test_predictor_count_can_be_disabled_without_changing_predictor_shape(self):
        config = tiny_config()
        config.use_predictor_mutation_count = False
        model = SerumMutationSetMinusModel(config, self.model.ha_distance_matrix)
        model.eval()
        z_mutation = torch.randn(1, 2, config.mutation_dim)
        z_mutation[:, 1] = z_mutation[:, 0]
        theta = torch.randn(1, config.theta_dim)
        passage = torch.ones(1, 2, dtype=torch.long)
        pair = torch.ones(1, 2, dtype=torch.long)
        subtype = torch.ones(1, config.subtype_dim)

        scores = model._predict_score(
            z_mutation,
            torch.tensor([[0.0, 5.0]]),
            theta,
            passage,
            pair,
            subtype,
        )

        self.assertEqual(model.predictor[0].in_features, 15)
        torch.testing.assert_close(scores[:, 0], scores[:, 1])

    def test_output_identity_bias_is_additive_and_unknown_ids_are_zero(self):
        distance = self.model.ha_distance_matrix
        torch.manual_seed(37)
        base = SerumMutationSetMinusModel(tiny_config(), distance)
        identity_config = tiny_config()
        identity_config.use_output_identity_bias = True
        identity_config.serum_name_vocab_size = 3
        identity_config.query_virus_vocab_size = 4
        torch.manual_seed(37)
        identity = SerumMutationSetMinusModel(identity_config, distance)
        identity.eval()
        base.eval()
        with torch.no_grad():
            identity.serum_name_bias.weight[1, 0] = 0.3
            identity.query_virus_bias.weight[1, 0] = 0.2
            identity.query_virus_bias.weight[2, 0] = -0.1
        batch = make_batch()
        batch.serum_name = torch.tensor([1])
        batch.query_virus = torch.tensor([[1, 0, 2]])

        base_out = base(batch)
        identity_out = identity(batch)

        expected = torch.tensor([[0.5, 0.3, 0.2]])
        torch.testing.assert_close(identity_out["output_identity_bias"], expected)
        torch.testing.assert_close(identity_out["mean"] - base_out["mean"], expected)

    def test_forward_handles_substitution_gap_and_empty_mutation_set(self):
        batch = make_batch()

        out = self.model(batch)

        self.assertEqual(tuple(out["mean"].shape), (1, 3))
        self.assertEqual(out["mutation_count"].tolist(), [[2.0, 1.0, 0.0]])
        self.assertTrue(torch.isfinite(out["mean"]).all())
        self.assertTrue(torch.isfinite(out["huber_loss"]))

    def test_direction_and_position_change_mutation_representation(self):
        base = make_batch()
        forward_z = self.model.encode_query_mutations(base)["z_mutation"]

        reversed_batch = make_batch()
        reversed_batch.reference_aa = torch.tensor([[5, 2, 6, 4]])
        reversed_batch.query_aa[0, 0] = torch.tensor([1, 2, 3, 4])
        reversed_batch.reference_ha = base.query_ha[:, 0].clone()
        reversed_batch.query_ha[0, 0] = base.reference_ha[0]
        reverse_z = self.model.encode_query_mutations(reversed_batch)["z_mutation"]

        moved_batch = make_batch()
        moved_batch.query_aa[0, 0] = torch.tensor([1, 5, 3, 6])
        moved_batch.query_ha[0, 0] = moved_batch.reference_ha[0]
        moved_batch.query_ha[0, 0, 1] += 0.7
        moved_batch.query_ha[0, 0, 3] -= 0.4
        moved_z = self.model.encode_query_mutations(moved_batch)["z_mutation"]

        self.assertFalse(torch.allclose(forward_z[:, 0], reverse_z[:, 0]))
        self.assertFalse(torch.allclose(forward_z[:, 0], moved_z[:, 0]))

    def test_unknown_x_mutations_are_excluded_from_mutation_attention(self):
        query_x = make_batch()
        query_x.query_aa[0, 0, 0] = 21  # UNKNOWN_ID / X
        query_encoded = self.model.encode_query_mutations(query_x)
        self.assertEqual(query_encoded["mutation_count"][0, 0].item(), 1.0)
        self.assertEqual(query_encoded["mutation_positions"][0, 0, 0].item(), 2)

        reference_x = make_batch()
        reference_x.reference_aa[0, 0] = 21  # UNKNOWN_ID / X
        reference_encoded = self.model.encode_query_mutations(reference_x)
        self.assertEqual(reference_encoded["mutation_count"][0, 0].item(), 1.0)
        self.assertEqual(reference_encoded["mutation_positions"][0, 0, 0].item(), 2)

    def test_backward_reaches_site_token_and_distance_parameters(self):
        self.model.train()
        batch = make_batch()

        out = self.model(batch)
        out["huber_loss"].backward()

        gradients = {
            "site": self.model.site_projection[1].weight.grad,
            "token": self.model.mutation_token_projection[0].weight.grad,
            "distance_alpha": self.model.mutation_blocks[0].attention.raw_alpha.grad,
            "distance_tau": self.model.mutation_blocks[0].attention.raw_tau.grad,
        }
        for name, gradient in gradients.items():
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_task_bias_penalty_adds_squared_unweighted_chunk_bias(self):
        base_config = tiny_config()
        penalized_config = tiny_config()
        penalized_config.task_bias_loss_weight = 0.1
        base = SerumMutationSetMinusModel(base_config, self.model.ha_distance_matrix)
        penalized = SerumMutationSetMinusModel(
            penalized_config,
            self.model.ha_distance_matrix,
        )
        penalized.load_state_dict(base.state_dict())
        base.eval()
        penalized.eval()
        batch = make_batch()

        base_out = base(batch)
        penalized_out = penalized(batch)

        expected_bias = (penalized_out["mean"] - batch.labels).mean()
        torch.testing.assert_close(penalized_out["task_bias"], expected_bias)
        torch.testing.assert_close(
            penalized_out["task_bias_loss"],
            expected_bias.square(),
        )
        torch.testing.assert_close(
            penalized_out["huber_loss"] - base_out["huber_loss"],
            0.1 * expected_bias.square(),
        )


if __name__ == "__main__":
    unittest.main()
