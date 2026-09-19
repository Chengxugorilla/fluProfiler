from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fluprofiler.models.serum_mutation_set_model import (  # noqa: E402
    SerumMutationSetBatch,
    SerumMutationSetConfig,
)
from fluprofiler.models.serum_mutation_set_name_residual_model import (  # noqa: E402
    SerumMutationSetMinusNameResidualModel,
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
    )


def make_batch() -> SerumMutationSetBatch:
    torch.manual_seed(7)
    reference_ha = torch.randn(1, 4, 4)
    batch = SerumMutationSetBatch(
        reference_ha=reference_ha,
        query_ha=reference_ha[:, None].repeat(1, 2, 1, 1),
        reference_aa=torch.tensor([[1, 2, 3, 4]]),
        query_aa=torch.tensor([[[5, 2, 3, 4], [1, 2, 3, 4]]]),
        reference_aligned_mask=torch.ones(1, 4),
        query_aligned_mask=torch.ones(1, 2, 4),
        reference_embedding_mask=torch.ones(1, 4),
        query_embedding_mask=torch.ones(1, 2, 4),
        serum_passage=torch.tensor([1]),
        query_passage=torch.tensor([[1, 2]]),
        passage_pair=torch.tensor([[4, 5]]),
        subtype=torch.tensor([1]),
        labels=torch.tensor([[1.0, 2.5]]),
        query_mask=torch.ones(1, 2),
    )
    batch.serum_name_id = torch.tensor([1])
    batch.virus_name_id = torch.tensor([[1, 2]])
    return batch


class NameResidualModelTests(unittest.TestCase):
    def setUp(self):
        distance = torch.eye(6)
        self.model = SerumMutationSetMinusNameResidualModel(
            tiny_config(),
            distance,
            serum_name_vocab_size=3,
            virus_name_vocab_size=4,
            serum_name_l2=1e-3,
            virus_name_l2=1e-2,
        )
        self.model.eval()

    def test_zero_initialization_keeps_sequence_prediction_and_unk_zero(self):
        batch = make_batch()
        out = self.model(batch)

        torch.testing.assert_close(out["mean"], out["sequence_mean"])
        self.assertEqual(float(out["serum_name_effect"].abs().sum()), 0.0)
        self.assertEqual(float(out["virus_name_effect"].abs().sum()), 0.0)
        self.assertEqual(float(self.model.serum_name_values.weight[0, 0]), 0.0)
        self.assertEqual(float(self.model.virus_name_values.weight[0, 0]), 0.0)

    def test_name_effects_are_added_only_at_final_prediction_and_regularized(self):
        batch = make_batch()
        with torch.no_grad():
            self.model.serum_name_values.weight[1, 0] = 0.4
            self.model.virus_name_values.weight[1, 0] = -0.1
            self.model.virus_name_values.weight[2, 0] = 0.2
        out = self.model(batch)

        torch.testing.assert_close(
            out["mean"],
            out["sequence_mean"] + torch.tensor([[0.3, 0.6]]),
        )
        self.assertAlmostEqual(float(out["name_regularization"]), 0.00066, places=6)
        self.assertGreater(float(out["huber_loss"]), float(out["prediction_loss"]))

    def test_unknown_name_ids_have_fixed_zero_effect(self):
        batch = make_batch()
        batch.serum_name_id = torch.tensor([0])
        batch.virus_name_id = torch.tensor([[0, 0]])
        with torch.no_grad():
            self.model.serum_name_values.weight[0, 0] = 7.0
            self.model.virus_name_values.weight[0, 0] = -5.0
        out = self.model(batch)

        self.assertEqual(float(out["serum_name_effect"].abs().sum()), 0.0)
        self.assertEqual(float(out["virus_name_effect"].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
