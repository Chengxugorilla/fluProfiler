from __future__ import annotations

import math
import unittest

import torch

from src.fluprofiler.evaluation.ah_distance import compute_archetti_horsfall_distance
from src.fluprofiler.features.na_glycan_features import (
    na_head_glycan_mismatch,
    na_head_glycan_jaccard,
    scan_n_linked_glycosylation,
)
from src.fluprofiler.losses.tail_aware_antigenic_loss import TailAwareAntigenicLoss


class NAGlycanFeatureTests(unittest.TestCase):
    def test_scan_n_linked_glycosylation_returns_zero_based_motif_starts(self):
        motifs = scan_n_linked_glycosylation("ANVTAAANSTNPT")

        self.assertEqual(motifs, {1, 7})

    def test_na_head_jaccard_filters_to_head_positions_and_is_symmetric(self):
        seq_a = "ANVTAAANSTNPT"
        seq_b = "ANVTAAAAAANPT"
        head_positions = {1, 7, 10}

        distance_ab = na_head_glycan_jaccard(seq_a, seq_b, head_positions)
        distance_ba = na_head_glycan_jaccard(seq_b, seq_a, head_positions)

        self.assertAlmostEqual(distance_ab, 0.5)
        self.assertAlmostEqual(distance_ba, distance_ab)

    def test_na_head_glycan_mismatch_is_binary_and_symmetric(self):
        seq_a = "ANVTAAANSTNPT"
        seq_b = "ANVTAAAAAANPT"
        head_positions = {1, 7, 10}

        mismatch_ab = na_head_glycan_mismatch(seq_a, seq_b, head_positions)
        mismatch_ba = na_head_glycan_mismatch(seq_b, seq_a, head_positions)
        identical = na_head_glycan_mismatch(seq_a, seq_a, head_positions)

        self.assertEqual(mismatch_ab, 1.0)
        self.assertEqual(mismatch_ba, 1.0)
        self.assertEqual(identical, 0.0)


class ArchettiHorsfallTests(unittest.TestCase):
    def test_compute_archetti_horsfall_distance_for_reciprocal_titers(self):
        distance = compute_archetti_horsfall_distance(640, 160, 320, 80)

        self.assertAlmostEqual(distance, 2.0)

    def test_compute_archetti_horsfall_distance_skips_censored_titers(self):
        distance = compute_archetti_horsfall_distance("<40", 160, 320, 80)

        self.assertIsNone(distance)


class TailAwareLossTests(unittest.TestCase):
    def test_tail_loss_penalizes_distribution_shrinkage_and_tail_errors(self):
        loss_fn = TailAwareAntigenicLoss(
            smoothl1_beta=0.5,
            q_low=1.0,
            q_high=4.0,
            lambda_tail=2.0,
            lambda_dist=0.25,
            lambda_reg=0.0,
            lambda_na_prior=0.0,
        )
        pred = torch.tensor([1.0, 1.0, 3.0, 5.0])
        target = torch.tensor([0.0, 1.0, 4.0, 5.0])

        out = loss_fn(pred, target)

        expected_hi = torch.nn.functional.smooth_l1_loss(pred, target, beta=0.5)
        tail_errors = torch.tensor([1.0, 0.0, 1.0, 0.0])
        expected_tail = tail_errors.mean()
        expected_dist = abs(torch.std(pred, unbiased=False) - torch.std(target, unbiased=False))
        expected = expected_hi + 2.0 * expected_tail + 0.25 * expected_dist

        self.assertTrue(torch.allclose(out["hi"], expected_hi))
        self.assertTrue(torch.allclose(out["tail"], expected_tail))
        self.assertTrue(torch.allclose(out["dist"], expected_dist))
        self.assertTrue(torch.allclose(out["loss"], expected))

    def test_ah_loss_is_optional_supervision_on_ha_distance(self):
        loss_fn = TailAwareAntigenicLoss(
            smoothl1_beta=0.5,
            q_low=-10.0,
            q_high=10.0,
            lambda_tail=0.0,
            lambda_dist=0.0,
            lambda_ah=0.5,
            lambda_reg=0.0,
            lambda_na_prior=0.0,
        )

        out = loss_fn(
            pred=torch.tensor([1.0, 1.0]),
            target=torch.tensor([1.0, 1.0]),
            d_ha=torch.tensor([1.0, 3.0]),
            d_ah=torch.tensor([2.0, 1.0]),
        )

        expected_ah = torch.tensor(2.5)
        self.assertTrue(torch.allclose(out["ah"], expected_ah))
        self.assertTrue(torch.allclose(out["loss"], 0.5 * expected_ah))


if __name__ == "__main__":
    unittest.main()
