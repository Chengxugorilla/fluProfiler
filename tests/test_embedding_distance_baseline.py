from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "metric_antigenic" / "evaluate_embedding_distance_baseline.py"


def _load_baseline_module():
    spec = importlib.util.spec_from_file_location("evaluate_embedding_distance_baseline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EmbeddingDistanceBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_baseline_module()

    def test_mean_pooled_euclidean_distance_uses_embedding_rows(self):
        serum = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        virus = torch.tensor([[1.0, 0.0], [1.0, 4.0]])

        distance = self.module.embedding_distance(serum, virus, metric="euclidean")

        self.assertAlmostEqual(distance, 2.0)

    def test_cosine_distance_is_one_minus_cosine_similarity_after_pooling(self):
        serum = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        virus = torch.tensor([[0.0, 2.0], [0.0, 2.0]])

        distance = self.module.embedding_distance(serum, virus, metric="cosine")

        self.assertAlmostEqual(distance, 1.0)

    def test_fit_linear_calibration_maps_train_distances_to_labels(self):
        distances = np.asarray([0.0, 1.0, 2.0], dtype=float)
        labels = np.asarray([1.0, 3.0, 5.0], dtype=float)

        calibration = self.module.fit_linear_calibration(distances, labels)
        prediction = self.module.apply_linear_calibration(distances, calibration)

        self.assertAlmostEqual(calibration["slope"], 2.0)
        self.assertAlmostEqual(calibration["intercept"], 1.0)
        self.assertTrue(np.allclose(prediction, labels))


if __name__ == "__main__":
    unittest.main()
