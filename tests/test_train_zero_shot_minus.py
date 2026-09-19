from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "serum_gate" / "train_zero_shot_minus.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("train_zero_shot_minus", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SubtypeMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_training_module()

    def test_subtype_regression_metrics_splits_prediction_frame_by_type(self):
        frame = pd.DataFrame(
            [
                {"Type": "H1N1", "task_key": "h1", "label": 1.0, "mean": 1.0, "log_var": 0.0},
                {"Type": "H1N1", "task_key": "h1", "label": 3.0, "mean": 2.0, "log_var": 0.0},
                {"Type": "H3N2", "task_key": "h3", "label": 1.0, "mean": 3.0, "log_var": 0.0},
                {"Type": "H3N2", "task_key": "h3", "label": 3.0, "mean": 3.0, "log_var": 0.0},
            ]
        )

        metrics = self.module.subtype_regression_metrics(frame)

        self.assertEqual(set(metrics), {"h1n1", "h3n2"})
        self.assertAlmostEqual(metrics["h1n1"]["pooled_mae"], 0.5)
        self.assertAlmostEqual(metrics["h3n2"]["pooled_mae"], 1.0)

    def test_subtype_regression_metrics_keeps_empty_subtype_metrics_schema(self):
        frame = pd.DataFrame(
            [{"Type": "H1N1", "task_key": "h1", "label": 1.0, "mean": 1.0, "log_var": 0.0}]
        )

        metrics = self.module.subtype_regression_metrics(frame)

        self.assertEqual(metrics["h3n2"]["pooled_mae"], 0.0)
        self.assertIn("coverage_95", metrics["h3n2"])


    def test_subtype_metrics_are_enabled_only_for_all_subtypes(self):
        self.assertTrue(self.module.subtype_metrics_enabled(""))
        self.assertFalse(self.module.subtype_metrics_enabled("H1N1"))
        self.assertFalse(self.module.subtype_metrics_enabled("H3N2"))

    def test_flatten_epoch_metrics_expands_subtype_metric_columns(self):
        row = self.module.flatten_epoch_metrics(
            {
                "epoch": 1,
                "test": {
                    "h1n1": {"pooled_mae": 1.23456},
                    "h3n2": {"pooled_mae": 2.34567},
                },
            }
        )

        self.assertEqual(row["test_h1n1_pooled_mae"], 1.2346)
        self.assertEqual(row["test_h3n2_pooled_mae"], 2.3457)


if __name__ == "__main__":
    unittest.main()
