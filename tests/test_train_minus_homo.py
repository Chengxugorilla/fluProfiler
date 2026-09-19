from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "serum_gate" / "train_minus_homo.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_minus_homo", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainMinusHomoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_prediction_metrics_include_distance_homologous_and_query_mse(self):
        frame = pd.DataFrame(
            [
                {
                    "task_key": "serum_a",
                    "label": 6.0,
                    "homo_label": 10.0,
                    "diff_label": 4.0,
                    "mean": 3.0,
                    "self_score": 9.0,
                    "query_score": 6.5,
                    "log_var": 0.0,
                },
                {
                    "task_key": "serum_a",
                    "label": 8.0,
                    "homo_label": 10.0,
                    "diff_label": 2.0,
                    "mean": 2.0,
                    "self_score": 11.0,
                    "query_score": 7.0,
                    "log_var": 0.0,
                },
            ]
        )

        metrics = self.module.serum_homo_regression_metrics(frame)

        self.assertAlmostEqual(metrics["diff_mse"], 0.5)
        self.assertAlmostEqual(metrics["homo_mse"], 1.0)
        self.assertAlmostEqual(metrics["query_mse"], 0.625)
        self.assertAlmostEqual(metrics["pooled_mse"], metrics["diff_mse"])

    def test_parse_args_defaults_to_new_label_columns(self):
        args = self.module.parse_args(
            [
                "--data-dir",
                "data",
                "--embedding-dir",
                "embeddings",
                "--output-dir",
                "out",
            ]
        )

        self.assertEqual(args.query_label_col, "label")
        self.assertEqual(args.homo_label_col, "homo_label")
        self.assertEqual(args.distance_label_col, "diff_label")
        self.assertEqual(args.best_metric, "diff_mse")

    def test_run_training_smoke_with_three_label_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "splits"
            embedding_dir = root / "embeddings"
            output_dir = root / "out"
            data_dir.mkdir()
            embedding_dir.mkdir()

            rows = [
                {
                    "seq_id_a": "ref",
                    "seq_id_b": "na_ref",
                    "seq_id_c": "query_a",
                    "seq_id_d": "na_query_a",
                    "seq_b": "AAAA",
                    "seq_d": "AAAA",
                    "serumPassCat": "cell",
                    "virusPassCat": "cell",
                    "Type": "H1N1",
                    "label": 6.0,
                    "homo_label": 8.0,
                    "diff_label": 2.0,
                },
                {
                    "seq_id_a": "ref",
                    "seq_id_b": "na_ref",
                    "seq_id_c": "query_b",
                    "seq_id_d": "na_query_b",
                    "seq_b": "AAAA",
                    "seq_d": "AAANST",
                    "serumPassCat": "cell",
                    "virusPassCat": "egg",
                    "Type": "H1N1",
                    "label": 5.0,
                    "homo_label": 8.0,
                    "diff_label": 3.0,
                },
            ]
            pd.DataFrame(rows).to_csv(data_dir / "train.csv", index=False)
            pd.DataFrame(rows[:1]).to_csv(data_dir / "valid.csv", index=False)
            pd.DataFrame(rows[1:]).to_csv(data_dir / "test.csv", index=False)

            for seq_id in ("ref", "query_a", "query_b"):
                torch.save(torch.ones(3, 4), embedding_dir / f"matrix_{seq_id}.pt")

            args = self.module.parse_args(
                [
                    "--data-dir",
                    str(data_dir),
                    "--embedding-dir",
                    str(embedding_dir),
                    "--output-dir",
                    str(output_dir),
                    "--type",
                    "H1N1",
                    "--epochs",
                    "1",
                    "--latent-dim",
                    "2",
                    "--theta-dim",
                    "4",
                    "--predictor-hidden-dim",
                    "4",
                    "--distance-hidden-dim",
                    "4",
                    "--calibration-hidden-dim",
                    "4",
                    "--residual-hidden-dim",
                    "4",
                    "--ha-pooling",
                    "mean",
                    "--skip-test-eval",
                    "--no-progress",
                ]
            )

            result = self.module.run_training(args)

            metrics = pd.read_csv(result["paths"]["metrics"])
            self.assertIn("valid_diff_mse", metrics.columns)
            self.assertIn("valid_homo_mse", metrics.columns)
            self.assertIn("valid_query_mse", metrics.columns)

            predictions = pd.read_csv(output_dir / "predictions_valid.csv")
            for column in ("label", "homo_label", "diff_label", "mean", "self_score", "query_score"):
                self.assertIn(column, predictions.columns)


if __name__ == "__main__":
    unittest.main()
