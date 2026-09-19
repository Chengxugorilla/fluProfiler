from __future__ import annotations

import importlib.util
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "benchmark_pairwise" / "train_adaboost_fixed_split.py"


def _load_baseline_module():
    spec = importlib.util.spec_from_file_location("train_adaboost_fixed_split", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(
    serum_ha: str = "ABCDE",
    virus_ha: str = "ABCXE",
    serum_passage: str = "<EGG>",
    virus_passage: str = "<CELL>",
    subtype: str = "H3N2",
    label: float = 1.0,
):
    return {
        "seq_id_a": "ha_a",
        "seq_id_c": "ha_c",
        "serumHA": serum_ha,
        "virusHA": virus_ha,
        "serumPassCat": serum_passage,
        "virusPassCat": virus_passage,
        "serumName": "serum",
        "virusName": "virus",
        "Type": subtype,
        "label": label,
    }


class PairwiseAdaBoostBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_baseline_module()

    def test_sequence_mismatch_vector_marks_different_aligned_positions(self):
        vector = self.module.sequence_mismatch_vector("ABCDE", "ABCXE", start=1, end=4)

        self.assertTrue(np.array_equal(vector, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)))

    def test_subtype_is_used_for_separate_models_not_as_meta_feature(self):
        self.assertNotIn("Type", self.module.META_COLUMNS)

    def test_feature_builder_reuses_train_categories_for_eval_splits(self):
        train = pd.DataFrame([_row(serum_passage="<EGG>", virus_passage="<CELL>", subtype="H3N2")])
        test = pd.DataFrame([_row(serum_passage="<UNKNOWN>", virus_passage="<CELL>", subtype="H1N1")])

        builder = self.module.PairwiseFeatureBuilder(sequence_start=0, sequence_end=5)
        train_features = builder.fit_transform(train)
        test_features = builder.transform(test)

        self.assertEqual(train_features.shape, test_features.shape)
        self.assertEqual(train_features.shape[0], 1)
        self.assertGreater(train_features.shape[1], 5)

    def test_feature_builder_handles_empty_eval_frame(self):
        train = pd.DataFrame([_row()])
        empty = train.iloc[:0].copy()

        builder = self.module.PairwiseFeatureBuilder(sequence_start=0, sequence_end=5)
        train_features = builder.fit_transform(train)
        empty_features = builder.transform(empty)

        self.assertEqual(empty_features.shape, (0, train_features.shape[1]))

    def test_load_fixed_split_frames_requires_only_train_and_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame([_row()]).to_csv(data_dir / "train.csv", index=False)
            pd.DataFrame([_row()]).to_csv(data_dir / "test.csv", index=False)

            frames = self.module.load_fixed_split_frames(data_dir)

            self.assertEqual(sorted(frames), ["test", "train"])

    def test_load_fixed_split_frames_can_include_valid_for_train_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame([_row(label=1.0)]).to_csv(data_dir / "train.csv", index=False)
            pd.DataFrame([_row(label=2.0)]).to_csv(data_dir / "valid.csv", index=False)
            pd.DataFrame([_row(label=3.0)]).to_csv(data_dir / "test.csv", index=False)

            frames = self.module.load_fixed_split_frames(data_dir, include_valid=True)

            self.assertEqual(sorted(frames), ["test", "train", "valid"])
            merged_train = pd.concat([frames["train"], frames["valid"]], axis=0, ignore_index=True)
            self.assertEqual(merged_train["label"].tolist(), [1.0, 2.0])

    def test_resolve_split_data_dir_appends_test_season(self):
        root = Path("data/dataset/H1H3_new/splited/v1/H1H3_new/season")

        resolved = self.module.resolve_split_data_dir(root, "26")

        self.assertEqual(resolved, root / "26")
        self.assertEqual(self.module.resolve_split_data_dir(root / "26", "26"), root / "26")

    def test_parse_args_requires_subtypes(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                self.module.parse_args(["--data-dir", "split", "--output-dir", "out"])

    def test_parse_args_accepts_nextflu_style_subtypes(self):
        args = self.module.parse_args(
            [
                "--data-dir",
                "split",
                "--output-dir",
                "out",
                "--subtypes",
                "H1N1",
                "H3N2",
            ]
        )

        self.assertEqual(args.subtypes, ["H1N1", "H3N2"])

    def test_filter_frames_by_subtypes_filters_every_split(self):
        frames = {
            name: pd.DataFrame([_row(subtype="H1N1"), _row(subtype="H3N2")])
            for name in ("train", "valid", "test")
        }

        filtered = self.module.filter_frames_by_subtypes(frames, ["H3N2"])

        self.assertTrue(all(frame["Type"].tolist() == ["H3N2"] for frame in filtered.values()))

    def test_validate_training_subtypes_rejects_missing_effective_train_subtype(self):
        frames = {
            "train": pd.DataFrame([_row(subtype="H1N1")]),
            "valid": pd.DataFrame([_row(subtype="H1N1")]),
            "test": pd.DataFrame([_row(subtype="H3N2")]),
        }

        with self.assertRaisesRegex(ValueError, "H3N2"):
            self.module.validate_training_subtypes(frames, ["H3N2"], merge_valid_into_train=True)

    def test_regression_metrics_include_common_scores(self):
        metrics = self.module.regression_metrics([1.0, 2.0, 3.0], [1.0, 2.5, 2.5])

        self.assertIn("mae", metrics)
        self.assertIn("mse", metrics)
        self.assertIn("pearson", metrics)
        self.assertIn("r2", metrics)
        self.assertAlmostEqual(metrics["mae"], 1.0 / 3.0)

    def test_test_split_also_reports_without_name_predictions_after_training_with_names(self):
        frames = {
            "train": pd.DataFrame(
                [
                    _row(serum_ha="AAAAA", virus_ha="AAAAT", label=1.0),
                    _row(serum_ha="AAAAA", virus_ha="AAATT", label=2.0),
                ]
            ),
            "test": pd.DataFrame([_row(serum_ha="AAAAA", virus_ha="AAATT", label=2.0)]),
        }

        result = self.module.train_subtype_model(
            frames,
            subtype="H3N2",
            sequence_start=0,
            sequence_end=5,
            random_state=100,
            fast=True,
        )

        test_predictions = result["predictions"]["test"]
        self.assertEqual(sorted(result["predictions"]), ["test", "train"])
        self.assertIn("prediction", test_predictions.columns)
        self.assertIn("prediction_without_name", test_predictions.columns)
        self.assertIn("without_name", result["metrics"]["test"])


if __name__ == "__main__":
    unittest.main()
