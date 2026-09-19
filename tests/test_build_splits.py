from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "tools" / "build_splits.py"
DATASET_SPLITS_SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_dataset_splits.py"


def _load_split_module():
    spec = importlib.util.spec_from_file_location("build_splits", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_dataset_splits_module():
    spec = importlib.util.spec_from_file_location("prepare_dataset_splits", DATASET_SPLITS_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildSplitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_split_module()

    def test_aggregates_with_separate_pre_split_and_train_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            raw_dir = tmpdir / "raw"
            raw_dir.mkdir()
            splits_root = tmpdir / "splits"

            pd.DataFrame(
                [
                    {
                        "seq_id_a": "serum_1",
                        "seq_id_b": "na_1",
                        "seq_id_c": "virus_1",
                        "seq_id_d": "na_2",
                        "serumPassCat": "<EGG>",
                        "virusPassCat": "<CELL>",
                        "label": 1.0,
                        "source_note": "pre duplicate first",
                    },
                    {
                        "seq_id_a": "serum_1",
                        "seq_id_b": "na_1",
                        "seq_id_c": "virus_1",
                        "seq_id_d": "na_2",
                        "serumPassCat": "<EGG>",
                        "virusPassCat": "<CELL>",
                        "label": 9.0,
                        "source_note": "pre duplicate second",
                    },
                    {
                        "seq_id_a": "serum_1",
                        "seq_id_b": "na_9",
                        "seq_id_c": "virus_1",
                        "seq_id_d": "na_9",
                        "serumPassCat": "<EGG>",
                        "virusPassCat": "<CELL>",
                        "label": 7.0,
                        "source_note": "train duplicate broader subset",
                    },
                    {
                        "seq_id_a": "serum_1",
                        "seq_id_b": "na_1",
                        "seq_id_c": "virus_1",
                        "seq_id_d": "na_2",
                        "serumPassCat": "<EGG>",
                        "virusPassCat": "<NONE>",
                        "label": 2.0,
                        "source_note": "different passage",
                    },
                    {
                        "seq_id_a": "serum_2",
                        "seq_id_b": "na_3",
                        "seq_id_c": "virus_2",
                        "seq_id_d": "na_4",
                        "serumPassCat": "<CELL>",
                        "virusPassCat": "<EGG>",
                        "label": 3.0,
                        "source_note": "unique",
                    },
                ]
            ).to_csv(raw_dir / "source.csv", index=False)

            argv = [
                "build_splits.py",
                "--raw-version-dir",
                str(raw_dir),
                "--dataset-name",
                "unit_dataset",
                "--splits-root",
                str(splits_root),
                "--protocol-version",
                "v1",
                "--split-id",
                "unit_split",
                "--seed",
                "0",
                "--test-ratio",
                "0.2",
                "--valid-ratio",
                "0.2",
                "--pre-split-agg-cols",
                "seq_id_a,seq_id_b,seq_id_c,seq_id_d",
                "--train-agg-cols",
                "seq_id_a,seq_id_c",
                "--strain-col",
                "seq_id_c",
                "--serum-col",
                "seq_id_a",
                "--split-modes",
                "titer",
            ]
            with patch.object(sys, "argv", argv):
                self.module.main()

            out_dir = splits_root / "v1" / "raw" / "unit_split" / "titer"
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            train = pd.read_csv(out_dir / "train.csv")
            valid = pd.read_csv(out_dir / "valid.csv")
            test = pd.read_csv(out_dir / "test.csv")
            split_frames = [frame for frame in (train, valid, test) if not frame.empty]
            combined = pd.concat(split_frames, ignore_index=True)

            self.assertEqual(
                manifest["pre_split_aggregation"]["subset"],
                ["serumPassCat", "virusPassCat", "seq_id_a", "seq_id_b", "seq_id_c", "seq_id_d"],
            )
            self.assertEqual(manifest["pre_split_aggregation"]["before"], 5)
            self.assertEqual(manifest["pre_split_aggregation"]["after"], 4)
            self.assertEqual(
                manifest["train_pool_aggregation"]["subset"],
                ["serumPassCat", "virusPassCat", "seq_id_a", "seq_id_c"],
            )
            self.assertEqual(manifest["counts"]["all"], len(combined))
            self.assertEqual(
                combined[
                    (combined["seq_id_a"] == "serum_1")
                    & (combined["seq_id_c"] == "virus_1")
                    & (combined["serumPassCat"] == "<EGG>")
                    & (combined["virusPassCat"] == "<CELL>")
                ].shape[0],
                1,
            )
            merged_label = combined.loc[
                (combined["seq_id_a"] == "serum_1")
                & (combined["seq_id_c"] == "virus_1")
                & (combined["serumPassCat"] == "<EGG>")
                & (combined["virusPassCat"] == "<CELL>"),
                "label",
            ].iloc[0]
            self.assertEqual(merged_label, 6.0)
            self.assertIn("<NONE>", combined["virusPassCat"].tolist())

    def test_dataset_scoped_output_omits_dataset_version_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            raw_dir = tmpdir / "raw"
            raw_dir.mkdir()
            splits_root = tmpdir / "dataset" / "splited"
            pd.DataFrame(
                [
                    {
                        "seq_id_a": "serum_1",
                        "seq_id_b": "na_1",
                        "seq_id_c": "virus_1",
                        "seq_id_d": "na_2",
                        "serumPassCat": "<EGG>",
                        "virusPassCat": "<CELL>",
                        "label": 1.0,
                    },
                    {
                        "seq_id_a": "serum_2",
                        "seq_id_b": "na_3",
                        "seq_id_c": "virus_2",
                        "seq_id_d": "na_4",
                        "serumPassCat": "<CELL>",
                        "virusPassCat": "<EGG>",
                        "label": 2.0,
                    },
                    {
                        "seq_id_a": "serum_3",
                        "seq_id_b": "na_5",
                        "seq_id_c": "virus_3",
                        "seq_id_d": "na_6",
                        "serumPassCat": "<NONE>",
                        "virusPassCat": "<NONE>",
                        "label": 3.0,
                    },
                ]
            ).to_csv(raw_dir / "source.csv", index=False)

            argv = [
                "build_splits.py",
                "--raw-version-dir",
                str(raw_dir),
                "--dataset-name",
                "unit_dataset",
                "--dataset-version-id",
                "unit_version",
                "--splits-root",
                str(splits_root),
                "--protocol-version",
                "v1",
                "--split-id",
                "unit_split",
                "--seed",
                "0",
                "--test-ratio",
                "0.2",
                "--valid-ratio",
                "0.2",
                "--strain-col",
                "seq_id_c",
                "--serum-col",
                "seq_id_a",
                "--split-modes",
                "titer",
                "--dataset-scoped-output",
            ]
            with patch.object(sys, "argv", argv):
                self.module.main()

            self.assertTrue((splits_root / "v1" / "unit_split" / "titer" / "manifest.json").is_file())
            self.assertFalse((splits_root / "v1" / "unit_version").exists())

    def test_dataset_splits_entrypoint_uses_processed_and_splited_dirs(self):
        module = _load_dataset_splits_module()
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "data" / "dataset" / "unit_dataset"
            processed_dir = dataset_dir / "processed"
            processed_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "seq_id_a": "serum_1",
                        "seq_id_b": "na_1",
                        "seq_id_c": "virus_1",
                        "seq_id_d": "na_2",
                        "serumPassCat": "<EGG>",
                        "virusPassCat": "<CELL>",
                        "label": 1.0,
                    },
                    {
                        "seq_id_a": "serum_2",
                        "seq_id_b": "na_3",
                        "seq_id_c": "virus_2",
                        "seq_id_d": "na_4",
                        "serumPassCat": "<CELL>",
                        "virusPassCat": "<EGG>",
                        "label": 2.0,
                    },
                    {
                        "seq_id_a": "serum_3",
                        "seq_id_b": "na_5",
                        "seq_id_c": "virus_3",
                        "seq_id_d": "na_6",
                        "serumPassCat": "<NONE>",
                        "virusPassCat": "<NONE>",
                        "label": 3.0,
                    },
                ]
            ).to_csv(processed_dir / "source.csv", index=False)

            argv = [
                "prepare_dataset_splits.py",
                "--dataset-dir",
                str(dataset_dir),
                "--split-id",
                "unit_split",
                "--split-modes",
                "titer",
            ]
            with patch.object(sys, "argv", argv):
                module.main()

            out_dir = dataset_dir / "splited" / "v1" / "unit_split" / "titer"
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue((out_dir / "train.csv").is_file())
            self.assertEqual(manifest["dataset_name"], "unit_dataset")
            self.assertEqual(
                manifest["source"]["input_csv"],
                str((processed_dir / "source.csv").resolve()),
            )

    def test_season_mode_writes_requested_test_seasons_under_season_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            raw_dir = tmpdir / "raw"
            raw_dir.mkdir()
            splits_root = tmpdir / "splits"

            rows = []
            season_row_counts = {24: 12, 25: 12, 26: 12, 27: 1, 28: 3}
            for season in range(24, 29):
                for idx in range(season_row_counts[season]):
                    rows.append(
                        {
                            "seq_id_a": f"serum_{season}_{idx}",
                            "seq_id_b": f"na_serum_{season}_{idx}",
                            "seq_id_c": f"virus_{season}_{idx}",
                            "seq_id_d": f"na_virus_{season}_{idx}",
                            "serumPassCat": "<EGG>",
                            "virusPassCat": "<CELL>",
                            "label": float(season + idx),
                            "sheet": f"{season}-{idx + 1}",
                        }
                    )
            # Same pair in a different season must not be aggregated away before
            # season assignment.
            rows.append(
                {
                    "seq_id_a": "shared_serum",
                    "seq_id_b": "shared_na_serum",
                    "seq_id_c": "shared_virus",
                    "seq_id_d": "shared_na_virus",
                    "serumPassCat": "<EGG>",
                    "virusPassCat": "<CELL>",
                    "label": 1.0,
                    "sheet": "25-9",
                }
            )
            rows.append(
                {
                    "seq_id_a": "shared_serum",
                    "seq_id_b": "shared_na_serum",
                    "seq_id_c": "shared_virus",
                    "seq_id_d": "shared_na_virus",
                    "serumPassCat": "<EGG>",
                    "virusPassCat": "<CELL>",
                    "label": 9.0,
                    "sheet": "26-9",
                }
            )
            pd.DataFrame(rows).to_csv(raw_dir / "source.csv", index=False)

            argv = [
                "build_splits.py",
                "--raw-version-dir",
                str(raw_dir),
                "--dataset-name",
                "unit_dataset",
                "--dataset-version-id",
                "unit_version",
                "--splits-root",
                str(splits_root),
                "--protocol-version",
                "v1",
                "--split-id",
                "unit_split",
                "--seed",
                "0",
                "--test-ratio",
                "0.1",
                "--valid-ratio",
                "0.1",
                "--pre-split-agg-cols",
                "seq_id_a,seq_id_b,seq_id_c,seq_id_d",
                "--train-agg-cols",
                "seq_id_a,seq_id_c",
                "--strain-col",
                "seq_id_c",
                "--serum-col",
                "seq_id_a",
                "--split-modes",
                "season",
                "--test-seasons",
                "26,28",
            ]
            with patch.object(sys, "argv", argv):
                self.module.main()

            season_root = splits_root / "v1" / "unit_version" / "unit_split" / "season"
            self.assertTrue((season_root / "26" / "manifest.json").is_file())
            self.assertTrue((season_root / "28" / "manifest.json").is_file())
            self.assertFalse((season_root / "27").exists())

            split_26 = season_root / "26"
            manifest_26 = json.loads((split_26 / "manifest.json").read_text(encoding="utf-8"))
            train_26 = pd.read_csv(split_26 / "train.csv")
            valid_26 = pd.read_csv(split_26 / "valid.csv")
            test_26 = pd.read_csv(split_26 / "test.csv")

            self.assertEqual(manifest_26["mode"], "season")
            self.assertEqual(manifest_26["test_season"], "26")
            self.assertEqual(manifest_26["season_col"], "sheet")
            self.assertEqual(manifest_26["season_key_rule"], "prefix_before_dash")
            self.assertEqual(manifest_26["valid_split_strategy"], "previous_season")
            self.assertEqual(manifest_26["train_seasons"], ["24"])
            self.assertEqual(manifest_26["valid_seasons"], ["25"])
            self.assertEqual(set(train_26["sheet"].str.split("-", n=1).str[0]), {"24"})
            self.assertEqual(set(valid_26["sheet"].str.split("-", n=1).str[0]), {"25"})
            self.assertEqual(set(test_26["sheet"].str.split("-", n=1).str[0]), {"26"})
            self.assertEqual(
                set(pd.concat([train_26, valid_26])["sheet"].str.split("-", n=1).str[0]),
                {"24", "25"},
            )
            self.assertEqual(manifest_26["unused_seasons"], ["27", "28"])
            self.assertEqual(
                manifest_26["pre_split_aggregation"]["subset"],
                ["_season_key", "serumPassCat", "virusPassCat", "seq_id_a", "seq_id_b", "seq_id_c", "seq_id_d"],
            )
            self.assertIn(9.0, test_26["label"].tolist())

            split_28 = season_root / "28"
            manifest_28 = json.loads((split_28 / "manifest.json").read_text(encoding="utf-8"))
            train_28 = pd.read_csv(split_28 / "train.csv")
            valid_28 = pd.read_csv(split_28 / "valid.csv")
            test_28 = pd.read_csv(split_28 / "test.csv")
            self.assertEqual(manifest_28["test_season"], "28")
            self.assertEqual(manifest_28["train_seasons"], ["24", "25", "26"])
            self.assertEqual(manifest_28["valid_seasons"], ["27"])
            self.assertEqual(set(train_28["sheet"].str.split("-", n=1).str[0]), {"24", "25", "26"})
            self.assertEqual(set(valid_28["sheet"].str.split("-", n=1).str[0]), {"27"})
            self.assertEqual(set(test_28["sheet"].str.split("-", n=1).str[0]), {"28"})


if __name__ == "__main__":
    unittest.main()
