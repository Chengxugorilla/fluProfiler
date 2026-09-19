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
SCRIPT_PATH = REPO_ROOT / "src" / "fluprofiler" / "dataset" / "processed2splited.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("processed2splited", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _overlap_count(a: pd.DataFrame, b: pd.DataFrame, col: str) -> int:
    return len(set(a[col].astype(str)) & set(b[col].astype(str)))


class Processed2SplitedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_builds_titer_strain_serum_splits_without_protocol_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "H1H3_HA1"
            processed_dir = dataset_dir / "processed"
            processed_dir.mkdir(parents=True)

            rows = []
            for idx in range(12):
                rows.append(
                    {
                        "seq_a": f"serum_{idx % 4}",
                        "seq_c": f"virus_{idx % 6}",
                        "serumPassCat": "<NONE>",
                        "virusPassCat": "<EGG>",
                        "label": float(idx),
                        "Type": "H1N1" if idx % 2 == 0 else "H3N2",
                        "sheet": f"{idx // 3 + 1}-x",
                    }
                )
            pd.DataFrame(rows).to_csv(processed_dir / "source.csv", index=False)
            (processed_dir / "source_config.json").write_text(
                json.dumps(
                    {
                        "group_cols": [
                            "seq_a",
                            "seq_c",
                            "serumPassCat",
                            "virusPassCat",
                        ]
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "processed2splited.py",
                "--dataset-dir",
                str(dataset_dir),
                "--seed",
                "0",
                "--test-ratio",
                "0.25",
                "--valid-ratio",
                "0.25",
            ]
            with patch.object(sys, "argv", argv):
                self.module.main()

            split_roots = sorted((dataset_dir / "splited").iterdir())
            self.assertEqual(len(split_roots), 1)
            split_root = split_roots[0]
            self.assertRegex(split_root.name, r"^\d{8}_\d{6}$")
            self.assertFalse((dataset_dir / "splited" / "v1").exists())

            for mode in ("titer", "strain", "serum"):
                mode_dir = split_root / mode / "seed_0"
                self.assertTrue((mode_dir / "train.csv").is_file())
                self.assertTrue((mode_dir / "valid.csv").is_file())
                self.assertTrue((mode_dir / "test.csv").is_file())
                self.assertTrue((mode_dir / "manifest.json").is_file())

            strain_dir = split_root / "strain" / "seed_0"
            strain_train = pd.read_csv(strain_dir / "train.csv")
            strain_valid = pd.read_csv(strain_dir / "valid.csv")
            strain_test = pd.read_csv(strain_dir / "test.csv")
            self.assertEqual(_overlap_count(strain_train, strain_valid, "seq_c"), 0)
            self.assertEqual(_overlap_count(strain_train, strain_test, "seq_c"), 0)
            self.assertEqual(_overlap_count(strain_valid, strain_test, "seq_c"), 0)

            serum_dir = split_root / "serum" / "seed_0"
            serum_train = pd.read_csv(serum_dir / "train.csv")
            serum_valid = pd.read_csv(serum_dir / "valid.csv")
            serum_test = pd.read_csv(serum_dir / "test.csv")
            self.assertEqual(_overlap_count(serum_train, serum_valid, "seq_a"), 0)
            self.assertEqual(_overlap_count(serum_train, serum_test, "seq_a"), 0)
            self.assertEqual(_overlap_count(serum_valid, serum_test, "seq_a"), 0)

            manifest = json.loads((serum_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["valid_split"], "group")
            self.assertEqual(manifest["seed"], 0)
            self.assertEqual(manifest["source_duplicate_check"]["duplicate_rows"], 0)

    def test_multiple_seeds_create_seed_directories_per_random_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "H1H3_HA1"
            processed_dir = dataset_dir / "processed"
            processed_dir.mkdir(parents=True)

            rows = []
            for idx in range(20):
                rows.append(
                    {
                        "seq_a": f"serum_{idx}",
                        "seq_c": f"virus_{idx}",
                        "serumPassCat": "<NONE>",
                        "virusPassCat": "<EGG>",
                        "label": float(idx),
                        "sheet": f"{idx // 5 + 1}-x",
                    }
                )
            pd.DataFrame(rows).to_csv(processed_dir / "source.csv", index=False)
            (processed_dir / "source_config.json").write_text(
                json.dumps(
                    {
                        "group_cols": [
                            "seq_a",
                            "seq_c",
                            "serumPassCat",
                            "virusPassCat",
                        ]
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "processed2splited.py",
                "--dataset-dir",
                str(dataset_dir),
                "--seed",
                "1,2",
                "--test-ratio",
                "0.2",
                "--valid-ratio",
                "0.2",
                "--split-modes",
                "titer,strain",
            ]
            with patch.object(sys, "argv", argv):
                self.module.main()

            split_root = next((dataset_dir / "splited").iterdir())
            for mode in ("titer", "strain"):
                self.assertTrue((split_root / mode / "seed_1" / "train.csv").is_file())
                self.assertTrue((split_root / mode / "seed_2" / "train.csv").is_file())
                manifest_1 = json.loads(
                    (split_root / mode / "seed_1" / "manifest.json").read_text(encoding="utf-8")
                )
                manifest_2 = json.loads(
                    (split_root / mode / "seed_2" / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest_1["seed"], 1)
                self.assertEqual(manifest_2["seed"], 2)

    def test_cli_accepts_comma_separated_group_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "H1H3_HA1"
            processed_dir = dataset_dir / "processed"
            processed_dir.mkdir(parents=True)

            rows = []
            for idx in range(20):
                rows.append(
                    {
                        "seq_a": f"serum_{idx % 5}",
                        "serumPassCat": "<NONE>" if idx % 2 else "<EGG>",
                        "serumName": f"serum_name_{idx}",
                        "seq_c": f"virus_{idx % 10}",
                        "virusPassCat": "<MDCK>" if idx % 2 else "<EGG>",
                        "virusName": f"virus_name_{idx}",
                        "label": float(idx),
                        "sheet": f"{idx // 5 + 1}-x",
                    }
                )
            pd.DataFrame(rows).to_csv(processed_dir / "source.csv", index=False)
            (processed_dir / "source_config.json").write_text(
                json.dumps(
                    {
                        "group_cols": [
                            "seq_a",
                            "seq_c",
                            "serumPassCat",
                            "virusPassCat",
                            "serumName",
                            "virusName",
                        ]
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "processed2splited.py",
                "--dataset-dir",
                str(dataset_dir),
                "--seed",
                "1",
                "--test-ratio",
                "0.2",
                "--valid-ratio",
                "0.2",
                "--split-modes",
                "strain,serum",
                "--strain-col",
                "seq_c,virusPassCat,virusName",
                "--serum-col",
                "seq_a,serumPassCat,serumName",
            ]
            with patch.object(sys, "argv", argv):
                self.module.main()

            split_root = next((dataset_dir / "splited").iterdir())
            strain_manifest = json.loads(
                (split_root / "strain" / "seed_1" / "manifest.json").read_text(encoding="utf-8")
            )
            serum_manifest = json.loads(
                (split_root / "serum" / "seed_1" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(strain_manifest["group_columns"], ["seq_c", "virusPassCat", "virusName"])
            self.assertEqual(serum_manifest["group_columns"], ["seq_a", "serumPassCat", "serumName"])

    def test_split_helpers_accept_seed_lists(self):
        df = pd.DataFrame(
            [
                {"seq_a": f"serum_{idx % 4}", "seq_c": f"virus_{idx}", "label": float(idx)}
                for idx in range(12)
            ]
        )

        row_splits = self.module.split_rows(df, [1, 2], 0.25, 0.25)
        self.assertEqual(set(row_splits), {1, 2})
        self.assertEqual([len(frame) for frame in row_splits[1]], [6, 3, 3])
        self.assertEqual([len(frame) for frame in row_splits[2]], [6, 3, 3])

        group_splits = self.module.split_by_group(df, ["seq_c"], [1, 2], 0.25, 0.25, "group")
        self.assertEqual(set(group_splits), {1, 2})
        self.assertEqual([len(frame) for frame in group_splits[1]], [6, 3, 3])
        self.assertEqual([len(frame) for frame in group_splits[2]], [6, 3, 3])

    def test_missing_source_config_raises_even_with_default_group_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "H1H3_HA1"
            processed_dir = dataset_dir / "processed"
            processed_dir.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "seq_a": "serum_0",
                        "seq_c": "virus_0",
                        "serumPassCat": "<NONE>",
                        "virusPassCat": "<EGG>",
                        "label": 0.0,
                        "sheet": "1-x",
                    }
                ]
            ).to_csv(processed_dir / "source.csv", index=False)

            with self.assertRaisesRegex(FileNotFoundError, "source_config.json"):
                self.module.load_source(dataset_dir)


if __name__ == "__main__":
    unittest.main()
