from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "HA_only" / "train_fixed_split.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("train_fixed_split", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(
    seq_id_a: str,
    seq_id_c: str,
    label: float = 1.0,
    serum_name: str = "serum",
    virus_name: str = "virus",
):
    return {
        "seq_id_a": seq_id_a,
        "seq_id_c": seq_id_c,
        "serumPassCat": "<EGG>",
        "virusPassCat": "<CELL>",
        "serumName": serum_name,
        "virusName": virus_name,
        "label": label,
    }


class FixedSplitInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_training_module()

    def test_missing_valid_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame([_row("train_a", "train_c")]).to_csv(data_dir / "train.csv", index=False)
            pd.DataFrame([_row("test_a", "test_c")]).to_csv(data_dir / "test.csv", index=False)

            with self.assertRaisesRegex(FileNotFoundError, "valid.csv"):
                self.module.load_fixed_split_frames(data_dir)

    def test_existing_valid_csv_is_used_as_validation_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame([_row("train_a", "train_c")]).to_csv(data_dir / "train.csv", index=False)
            pd.DataFrame([_row("provided_valid_a", "provided_valid_c")]).to_csv(
                data_dir / "valid.csv", index=False
            )
            pd.DataFrame([_row("test_a", "test_c")]).to_csv(data_dir / "test.csv", index=False)

            frames = self.module.load_fixed_split_frames(data_dir)

            self.assertEqual(frames["valid"]["seq_id_a"].tolist(), ["provided_valid_a"])

    def test_missing_required_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            incomplete = pd.DataFrame([_row("a", "c")]).drop(columns=["virusPassCat"])
            for filename in ("train.csv", "valid.csv", "test.csv"):
                incomplete.to_csv(data_dir / filename, index=False)

            with self.assertRaisesRegex(ValueError, "virusPassCat"):
                self.module.load_fixed_split_frames(data_dir)

    def test_missing_embedding_is_reported_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            embedding_dir = Path(tmp)
            required_files = ["matrix_present.pt", "matrix_missing.pt"]
            (embedding_dir / "matrix_present.pt").touch()

            with self.assertRaisesRegex(FileNotFoundError, "matrix_missing.pt"):
                self.module.validate_embedding_files(embedding_dir, required_files)

    def test_distance_model_impl_selects_distance_model(self):
        config = type("Config", (), {"hidden_size": 2, "matrix_fc_size": None, "loss_reduction": "mean"})()
        args = type("Args", (), {"output_mode": "regression", "ignore_index": -100})()

        model = self.module.build_ha_only_model(config, args, model_impl="distance")

        self.assertEqual(model.__class__.__name__, "fluProfiler_HA_only_distance_v2")

    def test_name_vocabs_are_built_from_train_and_unknown_maps_to_zero(self):
        frames = {
            "train": pd.DataFrame([_row("train_a", "train_c", serum_name="seen_serum", virus_name="seen_virus")]),
            "valid": pd.DataFrame([_row("valid_a", "valid_c", serum_name="seen_serum", virus_name="seen_virus")]),
            "test": pd.DataFrame([_row("test_a", "test_c", serum_name="new_serum", virus_name="new_virus")]),
        }

        vocabs = self.module.build_name_vocabs(frames["train"])
        datasets = self.module.build_datasets(frames, add_special_token=True, name_vocabs=vocabs)

        train_item = datasets["train"][0]
        test_item = datasets["test"][0]

        self.assertEqual(train_item[4].item(), 1)
        self.assertEqual(train_item[5].item(), 1)
        self.assertEqual(test_item[4].item(), 0)
        self.assertEqual(test_item[5].item(), 0)

if __name__ == "__main__":
    unittest.main()
