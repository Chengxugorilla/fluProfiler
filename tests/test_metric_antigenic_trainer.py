from __future__ import annotations

import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "metric_antigenic" / "train_metric_ha.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("train_metric_ha", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(
    seq_id_a: str = "ha_a",
    seq_id_b: str = "na_a",
    seq_id_c: str = "ha_c",
    seq_id_d: str = "na_c",
    seq_b: str = "ANVTAAANST",
    seq_d: str = "ANVTAAAAAA",
    serum_passage: str = "<EGG>",
    virus_passage: str = "<CELL>",
    subtype: str = "H3N2",
    label: float = 1.0,
    serum_name: str = "serum_name",
    virus_name: str = "virus_name",
):
    return {
        "seq_id_a": seq_id_a,
        "seq_id_b": seq_id_b,
        "seq_id_c": seq_id_c,
        "seq_id_d": seq_id_d,
        "seq_a": "HA_A",
        "seq_b": seq_b,
        "seq_c": "HA_C",
        "seq_d": seq_d,
        "serumPassCat": serum_passage,
        "virusPassCat": virus_passage,
        "serumName": serum_name,
        "virusName": virus_name,
        "serumDate": "2001-01-01",
        "virusDate": "2002-02-02",
        "serumIslID": "serum_isl",
        "virusIslID": "virus_isl",
        "serumHA": "SERUM_HA_ALN",
        "virusHA": "VIRUS_HA_ALN",
        "Type": subtype,
        "label": label,
    }


class MetricTrainerInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_training_module()

    def test_missing_valid_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame([_row()]).to_csv(data_dir / "train.csv", index=False)
            pd.DataFrame([_row()]).to_csv(data_dir / "test.csv", index=False)

            with self.assertRaisesRegex(FileNotFoundError, "valid.csv"):
                self.module.load_fixed_split_frames(data_dir)

    def test_missing_required_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            incomplete = pd.DataFrame([_row()]).drop(columns=["seq_d"])
            for filename in ("train.csv", "valid.csv", "test.csv"):
                incomplete.to_csv(data_dir / filename, index=False)

            with self.assertRaisesRegex(ValueError, "seq_d"):
                self.module.load_fixed_split_frames(data_dir)

    def test_required_embedding_files_use_ha_ids_only(self):
        frames = {
            "train": pd.DataFrame([_row(seq_id_a="ha_a", seq_id_c="ha_c")]),
            "valid": pd.DataFrame([_row(seq_id_a="ha_v", seq_id_c="ha_c")]),
            "test": pd.DataFrame([_row(seq_id_a="ha_a", seq_id_c="ha_t")]),
        }

        files = self.module.required_embedding_files(frames)

        self.assertEqual(
            files,
            ["matrix_ha_a.pt", "matrix_ha_c.pt", "matrix_ha_t.pt", "matrix_ha_v.pt"],
        )

    def test_missing_embedding_is_reported_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            embedding_dir = Path(tmp)
            (embedding_dir / "matrix_present.pt").touch()

            with self.assertRaisesRegex(FileNotFoundError, "matrix_missing.pt"):
                self.module.validate_embedding_files(
                    embedding_dir,
                    ["matrix_present.pt", "matrix_missing.pt"],
                )

    def test_feature_vocabs_include_unknown_and_training_categories(self):
        frames = {
            "train": pd.DataFrame(
                [
                    _row(serum_passage="<EGG>", virus_passage="<CELL>", subtype="H3N2"),
                    _row(serum_passage="", virus_passage="<EGG>", subtype="H1N1"),
                ]
            ),
            "valid": pd.DataFrame([_row(serum_passage="<CELL>", virus_passage="<EGG>", subtype="H3N2")]),
            "test": pd.DataFrame([_row(serum_passage="<EGG>", virus_passage="<CELL>", subtype="H1N1")]),
        }

        vocabs = self.module.build_feature_vocabs(frames)

        self.assertEqual(vocabs.passage_to_id["unknown"], 0)
        self.assertIn("egg", vocabs.passage_to_id)
        self.assertIn("cell", vocabs.passage_to_id)
        self.assertIn("H1N1", vocabs.subtype_to_id)
        self.assertIn("H3N2", vocabs.subtype_to_id)

    def test_dataset_computes_binary_na_glycan_mismatch_and_pair_id(self):
        frame = pd.DataFrame(
            [
                _row(
                    seq_id_a="ha_a",
                    seq_id_c="ha_c",
                    seq_b="ANVTAAANST",
                    seq_d="ANVTAAAAAA",
                    serum_passage="<EGG>",
                    virus_passage="<CELL>",
                    subtype="H3N2",
                    label=2.5,
                )
            ]
        )
        vocabs = self.module.build_feature_vocabs({"train": frame, "valid": frame, "test": frame})

        dataset = self.module.MetricHADataset(frame, vocabs)
        item = dataset[0]

        self.assertEqual(item["serum_ha_key"], "matrix_ha_a")
        self.assertEqual(item["virus_ha_key"], "matrix_ha_c")
        self.assertEqual(float(item["s_nagly"]), 1.0)
        self.assertEqual(item["passage_pair"], item["serum_passage"] * len(vocabs.passage_to_id) + item["test_passage"])
        self.assertEqual(float(item["label"]), 2.5)

    def test_dataset_computes_ha_mismatch_vector_from_aligned_sequences(self):
        serum_ha = "A" * 16 + "ACD" + "A" * 326
        virus_ha = "A" * 16 + "ATD" + "A" * 326
        frame = pd.DataFrame([_row()])
        frame.loc[0, "serumHA"] = serum_ha
        frame.loc[0, "virusHA"] = virus_ha
        vocabs = self.module.build_feature_vocabs({"train": frame, "valid": frame, "test": frame})

        item = self.module.MetricHADataset(frame, vocabs)[0]

        self.assertEqual(item["ha_mismatch"].shape, torch.Size([329]))
        self.assertTrue(torch.allclose(item["ha_mismatch"][:5], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])))

    def test_name_vocabs_are_train_only_and_dataset_maps_unknown_to_zero(self):
        train = pd.DataFrame([_row(serum_name="seen_serum", virus_name="seen_virus")])
        valid = pd.DataFrame([_row(serum_name="seen_serum", virus_name="seen_virus")])
        test = pd.DataFrame([_row(serum_name="new_serum", virus_name="new_virus")])
        feature_vocabs = self.module.build_feature_vocabs({"train": train, "valid": valid, "test": test})
        name_vocabs = self.module.build_name_vocabs(train)

        train_item = self.module.MetricHADataset(train, feature_vocabs, name_vocabs=name_vocabs)[0]
        test_item = self.module.MetricHADataset(test, feature_vocabs, name_vocabs=name_vocabs)[0]

        self.assertEqual(train_item["serum_name"].item(), 1)
        self.assertEqual(train_item["virus_name"].item(), 1)
        self.assertEqual(test_item["serum_name"].item(), 0)
        self.assertEqual(test_item["virus_name"].item(), 0)

    def test_build_reverse_pair_frame_swaps_serum_and_virus_fields(self):
        frame = pd.DataFrame(
            [
                _row(
                    seq_id_a="ha_serum",
                    seq_id_b="na_serum",
                    seq_id_c="ha_virus",
                    seq_id_d="na_virus",
                    seq_b="NA_SERUM",
                    seq_d="NA_VIRUS",
                    serum_passage="<EGG>",
                    virus_passage="<CELL>",
                    label=4.0,
                )
            ]
        )

        reversed_frame = self.module.build_reverse_pair_frame(frame)
        row = reversed_frame.iloc[0]

        self.assertEqual(row["seq_id_a"], "ha_virus")
        self.assertEqual(row["seq_id_c"], "ha_serum")
        self.assertEqual(row["seq_id_b"], "na_virus")
        self.assertEqual(row["seq_id_d"], "na_serum")
        self.assertEqual(row["seq_a"], "HA_C")
        self.assertEqual(row["seq_c"], "HA_A")
        self.assertEqual(row["serumPassCat"], "<CELL>")
        self.assertEqual(row["virusPassCat"], "<EGG>")
        self.assertEqual(row["serumName"], "virus_name")
        self.assertEqual(row["virusName"], "serum_name")
        self.assertEqual(row["serumDate"], "2002-02-02")
        self.assertEqual(row["virusDate"], "2001-01-01")
        self.assertEqual(row["serumIslID"], "virus_isl")
        self.assertEqual(row["virusIslID"], "serum_isl")
        self.assertEqual(row["serumHA"], "VIRUS_HA_ALN")
        self.assertEqual(row["virusHA"], "SERUM_HA_ALN")
        self.assertEqual(row["label"], 4.0)
        self.assertTrue(row["is_reverse_artificial"])

    def test_external_artificial_frame_is_merged_with_real_train_for_warmup(self):
        real_frame = pd.DataFrame([_row(seq_id_a="real_serum", seq_id_c="real_virus")])
        artificial_frame = pd.DataFrame([_row(seq_id_a="art_serum", seq_id_c="art_virus")])

        merged = self.module.build_artificial_augmented_train_frame(real_frame, artificial_frame)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.iloc[0]["seq_id_a"], "real_serum")
        self.assertEqual(merged.iloc[1]["seq_id_a"], "art_serum")
        self.assertFalse(bool(merged.iloc[0]["is_artificial"]))
        self.assertTrue(bool(merged.iloc[1]["is_artificial"]))

    def test_warmup_train_loader_can_use_external_artificial_frame(self):
        real_frame = pd.DataFrame([_row(seq_id_a="real_serum", seq_id_c="real_virus")])
        artificial_frame = pd.DataFrame([_row(seq_id_a="art_serum", seq_id_c="art_virus")])
        vocabs = self.module.build_feature_vocabs(
            {"train": real_frame, "valid": real_frame, "test": real_frame}
        )

        loader = self.module.build_artificial_warmup_train_loader(
            real_frame,
            artificial_frame,
            vocabs,
            batch_size=1,
            enabled=True,
        )

        self.assertIsNotNone(loader)
        self.assertEqual(len(loader.dataset), 2)
        serum_keys = {loader.dataset[idx]["serum_ha_key"] for idx in range(len(loader.dataset))}
        self.assertEqual(serum_keys, {"matrix_real_serum", "matrix_art_serum"})

    def test_warmup_train_loader_uses_augmented_data_only_for_configured_epochs(self):
        real_loader = DataLoader([0, 1, 2])
        augmented_loader = DataLoader([0, 1, 2, 3, 4, 5])

        self.assertIs(
            self.module.train_loader_for_epoch(real_loader, augmented_loader, epoch_index=0, reverse_warmup_epochs=1),
            augmented_loader,
        )
        self.assertIs(
            self.module.train_loader_for_epoch(real_loader, augmented_loader, epoch_index=1, reverse_warmup_epochs=1),
            real_loader,
        )
        self.assertIs(
            self.module.train_loader_for_epoch(real_loader, None, epoch_index=0, reverse_warmup_epochs=1),
            real_loader,
        )

    def test_train_one_epoch_updates_progress_once_per_batch(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, _batch):
                return {"loss": self.weight * 0.0 + 1.0}

        class Progress:
            def __init__(self):
                self.updates = []

            def update(self, value):
                self.updates.append(value)

        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        progress = Progress()
        dataloader = DataLoader([0, 1, 2], batch_size=1)

        with patch.object(self.module, "batch_to_model_input", return_value=object()):
            self.module.train_one_epoch(
                model,
                dataloader,
                torch.device("cpu"),
                cache=None,
                optimizer=optimizer,
                progress_bar=progress,
            )

        self.assertEqual(progress.updates, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
