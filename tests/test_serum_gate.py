from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import SGD


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "serum_gate" / "train_zero_shot.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("train_zero_shot", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(
    seq_id_a: str = "ref_ha_1",
    seq_id_b: str = "ref_na_1",
    seq_id_c: str = "test_ha_1",
    seq_id_d: str = "test_na_1",
    serum_passage: str = "<EGG>",
    virus_passage: str = "<CELL>",
    label: float = 1.0,
    subtype: str = "H3N2",
    serum_date: str = "2020-01-01",
    virus_date: str = "2021-01-01",
):
    return {
        "seq_id_a": seq_id_a,
        "seq_id_b": seq_id_b,
        "seq_id_c": seq_id_c,
        "seq_id_d": seq_id_d,
        "seq_a": "AAAA",
        "seq_b": "ANVTAAANST",
        "seq_c": "AAAT",
        "seq_d": "ANVTAAAAAA",
        "serumPassCat": serum_passage,
        "virusPassCat": virus_passage,
        "serumName": f"serum_{seq_id_a}",
        "virusName": f"virus_{seq_id_c}",
        "label": label,
        "serumDate": serum_date,
        "Type": subtype,
        "virusDate": virus_date,
        "serumIslID": f"isl_{seq_id_a}",
        "virusIslID": f"isl_{seq_id_c}",
        "sheet": "sheet_1",
        "serumHA": "A" * 20,
        "virusHA": "A" * 19 + "T",
    }


class SerumGateDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_training_module()

    def test_make_task_key_uses_reference_ha_na_and_serum_passage(self):
        frame = pd.DataFrame(
            [
                _row(seq_id_a="ha1", seq_id_b="na1", serum_passage="<EGG>"),
                _row(seq_id_a="ha1", seq_id_b="na2", serum_passage="<EGG>"),
                _row(seq_id_a="ha1", seq_id_b="na1", serum_passage="<CELL>"),
            ]
        )

        keys = self.module.make_task_keys(frame, ["seq_id_a", "seq_id_b", "serumPassCat"])

        self.assertEqual(keys.tolist(), ["ha1||na1||<EGG>", "ha1||na2||<EGG>", "ha1||na1||<CELL>"])

    def test_make_task_key_returns_empty_series_for_empty_frame(self):
        frame = pd.DataFrame(columns=list(_row().keys()))

        keys = self.module.make_task_keys(frame, ["seq_id_a", "seq_id_b", "serumPassCat"])

        self.assertEqual(keys.tolist(), [])
        self.assertEqual(len(keys), 0)

    def test_load_fixed_split_preserves_external_protocol_with_task_overlap(self):
        train = pd.DataFrame([_row(seq_id_a="shared_ref", seq_id_b="shared_na", seq_id_c="train_query")])
        valid = pd.DataFrame([_row(seq_id_a="shared_ref", seq_id_b="shared_na", seq_id_c="valid_query")])
        test = pd.DataFrame([_row(seq_id_a="shared_ref", seq_id_b="shared_na", seq_id_c="test_query")])

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            train.to_csv(data_dir / "train.csv", index=False)
            valid.to_csv(data_dir / "valid.csv", index=False)
            test.to_csv(data_dir / "test.csv", index=False)

            splits = self.module.load_fixed_split_frames(data_dir)

        pd.testing.assert_frame_equal(splits["train"].reset_index(drop=True), train)
        pd.testing.assert_frame_equal(splits["valid"].reset_index(drop=True), valid)
        pd.testing.assert_frame_equal(splits["test"].reset_index(drop=True), test)

    def test_refit_train_valid_merges_validation_into_training(self):
        train = pd.DataFrame([_row(seq_id_a="train_ref", seq_id_b="train_na", seq_id_c="train_query")])
        valid = pd.DataFrame([_row(seq_id_a="valid_ref", seq_id_b="valid_na", seq_id_c="valid_query")])
        test = pd.DataFrame([_row(seq_id_a="test_ref", seq_id_b="test_na", seq_id_c="test_query")])

        splits = self.module.merge_train_valid_for_refit({"train": train, "valid": valid, "test": test})

        self.assertEqual(len(splits["train"]), 2)
        self.assertEqual(len(splits["valid"]), 0)
        self.assertEqual(splits["train"]["seq_id_a"].tolist(), ["train_ref", "valid_ref"])
        pd.testing.assert_frame_equal(splits["test"].reset_index(drop=True), test.reset_index(drop=True))

    def test_quantile_label_weight_thresholds_use_current_training_frame(self):
        train = pd.DataFrame([_row(label=value) for value in [0.0, 10.0, 20.0, 30.0, 40.0]])
        valid = pd.DataFrame([_row(label=100.0)])
        test = pd.DataFrame([_row(label=1000.0)])

        thresholds = self.module.resolve_label_weight_thresholds(
            {"train": train, "valid": valid, "test": test},
            mode="quantile",
            thresholds_arg="2,4,6",
            quantiles_arg="0.35,0.75,0.95",
        )

        self.assertEqual(thresholds, [14.0, 30.0, 38.0])

    def test_quantile_label_weight_thresholds_reflect_refit_train_valid_merge(self):
        train = pd.DataFrame([_row(label=value) for value in [0.0, 10.0, 20.0, 30.0]])
        valid = pd.DataFrame([_row(label=value) for value in [40.0, 50.0]])
        test = pd.DataFrame([_row(label=1000.0)])
        frames = self.module.merge_train_valid_for_refit({"train": train, "valid": valid, "test": test})

        thresholds = self.module.resolve_label_weight_thresholds(
            frames,
            mode="quantile",
            thresholds_arg="2,4,6",
            quantiles_arg="0.35,0.75,0.95",
        )

        self.assertEqual(thresholds, [17.5, 37.5, 47.5])

    def test_serum_task_dataset_groups_queries_and_keeps_empty_support(self):
        frame = pd.DataFrame(
            [
                _row(seq_id_a="ref_ha", seq_id_b="ref_na", seq_id_c="test_1", label=1.0),
                _row(seq_id_a="ref_ha", seq_id_b="ref_na", seq_id_c="test_2", label=2.0),
            ]
        )
        vocabs = self.module.build_serum_gate_vocabs({"train": frame, "valid": frame, "test": frame})

        dataset = self.module.SerumGateTaskDataset(frame, vocabs)
        item = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(item["task_key"], "ref_ha||ref_na||<EGG>")
        self.assertEqual(item["support_size"], 0)
        self.assertEqual(item["reference_ha_key"], "matrix_ref_ha")
        self.assertEqual(item["query_ha_keys"], ["matrix_test_1", "matrix_test_2"])
        self.assertTrue(torch.allclose(item["labels"], torch.tensor([1.0, 2.0])))

    def test_required_embedding_files_can_include_na_embeddings(self):
        frame = pd.DataFrame(
            [
                _row(seq_id_a="ref_ha", seq_id_b="ref_na", seq_id_c="test_ha", seq_id_d="test_na"),
            ]
        )

        files = self.module.required_embedding_files({"train": frame}, include_na_embeddings=True)

        self.assertEqual(
            files,
            [
                "matrix_ref_ha.pt",
                "matrix_ref_na.pt",
                "matrix_test_ha.pt",
                "matrix_test_na.pt",
            ],
        )

    def test_subtype_feature_can_be_disabled_for_subtype_specific_training(self):
        h1_frame = pd.DataFrame([_row(subtype="H1N1")])
        h3_frame = pd.DataFrame([_row(subtype="H3N2")])

        vocabs = self.module.build_serum_gate_vocabs(
            {"train": h1_frame, "valid": h1_frame, "test": h3_frame},
            use_subtype_feature=False,
        )

        self.assertEqual(vocabs.subtype_to_id, {"constant": 0})
        self.assertEqual(
            self.module.SerumGateTaskDataset(h1_frame, vocabs)[0]["subtype"].item(),
            0,
        )

    def test_no_use_subtype_feature_sets_subtype_dim_to_zero(self):
        args = self.module.parse_args(
            [
                "--data-dir",
                "/tmp",
                "--embedding-dir",
                "/tmp",
                "--output-dir",
                "/tmp/out",
                "--no-use-subtype-feature",
            ]
        )

        self.assertFalse(args.use_subtype_feature)
        self.assertEqual(args.subtype_dim, 0)

    def test_parse_args_accepts_type_filter_for_subtype_specific_training(self):
        args = self.module.parse_args(
            [
                "--data-dir",
                "/tmp",
                "--embedding-dir",
                "/tmp",
                "--output-dir",
                "/tmp/out",
                "--type",
                "H1N1",
            ]
        )

        self.assertEqual(args.type_filter, "H1N1")
        self.assertFalse(args.use_subtype_feature)
        self.assertEqual(args.subtype_dim, 0)

    def test_filter_frames_by_type_keeps_only_matching_rows(self):
        frames = {
            "train": pd.DataFrame(
                [
                    _row(seq_id_a="h1_train", subtype="H1N1"),
                    _row(seq_id_a="h3_train", subtype="H3N2"),
                ]
            ),
            "valid": pd.DataFrame([_row(seq_id_a="h3_valid", subtype="H3N2")]),
            "test": pd.DataFrame(
                [
                    _row(seq_id_a="h1_test", subtype="H1N1"),
                    _row(seq_id_a="h3_test", subtype="H3N2"),
                ]
            ),
        }

        filtered = self.module.filter_frames_by_type(frames, "H1N1")

        self.assertEqual(filtered["train"]["seq_id_a"].tolist(), ["h1_train"])
        self.assertEqual(filtered["valid"]["seq_id_a"].tolist(), [])
        self.assertEqual(filtered["test"]["seq_id_a"].tolist(), ["h1_test"])
        for frame in filtered.values():
            self.assertTrue(frame.empty or set(frame["Type"]) == {"H1N1"})

    def test_serum_task_dataset_and_collate_can_include_na_embeddings(self):
        frame = pd.DataFrame(
            [
                _row(seq_id_a="ref_ha", seq_id_b="ref_na", seq_id_c="test_1", seq_id_d="test_na_1", label=1.0),
                _row(seq_id_a="ref_ha", seq_id_b="ref_na", seq_id_c="test_2", seq_id_d="test_na_2", label=2.0),
            ]
        )
        vocabs = self.module.build_serum_gate_vocabs({"train": frame, "valid": frame, "test": frame})
        item = self.module.SerumGateTaskDataset(frame, vocabs)[0]
        embeddings = {
            "matrix_ref_ha": torch.ones(2, 3),
            "matrix_test_1": torch.ones(3, 3),
            "matrix_test_2": torch.ones(4, 3),
            "matrix_ref_na": torch.full((5, 3), 2.0),
            "matrix_test_na_1": torch.full((6, 3), 3.0),
            "matrix_test_na_2": torch.full((7, 3), 4.0),
        }

        self.assertEqual(item["reference_na_key"], "matrix_ref_na")
        self.assertEqual(item["query_na_keys"], ["matrix_test_na_1", "matrix_test_na_2"])
        batch, _items = self.module.collate_serum_gate_tasks([item], embeddings, include_na_embeddings=True)

        self.assertEqual(batch.reference_na.shape, torch.Size([1, 5, 3]))
        self.assertEqual(batch.query_na.shape, torch.Size([1, 2, 7, 3]))
        self.assertTrue(torch.allclose(batch.reference_na_mask, torch.ones(1, 5)))
        self.assertEqual(batch.query_na_mask.shape, torch.Size([1, 2, 7]))

    def test_serum_task_dataset_can_chunk_large_query_profiles(self):
        frame = pd.DataFrame(
            [
                _row(seq_id_a="ref_ha", seq_id_b="ref_na", seq_id_c=f"test_{idx}", label=float(idx))
                for idx in range(5)
            ]
        )
        vocabs = self.module.build_serum_gate_vocabs({"train": frame, "valid": frame, "test": frame})

        dataset = self.module.SerumGateTaskDataset(frame, vocabs, max_queries_per_task=2)

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0]["task_key"], dataset[1]["task_key"])
        self.assertEqual(dataset[0]["query_ha_keys"], ["matrix_test_0", "matrix_test_1"])
        self.assertEqual(dataset[1]["query_ha_keys"], ["matrix_test_2", "matrix_test_3"])
        self.assertEqual(dataset[2]["query_ha_keys"], ["matrix_test_4"])

    def test_delta_collate_aligns_different_length_embeddings_to_ha_coordinates(self):
        frame = pd.DataFrame(
            [
                {
                    **_row(seq_id_a="ref_ha", seq_id_b="ref_na", seq_id_c="test_ha", label=1.0),
                    "serumHA": "AB-CD",
                    "virusHA": "A-BCD",
                }
            ]
        )
        vocabs = self.module.build_serum_gate_vocabs({"train": frame, "valid": frame, "test": frame})
        item = self.module.SerumGateTaskDataset(frame, vocabs)[0]
        embeddings = {
            "matrix_ref_ha": torch.tensor([[1.0], [2.0], [3.0]]),
            "matrix_test_ha": torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
        }

        batch, _items = self.module.collate_serum_gate_tasks([item], embeddings, align_ha_embeddings=True)

        self.assertEqual(batch.reference_ha.shape, torch.Size([1, 5, 1]))
        self.assertEqual(batch.query_ha.shape, torch.Size([1, 1, 5, 1]))
        self.assertTrue(torch.allclose(batch.reference_ha[:, :, 0], torch.tensor([[1.0, 2.0, 0.0, 3.0, 0.0]])))
        self.assertTrue(torch.allclose(batch.query_ha[0, :, :, 0], torch.tensor([[10.0, 0.0, 20.0, 30.0, 40.0]])))
        self.assertTrue(torch.allclose(batch.reference_ha_mask, torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]])))
        self.assertTrue(torch.allclose(batch.query_ha_mask[0], torch.tensor([[1.0, 0.0, 1.0, 1.0, 1.0]])))

    def test_serum_regression_metrics_include_serum_bias_and_coverage(self):
        frame = pd.DataFrame(
            {
                "task_key": ["s1", "s1", "s2", "s2"],
                "label": [0.0, 2.0, 1.0, 3.0],
                "mean": [0.5, 1.5, 1.0, 4.0],
                "log_var": [0.0, 0.0, 0.0, 0.0],
            }
        )

        metrics = self.module.serum_regression_metrics(frame)

        self.assertAlmostEqual(metrics["pooled_mae"], 0.5)
        self.assertIn("per_serum_mae_mean", metrics)
        self.assertIn("serum_bias_abs_mean", metrics)
        self.assertIn("coverage_95", metrics)
        self.assertIn("nll", metrics)

    def test_flatten_epoch_metrics_rounds_floats_for_csv_output(self):
        epoch_metrics = {
            "epoch": 3,
            "train_loss": 0.123456,
            "valid": {"pooled_mae": 0.0, "pooled_spearman": 0.987654},
            "test": {"pooled_mae": 1.234567, "loss": 0.333333},
        }

        row = self.module.flatten_epoch_metrics(epoch_metrics)

        self.assertEqual(row["epoch"], 3)
        self.assertEqual(row["train_loss"], 0.1235)
        self.assertEqual(row["valid_pooled_spearman"], 0.9877)
        self.assertEqual(row["test_pooled_mae"], 1.2346)
        self.assertNotIn("valid", row)
        self.assertNotIn("test", row)

    def test_train_one_epoch_updates_progress_once_per_episode(self):
        from fluprofiler.models.serum_gate_model import SerumGateBatch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, _batch):
                loss = self.weight * 0.0 + 1.0
                return {"nll_loss": loss, "huber_loss": loss}

        class Progress:
            def __init__(self):
                self.count = 0

            def update(self, value):
                self.count += value

        batch = SerumGateBatch(
            reference_ha=torch.ones(1, 1, 2),
            query_ha=torch.ones(1, 1, 1, 2),
            labels=torch.zeros(1, 1),
        )
        loader = [(batch, []), (batch, [])]
        model = TinyModel()
        progress = Progress()

        self.module.train_one_epoch(
            model,
            loader,
            SGD(model.parameters(), lr=0.1),
            torch.device("cpu"),
            "nll",
            progress_bar=progress,
        )

        self.assertEqual(progress.count, 2)


class SerumGateModelTests(unittest.TestCase):
    def test_label_bin_weights_increase_for_high_labels(self):
        from fluprofiler.models.serum_gate_model import label_bin_weights, weighted_masked_mean

        labels = torch.tensor([[-1.0, 1.9, 2.0, 4.0, 6.0]])
        values = torch.ones_like(labels)
        weights = label_bin_weights(labels, thresholds=(2.0, 4.0, 6.0), values=(1.0, 1.3, 1.8, 2.5))

        self.assertTrue(torch.allclose(weights, torch.tensor([[1.0, 1.0, 1.3, 1.8, 2.5]])))
        self.assertAlmostEqual(float(weighted_masked_mean(values, weights, None)), 1.0)
        self.assertAlmostEqual(
            float(weighted_masked_mean(values, weights, torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]]))),
            1.0,
        )

    def test_pairwise_ranking_loss_penalizes_reversed_within_serum_order(self):
        from fluprofiler.models.serum_gate_model import pairwise_ranking_loss

        labels = torch.tensor([[0.0, 2.0, 1.0]])
        ordered = torch.tensor([[0.0, 2.0, 1.0]])
        reversed_order = torch.tensor([[2.0, 0.0, 1.0]])

        ordered_loss = pairwise_ranking_loss(ordered, labels, margin=0.5)
        reversed_loss = pairwise_ranking_loss(reversed_order, labels, margin=0.5)

        self.assertAlmostEqual(float(ordered_loss), 0.0)
        self.assertGreater(float(reversed_loss), 0.0)

    def test_subtype_dim_zero_omits_subtype_embedding_parameter(self):
        from fluprofiler.models.serum_gate_model import SerumGateConfig, SerumGateModel

        model = SerumGateModel(
            SerumGateConfig(
                hidden_size=2,
                latent_dim=3,
                theta_dim=4,
                passage_vocab_size=3,
                passage_pair_vocab_size=9,
                subtype_vocab_size=1,
                subtype_dim=0,
            )
        )

        self.assertIsNone(model.subtype_embedding)
        self.assertFalse(any(name.startswith("subtype_embedding") for name, _ in model.named_parameters()))

    def test_model_forward_adds_weighted_rank_loss_to_regression_losses(self):
        from fluprofiler.models.serum_gate_model import (
            SerumGateBatch,
            SerumGateConfig,
            SerumGateModel,
            label_bin_weights,
            weighted_masked_mean,
        )

        class FixedPredictor(torch.nn.Module):
            def forward(self, pair_features, theta):
                mean = torch.tensor([[2.0, 0.0, 1.0]], dtype=pair_features.dtype, device=pair_features.device)
                log_var = torch.zeros_like(mean)
                return mean, log_var

        config = SerumGateConfig(
            hidden_size=2,
            latent_dim=3,
            theta_dim=4,
            passage_vocab_size=3,
            passage_pair_vocab_size=9,
            subtype_vocab_size=2,
            rank_loss_weight=0.5,
            rank_loss_margin=0.5,
        )
        model = SerumGateModel(config)
        model.predictor = FixedPredictor()
        batch = SerumGateBatch(
            reference_ha=torch.ones(1, 2, 2),
            query_ha=torch.ones(1, 3, 2, 2),
            reference_ha_mask=torch.ones(1, 2),
            query_ha_mask=torch.ones(1, 3, 2),
            serum_passage=torch.tensor([1]),
            query_passage=torch.tensor([[1, 2, 1]]),
            passage_pair=torch.tensor([[4, 5, 4]]),
            subtype=torch.tensor([1]),
            s_nagly=torch.tensor([[0.0, 1.0, 0.0]]),
            labels=torch.tensor([[0.0, 2.0, 1.0]]),
            query_mask=torch.tensor([[1.0, 1.0, 1.0]]),
        )

        out = model(batch)
        labels = batch.labels.float()
        nll = 0.5 * (labels - out["mean"]).pow(2)
        weights = label_bin_weights(labels, config.label_weight_thresholds, config.label_weight_values)
        expected_nll = weighted_masked_mean(nll, weights, batch.query_mask)

        self.assertGreater(float(out["rank_loss"]), 0.0)
        self.assertTrue(torch.allclose(out["nll_loss"], expected_nll + 0.5 * out["rank_loss"]))

    def test_calibrated_metric_model_has_expected_parameter_count(self):
        from fluprofiler.models.serum_gate_model import SerumGateConfig, SerumGateModel

        config = SerumGateConfig(
            hidden_size=2560,
            passage_vocab_size=4,
            passage_pair_vocab_size=16,
            subtype_vocab_size=3,
            ha_pooling="lowrank_attention",
            ha_attention_dim=64,
            ha_attention_heads=4,
            ha_attention_dropout=0.2,
            ha_mean_gate_init=0.05,
            predictor_arch="calibrated_metric",
            distance_hidden_dim=64,
            calibration_hidden_dim=64,
            residual_hidden_dim=32,
            residual_scale=0.25,
        )
        model = SerumGateModel(config)

        self.assertEqual(sum(param.numel() for param in model.parameters()), 570271)

    def test_calibrated_metric_forward_returns_calibration_parts(self):
        from fluprofiler.models.serum_gate_model import (
            SerumGateBatch,
            SerumGateConfig,
            SerumGateModel,
        )

        config = SerumGateConfig(
            hidden_size=2,
            latent_dim=3,
            theta_dim=4,
            passage_vocab_size=3,
            passage_pair_vocab_size=9,
            subtype_vocab_size=2,
            ha_pooling="mean",
            predictor_arch="calibrated_metric",
            distance_hidden_dim=5,
            calibration_hidden_dim=6,
            residual_hidden_dim=4,
            residual_scale=0.25,
        )
        model = SerumGateModel(config)
        batch = SerumGateBatch(
            reference_ha=torch.ones(1, 2, 2),
            query_ha=torch.ones(1, 3, 2, 2),
            reference_ha_mask=torch.ones(1, 2),
            query_ha_mask=torch.ones(1, 3, 2),
            serum_passage=torch.tensor([1]),
            query_passage=torch.tensor([[1, 2, 1]]),
            passage_pair=torch.tensor([[4, 5, 4]]),
            subtype=torch.tensor([1]),
            s_nagly=torch.tensor([[0.0, 1.0, 0.0]]),
            labels=torch.tensor([[0.0, 1.0, 2.0]]),
            query_mask=torch.tensor([[1.0, 1.0, 1.0]]),
        )

        out = model(batch)

        self.assertEqual(out["mean"].shape, torch.Size([1, 3]))
        self.assertEqual(out["log_var"].shape, torch.Size([1, 3]))
        self.assertEqual(out["distance"].shape, torch.Size([1, 3]))
        self.assertEqual(out["residual"].shape, torch.Size([1, 3]))
        self.assertEqual(out["serum_bias"].shape, torch.Size([1]))
        self.assertEqual(out["serum_scale"].shape, torch.Size([1]))
        self.assertTrue(torch.all(out["distance"] >= 0.0))
        self.assertTrue(torch.all(out["serum_scale"] > 0.0))
        self.assertTrue(torch.isfinite(out["nll_loss"]))

    def test_na_pair_branch_adds_na_effect_to_predictions(self):
        from fluprofiler.models.serum_gate_model import (
            SerumGateBatch,
            SerumGateConfig,
            SerumGateModel,
        )

        config = SerumGateConfig(
            hidden_size=2,
            latent_dim=3,
            theta_dim=4,
            passage_vocab_size=3,
            passage_pair_vocab_size=9,
            subtype_vocab_size=2,
            ha_pooling="mean",
            na_branch="pair",
            na_pooling="mean",
            na_latent_dim=3,
            na_hidden_dim=5,
            na_effect_init=0.1,
        )
        model = SerumGateModel(config)
        batch = SerumGateBatch(
            reference_ha=torch.ones(1, 2, 2),
            query_ha=torch.ones(1, 3, 2, 2),
            reference_ha_mask=torch.ones(1, 2),
            query_ha_mask=torch.ones(1, 3, 2),
            reference_na=torch.ones(1, 4, 2),
            query_na=torch.ones(1, 3, 5, 2),
            reference_na_mask=torch.ones(1, 4),
            query_na_mask=torch.ones(1, 3, 5),
            serum_passage=torch.tensor([1]),
            query_passage=torch.tensor([[1, 2, 1]]),
            passage_pair=torch.tensor([[4, 5, 4]]),
            subtype=torch.tensor([1]),
            s_nagly=torch.tensor([[0.0, 1.0, 0.0]]),
            labels=torch.tensor([[0.0, 1.0, 2.0]]),
            query_mask=torch.tensor([[1.0, 1.0, 1.0]]),
        )

        out = model(batch)

        self.assertEqual(out["mean"].shape, torch.Size([1, 3]))
        self.assertEqual(out["na_effect"].shape, torch.Size([1, 3]))
        self.assertEqual(out["z_na_ref"].shape, torch.Size([1, 3]))
        self.assertEqual(out["z_na_query"].shape, torch.Size([1, 3, 3]))
        self.assertTrue(torch.isfinite(out["nll_loss"]))

    def test_lightweight_na_pair_branch_has_expected_parameter_count(self):
        from fluprofiler.models.serum_gate_model import SerumGateConfig, SerumGateModel

        config = SerumGateConfig(
            hidden_size=2560,
            passage_vocab_size=4,
            passage_pair_vocab_size=16,
            subtype_vocab_size=2,
            ha_pooling="lowrank_attention",
            ha_attention_dim=64,
            ha_attention_heads=4,
            ha_attention_dropout=0.2,
            ha_mean_gate_init=0.05,
            na_branch="pair",
            na_pooling="mean",
            na_latent_dim=32,
            na_hidden_dim=32,
            na_effect_init=0.1,
        )
        model = SerumGateModel(config)

        self.assertEqual(sum(param.numel() for param in model.parameters()), 904982)

    def test_model_forward_returns_mean_log_var_and_losses(self):
        from fluprofiler.models.serum_gate_model import (
            SerumGateBatch,
            SerumGateConfig,
            SerumGateModel,
        )

        config = SerumGateConfig(
            hidden_size=2,
            latent_dim=3,
            theta_dim=4,
            passage_vocab_size=3,
            passage_pair_vocab_size=9,
            subtype_vocab_size=2,
        )
        model = SerumGateModel(config)
        batch = SerumGateBatch(
            reference_ha=torch.ones(1, 2, 2),
            query_ha=torch.ones(1, 3, 2, 2),
            reference_ha_mask=torch.ones(1, 2),
            query_ha_mask=torch.ones(1, 3, 2),
            serum_passage=torch.tensor([1]),
            query_passage=torch.tensor([[1, 2, 1]]),
            passage_pair=torch.tensor([[4, 5, 4]]),
            subtype=torch.tensor([1]),
            s_nagly=torch.tensor([[0.0, 1.0, 0.0]]),
            labels=torch.tensor([[0.0, 1.0, 2.0]]),
            query_mask=torch.tensor([[1.0, 1.0, 1.0]]),
        )

        out = model(batch)

        self.assertEqual(out["mean"].shape, torch.Size([1, 3]))
        self.assertEqual(out["log_var"].shape, torch.Size([1, 3]))
        self.assertTrue(torch.isfinite(out["huber_loss"]))
        self.assertTrue(torch.isfinite(out["nll_loss"]))

    def test_ha_pooling_can_use_attention_mean_or_lowrank_attention(self):
        from fluprofiler.models.serum_gate_model import SerumGateConfig, SerumGateModel

        attention_model = SerumGateModel(SerumGateConfig(hidden_size=2, ha_pooling="attention"))
        mean_model = SerumGateModel(SerumGateConfig(hidden_size=2, ha_pooling="mean"))
        lowrank_model = SerumGateModel(
            SerumGateConfig(
                hidden_size=8,
                latent_dim=4,
                theta_dim=4,
                ha_pooling="lowrank_attention",
                ha_attention_dim=4,
                ha_attention_heads=2,
            )
        )

        self.assertEqual(attention_model.ha_encoder.pooling, "attention")
        self.assertIsNotNone(attention_model.ha_encoder.attention_pooler)
        self.assertEqual(mean_model.ha_encoder.pooling, "mean")
        self.assertIsNone(mean_model.ha_encoder.attention_pooler)
        self.assertEqual(lowrank_model.ha_encoder.pooling, "lowrank_attention")
        self.assertIsNotNone(lowrank_model.ha_encoder.lowrank_pooler)

        batch = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        pooled = lowrank_model.ha_encoder(batch, mask)

        self.assertEqual(pooled.shape, torch.Size([1, 4]))

    def test_lowrank_attention_pooling_is_much_smaller_than_full_attention(self):
        from fluprofiler.models.serum_gate_model import SerumGateConfig, SerumGateModel

        full_attention = SerumGateModel(
            SerumGateConfig(hidden_size=256, ha_pooling="attention")
        ).ha_encoder
        lowrank_attention = SerumGateModel(
            SerumGateConfig(
                hidden_size=256,
                ha_pooling="lowrank_attention",
                ha_attention_dim=16,
                ha_attention_heads=4,
            )
        ).ha_encoder

        full_params = sum(param.numel() for param in full_attention.parameters())
        lowrank_params = sum(param.numel() for param in lowrank_attention.parameters())

        self.assertLess(lowrank_params, full_params // 4)

    def test_parse_args_accepts_lowrank_attention_pooling(self):
        module = _load_training_module()

        args = module.parse_args(
            [
                "--data-dir",
                "data",
                "--embedding-dir",
                "embeddings",
                "--output-dir",
                "out",
                "--ha-pooling",
                "lowrank_attention",
            ]
        )

        self.assertEqual(args.ha_pooling, "lowrank_attention")

    def test_parse_args_rejects_training_time_season_resolution(self):
        module = _load_training_module()

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            module.parse_args(
                [
                    "--data-dir",
                    "data/dataset/H1H3_new/splited/v1/H1H3_new/season",
                    "--test-season",
                    "26",
                    "--embedding-dir",
                    "embeddings",
                    "--output-dir",
                    "out",
                ]
            )

    def test_parse_args_accepts_within_serum_ranking_loss_options(self):
        module = _load_training_module()

        args = module.parse_args(
            [
                "--data-dir",
                "data",
                "--embedding-dir",
                "embeddings",
                "--output-dir",
                "out",
                "--within-serum-rank-loss-weight",
                "0.2",
                "--within-serum-rank-margin",
                "0.4",
                "--within-serum-rank-min-label-delta",
                "0.5",
            ]
        )

        self.assertAlmostEqual(args.rank_loss_weight, 0.2)
        self.assertAlmostEqual(args.rank_loss_margin, 0.4)
        self.assertAlmostEqual(args.rank_loss_min_label_delta, 0.5)

    def test_parse_args_accepts_calibrated_metric_predictor_options(self):
        module = _load_training_module()

        args = module.parse_args(
            [
                "--data-dir",
                "data",
                "--embedding-dir",
                "embeddings",
                "--output-dir",
                "out",
                "--predictor-arch",
                "calibrated_metric",
                "--distance-hidden-dim",
                "64",
                "--calibration-hidden-dim",
                "64",
                "--residual-hidden-dim",
                "32",
                "--residual-scale",
                "0.25",
            ]
        )

        self.assertEqual(args.predictor_arch, "calibrated_metric")
        self.assertEqual(args.distance_hidden_dim, 64)
        self.assertEqual(args.calibration_hidden_dim, 64)
        self.assertEqual(args.residual_hidden_dim, 32)
        self.assertAlmostEqual(args.residual_scale, 0.25)

    def test_parse_args_accepts_na_pair_branch_options(self):
        module = _load_training_module()

        args = module.parse_args(
            [
                "--data-dir",
                "data",
                "--embedding-dir",
                "embeddings",
                "--output-dir",
                "out",
                "--na-branch",
                "pair",
                "--na-pooling",
                "mean",
                "--na-latent-dim",
                "32",
                "--na-hidden-dim",
                "64",
                "--na-effect-init",
                "0.1",
            ]
        )

        self.assertEqual(args.na_branch, "pair")
        self.assertEqual(args.na_pooling, "mean")
        self.assertEqual(args.na_latent_dim, 32)
        self.assertEqual(args.na_hidden_dim, 64)
        self.assertAlmostEqual(args.na_effect_init, 0.1)

    def test_delta_pair_mode_uses_compact_ha_pair_features(self):
        from fluprofiler.models.serum_gate_model import (
            SerumGateBatch,
            SerumGateConfig,
            SerumGateModel,
        )

        config = SerumGateConfig(
            hidden_size=2,
            latent_dim=3,
            theta_dim=4,
            passage_vocab_size=3,
            passage_pair_vocab_size=9,
            subtype_vocab_size=2,
            ha_pooling="mean",
            ha_pair_mode="delta",
        )
        model = SerumGateModel(config)

        self.assertEqual(model.predictor.input_layer.in_features, 3 + 8 + 8 + 8 + 1)

        batch = SerumGateBatch(
            reference_ha=torch.zeros(1, 2, 2),
            query_ha=torch.ones(1, 3, 2, 2),
            reference_ha_mask=torch.ones(1, 2),
            query_ha_mask=torch.ones(1, 3, 2),
            serum_passage=torch.tensor([1]),
            query_passage=torch.tensor([[1, 2, 1]]),
            passage_pair=torch.tensor([[4, 5, 4]]),
            subtype=torch.tensor([1]),
            s_nagly=torch.tensor([[0.0, 1.0, 0.0]]),
            labels=torch.tensor([[0.0, 1.0, 2.0]]),
            query_mask=torch.tensor([[1.0, 1.0, 1.0]]),
        )

        out = model(batch)

        self.assertEqual(out["mean"].shape, torch.Size([1, 3]))
        self.assertTrue(torch.isfinite(out["nll_loss"]))


if __name__ == "__main__":
    unittest.main()
