from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "serum_gate" / "train_serum_mutation_set.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("train_serum_mutation_set", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainerDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_trainer()

    def test_save_epoch_defaults_to_last_epoch_and_accepts_multiple_epochs(self):
        arguments = [
            "--data-dir", "data",
            "--embedding-dir", "embeddings",
            "--ha-distance-matrix", "distance.npy",
            "--output-dir", "output",
            "--epochs", "100",
        ]
        self.assertEqual(self.module.parse_args(arguments).save_epoch, [100])
        self.assertEqual(
            self.module.parse_args([*arguments, "--save-epoch", "20", "5", "20"]).save_epoch,
            [5, 20],
        )

    def test_save_epoch_rejects_values_outside_training_range(self):
        arguments = [
            "--data-dir", "data", "--embedding-dir", "embeddings",
            "--ha-distance-matrix", "distance.npy", "--output-dir", "output",
            "--epochs", "10", "--save-epoch", "11",
        ]
        with self.assertRaises(SystemExit):
            self.module.parse_args(arguments)

    def test_ha_only_frame_validation_does_not_require_na_columns(self):
        frame = pd.DataFrame(
            [{
                "seq_id_a": "ref",
                "seq_id_c": "query",
                "serumHA": "AC-D",
                "virusHA": "ATGD",
                "serumPassCat": "<CELL>",
                "virusPassCat": "<EGG>",
                "label": 2.0,
                "Type": "H3N2",
            }]
        )

        self.module.validate_ha_only_frame(frame, "synthetic.csv")

    def test_align_embedding_preserves_gap_coordinates_and_rejects_truncation(self):
        embedding = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        special_token_embedding = torch.tensor(
            [
                [-1.0, -1.0],
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [-2.0, -2.0],
            ]
        )
        special_aligned, _, special_mask, _ = self.module.align_embedding_to_sequence(
            special_token_embedding,
            "AC-D",
        )
        torch.testing.assert_close(
            special_aligned,
            torch.tensor([[1.0, 10.0], [2.0, 20.0], [0.0, 0.0], [3.0, 30.0]]),
        )
        self.assertEqual(special_mask.tolist(), [1.0, 1.0, 0.0, 1.0])

        aligned, aligned_mask, embedding_mask, residue_ids = (
            self.module.align_embedding_to_sequence(embedding, "AC-D")
        )

        torch.testing.assert_close(
            aligned,
            torch.tensor(
                [[1.0, 10.0], [2.0, 20.0], [0.0, 0.0], [3.0, 30.0]]
            ),
        )
        self.assertEqual(aligned_mask.tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(embedding_mask.tolist(), [1.0, 1.0, 0.0, 1.0])
        self.assertEqual(
            residue_ids.tolist(),
            [
                self.module.AMINO_ACID_TO_ID["A"],
                self.module.AMINO_ACID_TO_ID["C"],
                self.module.GAP_ID,
                self.module.AMINO_ACID_TO_ID["D"],
            ],
        )
        with self.assertRaisesRegex(ValueError, "non-gap"):
            self.module.align_embedding_to_sequence(embedding[:2], "AC-D")

    def test_real_collator_encodes_substitution_insertion_and_empty_query(self):
        frame = pd.DataFrame(
            [
                {
                    "seq_id_a": "ref",
                    "seq_id_c": "sub",
                    "serumHA": "AC-D",
                    "virusHA": "AT-D",
                    "serumPassCat": "<CELL>",
                    "virusPassCat": "<CELL>",
                    "label": 1.0,
                    "Type": "H3N2",
                },
                {
                    "seq_id_a": "ref",
                    "seq_id_c": "ins",
                    "serumHA": "AC-D",
                    "virusHA": "ACGD",
                    "serumPassCat": "<CELL>",
                    "virusPassCat": "<EGG>",
                    "label": 2.0,
                    "Type": "H3N2",
                },
                {
                    "seq_id_a": "ref",
                    "seq_id_c": "same",
                    "serumHA": "AC-D",
                    "virusHA": "AC-D",
                    "serumPassCat": "<CELL>",
                    "virusPassCat": "<CELL>",
                    "label": 0.0,
                    "Type": "H3N2",
                },
            ]
        )
        vocabs = self.module.build_vocabs({"train": frame}, use_subtype_feature=True)
        dataset = self.module.SerumMutationSetTaskDataset(
            frame,
            vocabs,
            task_cols=["seq_id_a", "serumPassCat"],
            max_queries_per_task=8,
        )
        item = dataset[0]
        embeddings = {
            "matrix_ref": torch.randn(3, 2),
            "matrix_sub": torch.randn(3, 2),
            "matrix_ins": torch.randn(4, 2),
            "matrix_same": torch.randn(3, 2),
        }

        batch, _ = self.module.collate_mutation_set_tasks([item], embeddings)

        self.assertIsInstance(batch, self.module.SerumMutationSetBatch)
        reference = batch.reference_aa[:, None].expand_as(batch.query_aa)
        mutation_mask = (
            (batch.reference_aligned_mask[:, None] > 0)
            & (batch.query_aligned_mask > 0)
            & (reference != batch.query_aa)
        )
        self.assertEqual(mutation_mask.sum(dim=-1).tolist(), [[1, 1, 0]])
        self.assertEqual(float(batch.reference_embedding_mask[0, 2]), 0.0)
        self.assertEqual(float(batch.query_embedding_mask[0, 1, 2]), 1.0)

    def test_aligned_embedding_cache_reuses_and_preserves_alignment(self):
        embeddings = {"matrix_ref": torch.randn(3, 2)}
        cache = self.module.AlignedEmbeddingCache()

        first = cache.get(embeddings, "matrix_ref", "AC-D")
        second = cache.get(embeddings, "matrix_ref", "AC-D")

        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)
        self.assertIs(first, second)
        self.assertEqual(first[0].shape, (4, 2))
        self.assertEqual(first[2].tolist(), [1.0, 1.0, 0.0, 1.0])
        uncached = self.module._aligned_embedding(
            embeddings, "matrix_ref", "AC-D", None
        )
        torch.testing.assert_close(first[0], uncached[0])

    def test_distance_loader_preserves_nan_and_rejects_invalid_matrices(self):
        matrix = np.array(
            [[0.0, np.nan, 4.0], [np.nan, 0.0, 2.0], [4.0, 2.0, 0.0]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "distance.npy", matrix)
            np.savez(root / "distance.npz", distance=matrix)
            pd.DataFrame(matrix).to_csv(root / "distance.csv")
            pd.DataFrame(matrix).to_csv(root / "distance.tsv", sep="\t", index=False)
            loaded = [
                self.module.load_ha_distance_matrix(root / "distance.npy"),
                self.module.load_ha_distance_matrix(root / "distance.npz"),
                self.module.load_ha_distance_matrix(root / "distance.csv"),
                self.module.load_ha_distance_matrix(root / "distance.tsv"),
            ]

            for value in loaded:
                self.assertEqual(tuple(value.shape), (3, 3))
                self.assertTrue(torch.isnan(value[0, 1]))
                self.assertEqual(float(value[1, 2]), 2.0)

            np.save(root / "nonsquare.npy", np.zeros((2, 3), dtype=np.float32))
            np.save(root / "negative.npy", np.array([[0.0, -1.0], [-1.0, 0.0]]))
            with self.assertRaisesRegex(ValueError, "square"):
                self.module.load_ha_distance_matrix(root / "nonsquare.npy")
            with self.assertRaisesRegex(ValueError, "non-negative"):
                self.module.load_ha_distance_matrix(root / "negative.npy")


class TrainerCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_trainer()

    def required_arguments(self) -> list[str]:
        return [
            "--data-dir", "/tmp/data",
            "--embedding-dir", "/tmp/embedding",
            "--ha-distance-matrix", "/tmp/distance.npy",
            "--output-dir", "/tmp/output",
        ]

    def test_distance_matrix_is_required(self):
        with self.assertRaises(SystemExit):
            self.module.parse_args(
                [
                    "--data-dir", "/tmp/data",
                    "--embedding-dir", "/tmp/embedding",
                    "--output-dir", "/tmp/output",
                ]
            )

    def test_subtype_filter_and_query_chunk_normalization(self):
        args = self.module.parse_args(
            self.required_arguments()
            + ["--type", "H3N2", "--max-queries-per-task", "0"]
        )

        self.assertFalse(args.use_subtype_feature)
        self.assertEqual(args.subtype_dim, 0)
        self.assertIsNone(args.max_queries_per_task)

    def test_task_chunk_reshuffle_is_deterministic_and_preserves_all_rows(self):
        frame = pd.DataFrame(
            [
                {
                    "seq_id_a": "ref",
                    "seq_id_c": f"query_{index}",
                    "serumHA": "ACD",
                    "virusHA": "ACD",
                    "serumPassCat": "<CELL>",
                    "virusPassCat": "<CELL>",
                    "serumName": "serum",
                    "label": float(index),
                    "Type": "H3N2",
                }
                for index in range(65)
            ]
        )
        vocabs = self.module.build_vocabs({"train": frame}, use_subtype_feature=False)
        dataset = self.module.SerumMutationSetTaskDataset(
            frame,
            vocabs,
            task_cols=["seq_id_a", "serumPassCat", "serumName"],
            max_queries_per_task=32,
        )
        original = [indices.tolist() for _, indices in dataset.task_chunks]

        dataset.reshuffle_queries_within_tasks(123)
        first = [indices.tolist() for _, indices in dataset.task_chunks]
        dataset.reshuffle_queries_within_tasks(123)
        second = [indices.tolist() for _, indices in dataset.task_chunks]

        self.assertEqual([len(indices) for indices in first], [32, 32, 1])
        self.assertEqual(dataset.task_chunk_groups, [("ref||<CELL>||serum", [0, 1, 2])])
        self.assertEqual(first, second)
        self.assertNotEqual(first, original)
        self.assertEqual(sorted(index for chunk in first for index in chunk), list(range(65)))

    def test_epochwise_query_reshuffle_is_opt_in(self):
        default_args = self.module.parse_args(self.required_arguments())
        enabled_args = self.module.parse_args(
            self.required_arguments() + ["--shuffle-queries-within-task-each-epoch"]
        )

        self.assertFalse(default_args.shuffle_queries_within_task_each_epoch)
        self.assertTrue(enabled_args.shuffle_queries_within_task_each_epoch)

    def test_zero_init_film_is_opt_in(self):
        default_args = self.module.parse_args(self.required_arguments())
        enabled_args = self.module.parse_args(
            self.required_arguments() + ["--zero-init-film"]
        )

        self.assertFalse(default_args.zero_init_film)
        self.assertTrue(enabled_args.zero_init_film)

    def test_film_beta_can_be_disabled(self):
        default_args = self.module.parse_args(self.required_arguments())
        disabled_args = self.module.parse_args(
            self.required_arguments() + ["--no-use-film-beta"]
        )

        self.assertTrue(default_args.use_film_beta)
        self.assertFalse(disabled_args.use_film_beta)

    def test_pool_mutation_count_can_be_disabled(self):
        default_args = self.module.parse_args(self.required_arguments())
        disabled_args = self.module.parse_args(
            self.required_arguments() + ["--no-use-pool-mutation-count"]
        )

        self.assertTrue(default_args.use_pool_mutation_count)
        self.assertFalse(disabled_args.use_pool_mutation_count)

    def test_attention_pool_can_be_disabled(self):
        default_args = self.module.parse_args(self.required_arguments())
        disabled_args = self.module.parse_args(
            self.required_arguments() + ["--no-use-attention-pool"]
        )

        self.assertTrue(default_args.use_attention_pool)
        self.assertFalse(disabled_args.use_attention_pool)

    def test_predictor_mutation_count_can_be_disabled(self):
        default_args = self.module.parse_args(self.required_arguments())
        disabled_args = self.module.parse_args(
            self.required_arguments() + ["--no-use-predictor-mutation-count"]
        )

        self.assertTrue(default_args.use_predictor_mutation_count)
        self.assertFalse(disabled_args.use_predictor_mutation_count)

    def test_mutation_transformer_bypass_is_opt_in(self):
        default_args = self.module.parse_args(self.required_arguments())
        bypassed_args = self.module.parse_args(
            self.required_arguments() + ["--bypass-mutation-transformer"]
        )

        self.assertFalse(default_args.bypass_mutation_transformer)
        self.assertTrue(bypassed_args.bypass_mutation_transformer)

    def test_background_to_mutation_is_opt_in(self):
        default_args = self.module.parse_args(self.required_arguments())
        disabled_args = self.module.parse_args(
            self.required_arguments() + ["--no-use-background-to-mutation"]
        )

        self.assertTrue(default_args.use_background_to_mutation)
        self.assertFalse(disabled_args.use_background_to_mutation)

    def test_task_bias_loss_weight_is_validated(self):
        default_args = self.module.parse_args(self.required_arguments())
        enabled_args = self.module.parse_args(
            self.required_arguments() + ["--task-bias-loss-weight", "0.1"]
        )

        self.assertEqual(default_args.task_bias_loss_weight, 0.0)
        self.assertEqual(enabled_args.task_bias_loss_weight, 0.1)
        with self.assertRaises(SystemExit):
            self.module.parse_args(
                self.required_arguments() + ["--task-bias-loss-weight", "-0.1"]
            )

    def test_full_task_bias_loss_is_opt_in_and_requires_positive_weight(self):
        default_args = self.module.parse_args(self.required_arguments())
        enabled_args = self.module.parse_args(
            self.required_arguments()
            + ["--task-bias-loss-weight", "0.1", "--full-task-bias-loss"]
        )

        self.assertFalse(default_args.full_task_bias_loss)
        self.assertTrue(enabled_args.full_task_bias_loss)
        with self.assertRaises(SystemExit):
            self.module.parse_args(
                self.required_arguments() + ["--full-task-bias-loss"]
            )

    def test_full_task_bias_surrogate_has_exact_complete_task_gradient(self):
        first = torch.tensor([0.4, 1.7], requires_grad=True)
        second = torch.tensor([-0.2, 2.3, 0.8], requires_grad=True)
        first_labels = torch.tensor([0.0, 1.0])
        second_labels = torch.tensor([0.5, 1.5, 1.0])
        weight = 0.1

        residuals = torch.cat([first - first_labels, second - second_labels])
        direct_loss = weight * residuals.mean().square()
        direct_loss.backward()
        expected_first = first.grad.clone()
        expected_second = second.grad.clone()

        first.grad = None
        second.grad = None
        full_bias = residuals.detach().mean()
        surrogate = self.module._full_task_bias_gradient_term(
            (first - first_labels).sum(),
            full_bias,
            5,
            weight,
        ) + self.module._full_task_bias_gradient_term(
            (second - second_labels).sum(),
            full_bias,
            5,
            weight,
        )
        surrogate.backward()

        torch.testing.assert_close(first.grad, expected_first)
        torch.testing.assert_close(second.grad, expected_second)

    def test_direct_background_requires_matching_site_and_background_dims(self):
        default_args = self.module.parse_args(self.required_arguments())
        direct_args = self.module.parse_args(
            self.required_arguments()
            + ["--direct-background", "--site-dim", "64", "--background-dim", "64"]
        )

        self.assertFalse(default_args.direct_background)
        self.assertTrue(direct_args.direct_background)
        with self.assertRaises(SystemExit):
            self.module.parse_args(
                self.required_arguments()
                + ["--direct-background", "--site-dim", "64", "--background-dim", "128"]
            )

    def test_site_bottleneck_dimension_is_opt_in_and_non_negative(self):
        default_args = self.module.parse_args(self.required_arguments())
        bottleneck_args = self.module.parse_args(
            self.required_arguments() + ["--site-bottleneck-dim", "32"]
        )

        self.assertEqual(default_args.site_bottleneck_dim, 0)
        self.assertEqual(bottleneck_args.site_bottleneck_dim, 32)
        with self.assertRaises(SystemExit):
            self.module.parse_args(
                self.required_arguments() + ["--site-bottleneck-dim", "-1"]
            )

    def test_cli_has_no_na_branch_dependency(self):
        with self.assertRaises(SystemExit):
            self.module.parse_args(self.required_arguments() + ["--na-branch", "pair"])


class TrainerEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_trainer()

    def test_one_epoch_cpu_run_writes_expected_artifacts(self):
        rows = [
            {
                "seq_id_a": "ref_h1",
                "seq_id_c": "query_h1",
                "serumHA": "ACDG",
                "virusHA": "ATDG",
                "serumPassCat": "<CELL>",
                "virusPassCat": "<CELL>",
                "serumName": "serum_h1",
                "label": 1.0,
                "Type": "H1N1",
            },
            {
                "seq_id_a": "ref_h3",
                "seq_id_c": "query_h3",
                "serumHA": "ACDG",
                "virusHA": "ACNG",
                "serumPassCat": "<EGG>",
                "virusPassCat": "<CELL>",
                "serumName": "serum_h3",
                "label": 2.0,
                "Type": "H3N2",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            embedding_dir = root / "embeddings"
            output_dir = root / "output"
            data_dir.mkdir()
            embedding_dir.mkdir()
            for split in ("train", "valid", "test"):
                pd.DataFrame(rows).to_csv(data_dir / f"{split}.csv", index=False)
            torch.manual_seed(19)
            for sequence_id in ("ref_h1", "query_h1", "ref_h3", "query_h3"):
                torch.save(torch.randn(4, 4), embedding_dir / f"matrix_{sequence_id}.pt")
            distance_path = root / "distance.npy"
            np.save(
                distance_path,
                np.array(
                    [
                        [0.0, 2.0, 6.0, np.nan],
                        [2.0, 0.0, 3.0, 8.0],
                        [6.0, 3.0, 0.0, 4.0],
                        [np.nan, 8.0, 4.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
            args = self.module.parse_args(
                [
                    "--data-dir", str(data_dir),
                    "--embedding-dir", str(embedding_dir),
                    "--ha-distance-matrix", str(distance_path),
                    "--output-dir", str(output_dir),
                    "--epochs", "1",
                    "--site-dim", "4",
                    "--background-dim", "8",
                    "--mutation-dim", "8",
                    "--position-dim", "4",
                    "--amino-acid-dim", "3",
                    "--presence-dim", "2",
                    "--theta-dim", "8",
                    "--passage-dim", "2",
                    "--subtype-dim", "2",
                    "--mutation-attention-heads", "2",
                    "--mutation-ffn-dim", "16",
                    "--predictor-hidden-dim", "12",
                    "--attention-dropout", "0",
                    "--predictor-dropout", "0",
                    "--device", "cpu",
                    "--no-progress",
                ]
            )
            result = self.module.run_training(args)

            self.assertTrue((output_dir / "checkpoints" / "best_model.pth").is_file())
            self.assertTrue((output_dir / "predictions_test.csv").is_file())
            run_config = __import__("json").loads(
                (output_dir / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_config["training_config"]["max_queries_per_task"], 128)
            self.assertEqual(run_config["training_config"]["learning_rate"], 1e-4)
            self.assertTrue((output_dir / "metrics.csv").is_file())
            subtype_metrics = pd.read_csv(output_dir / "metrics_by_subtype.csv")
            self.assertIn("test_h1n1_pooled_mae", subtype_metrics.columns)
            self.assertIn("test_h3n2_pooled_mae", subtype_metrics.columns)
            config_text = (output_dir / "run_config.json").read_text(encoding="utf-8")
            self.assertIn("\"model\": \"SerumMutationSet-Minus\"", config_text)
            self.assertIn("best", result)


if __name__ == "__main__":
    unittest.main()
