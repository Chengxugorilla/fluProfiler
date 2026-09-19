from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "serum_gate" / "train_serum_mutation_set_name_residual.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("train_serum_mutation_set_name_residual", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NameResidualTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_trainer()

    def test_name_vocabs_use_train_only_and_normalize_names(self):
        train = pd.DataFrame({
            "serumName": [" A/Example/1/2020 ", None],
            "virusName": ["A/Virus  1/2020", "A/Virus/2/2020"],
        })
        test = pd.DataFrame({
            "serumName": ["a/example/1/2020", "A/UNSEEN/2020"],
            "virusName": ["a/virus 1/2020", "A/UNSEEN/2020"],
        })

        vocabs = self.module.build_name_vocabs(train)

        self.assertEqual(vocabs.serum_name_to_id["<UNK>"], 0)
        self.assertEqual(vocabs.serum_name_to_id["A/EXAMPLE/1/2020"], 1)
        self.assertNotIn("A/UNSEEN/2020", vocabs.serum_name_to_id)
        self.assertEqual(self.module.name_id("A/UNSEEN/2020", vocabs.serum_name_to_id), 0)
        self.assertEqual(self.module.name_id(test.loc[0, "virusName"], vocabs.virus_name_to_id), 1)

    def test_cli_defaults_follow_confirmed_regularization_ratio(self):
        args = self.module.parse_args([
            "--data-dir", "data", "--embedding-dir", "embeddings",
            "--ha-distance-matrix", "distance.npy", "--output-dir", "output",
        ])
        self.assertEqual(args.serum_name_l2, 1e-3)
        self.assertEqual(args.virus_name_l2, 1e-2)


if __name__ == "__main__":
    unittest.main()
