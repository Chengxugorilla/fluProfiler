from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_metric_artificial_data.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_metric_artificial_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row():
    return {
        "seq_id_a": "ha_serum",
        "seq_id_b": "na_serum",
        "seq_id_c": "ha_virus",
        "seq_id_d": "na_virus",
        "seq_a": "HA_A",
        "seq_b": "NA_A",
        "seq_c": "HA_C",
        "seq_d": "NA_C",
        "serumPassCat": "<EGG>",
        "virusPassCat": "<CELL>",
        "serumName": "serum_name",
        "virusName": "virus_name",
        "serumDate": "2001-01-01",
        "virusDate": "2002-02-02",
        "serumIslID": "serum_isl",
        "virusIslID": "virus_isl",
        "serumHA": "SERUM_HA_ALN",
        "virusHA": "VIRUS_HA_ALN",
        "Type": "H3N2",
        "label": 4.0,
    }


class BuildMetricArtificialDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_script_module()

    def test_build_artificial_frame_creates_reverse_pair_rows(self):
        train_frame = pd.DataFrame([_row()])

        artificial_frame = self.module.build_artificial_frame(train_frame, method="reverse_pair")
        row = artificial_frame.iloc[0]

        self.assertEqual(len(artificial_frame), 1)
        self.assertEqual(row["seq_id_a"], "ha_virus")
        self.assertEqual(row["seq_id_c"], "ha_serum")
        self.assertEqual(row["serumPassCat"], "<CELL>")
        self.assertEqual(row["virusPassCat"], "<EGG>")
        self.assertEqual(row["label"], 4.0)
        self.assertTrue(row["is_artificial"])
        self.assertTrue(row["is_reverse_artificial"])
        self.assertEqual(row["artificial_method"], "reverse_pair")


if __name__ == "__main__":
    unittest.main()
