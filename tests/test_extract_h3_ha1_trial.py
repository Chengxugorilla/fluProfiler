from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "extract_h3_ha1_trial.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("extract_h3_ha1_trial", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExtractH3Ha1TrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_trims_signal_peptide_before_relaxed_ha1_start_motif(self):
        mature_ha1 = "QKLPGNNN" + ("A" * 313) + "NVPEKQTR"
        sequence = "ALSYILCLVFA" + mature_ha1 + "GIFGAIAGFI"

        ha1, info = self.module.extract_h3_ha1(sequence)

        self.assertEqual(ha1, mature_ha1)
        self.assertEqual(info["output_len"], 329)
        self.assertEqual(info["start_trim"], 11)


if __name__ == "__main__":
    unittest.main()
