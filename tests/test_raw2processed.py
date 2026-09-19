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
SCRIPT_PATH = REPO_ROOT / "src" / "fluprofiler" / "dataset" / "raw2processed.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("raw2processed", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Raw2ProcessedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_writes_pending_fasta_for_sequences_missing_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "data" / "dataset" / "unit"
            raw_dir = dataset_dir / "raw"
            raw_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "seq_a": "HA_EXISTING",
                        "seq_b": "NA_EXISTING",
                        "seq_c": "HA_NEW",
                        "seq_d": "NA_EXISTING",
                        "serumPassCat": "<NONE>",
                        "virusPassCat": "<NONE>",
                        "label": 2.0,
                        "serumType": "H1N1",
                        "virusType": "H1N1",
                    }
                ]
            ).to_csv(raw_dir / "data4model(unit).csv", index=False)

            registry = root / "data" / "embedding" / "registry" / "sequences.csv"
            registry.parent.mkdir(parents=True)
            files_dir = root / "data" / "embedding" / "files"
            files_dir.mkdir(parents=True)
            (files_dir / "matrix_HA_1.pt").write_bytes(b"ha")
            (files_dir / "matrix_NA_1.pt").write_bytes(b"na")
            pd.DataFrame(
                [
                    {
                        "seq_id": "HA_1",
                        "segment": "HA",
                        "sequence_hash": self.module.sequence_hash("HA_EXISTING"),
                        "sequence": "HA_EXISTING",
                        "embedding_path": str(files_dir / "matrix_HA_1.pt"),
                        "embedding_exists": "true",
                    },
                    {
                        "seq_id": "NA_1",
                        "segment": "NA",
                        "sequence_hash": self.module.sequence_hash("NA_EXISTING"),
                        "sequence": "NA_EXISTING",
                        "embedding_path": str(files_dir / "matrix_NA_1.pt"),
                        "embedding_exists": "true",
                    },
                ]
            ).to_csv(registry, index=False)

            pending_dir = root / "data" / "embedding" / "registry" / "pending"
            argv = [
                "raw2processed.py",
                "--dataset-dir",
                str(dataset_dir),
                "--registry-csv",
                str(registry),
                "--embedding-files-dir",
                str(files_dir),
                "--pending-dir",
                str(pending_dir),
                "--timestamp",
                "20260707_163200",
            ]
            with patch.object(sys, "argv", argv):
                self.module.main()

            pending = pending_dir / "20260707_163200.fasta"
            self.assertEqual(pending.read_text(encoding="utf-8"), ">HA_2\nHA_NEW\n")

            source_config = json.loads(
                (dataset_dir / "processed" / "source_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(source_config["missing_embedding_count"], 1)
            self.assertEqual(source_config["pending_fasta"], str(pending))

    def test_only_reads_data4model_parenthesized_csv_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "data" / "dataset" / "unit"
            raw_dir = dataset_dir / "raw"
            raw_dir.mkdir(parents=True)
            keep_row = {
                "seq_a": "HA_SERUM",
                "seq_b": "NA_SERUM",
                "seq_c": "HA_KEEP",
                "seq_d": "NA_VIRUS",
                "serumPassCat": "<NONE>",
                "virusPassCat": "<NONE>",
                "label": 2.0,
                "serumType": "H1N1",
                "virusType": "H1N1",
            }
            ignored_row = dict(keep_row, seq_c="HA_BACKUP", label=4.0)
            pd.DataFrame([keep_row]).to_csv(raw_dir / "data4model(keep).csv", index=False)
            pd.DataFrame([ignored_row]).to_csv(raw_dir / "data4model(keep)_back.csv", index=False)

            registry = root / "data" / "embedding" / "registry" / "sequences.csv"
            files_dir = root / "data" / "embedding" / "files"
            pending_dir = root / "data" / "embedding" / "registry" / "pending"
            argv = [
                "raw2processed.py",
                "--dataset-dir",
                str(dataset_dir),
                "--registry-csv",
                str(registry),
                "--embedding-files-dir",
                str(files_dir),
                "--pending-dir",
                str(pending_dir),
                "--timestamp",
                "20260707_170000",
            ]
            with patch.object(sys, "argv", argv):
                self.module.main()

            source = pd.read_csv(dataset_dir / "processed" / "source.csv")
            self.assertEqual(source["seq_c"].tolist(), ["HA_KEEP"])

            source_config = json.loads(
                (dataset_dir / "processed" / "source_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(source_config["input_files"], ["data4model(keep).csv"])
            self.assertEqual(source_config["rows_input"], 1)


if __name__ == "__main__":
    unittest.main()
