from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_dataset_processed.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_dataset_processed", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(label: float | str = 1.0, seq_a: str = "HA_EXISTING", seq_b: str = "NA_EXISTING"):
    return {
        "serumName": "serum",
        "serumPassCat": None,
        "serumDate": "2024-01-01",
        "serumType": "H1N1",
        "virusName": "virus",
        "virusPassCat": "<EGG>",
        "virusDate": "2024-02-01",
        "virusType": "H1N1",
        "sheet": "1-1",
        "serumIslID": "serum_isl",
        "virusIslID": "virus_isl",
        "label": label,
        "seq_a": seq_a,
        "seq_b": seq_b,
        "seq_c": "HA_NEW",
        "seq_d": "NA_EXISTING",
    }


class PrepareDatasetProcessedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_prepare_dataset_assigns_ids_updates_registry_and_writes_pending_fasta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "data" / "dataset" / "unit"
            raw_dir = dataset_dir / "raw"
            raw_dir.mkdir(parents=True)
            metadata_missing = _row(label=2.0)
            metadata_missing["serumDate"] = None
            bad_label = _row(label="bad")
            missing_sequence = _row(label=3.0, seq_a=" ")
            pd.DataFrame([_row(), metadata_missing, bad_label, missing_sequence]).to_csv(
                raw_dir / "part.csv", index=False
            )

            registry = root / "data" / "embedding" / "registry" / "sequences.csv"
            registry.parent.mkdir(parents=True)
            with registry.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "seq_id",
                        "segment",
                        "sequence_hash",
                        "sequence",
                        "embedding_path",
                        "embedding_exists",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "seq_id": "HA_1",
                        "segment": "HA",
                        "sequence_hash": self.module.sequence_hash("HA_EXISTING"),
                        "sequence": "HA_EXISTING",
                        "embedding_path": str(root / "data" / "embedding" / "files" / "matrix_HA_1.pt"),
                        "embedding_exists": "true",
                    }
                )
                writer.writerow(
                    {
                        "seq_id": "NA_1",
                        "segment": "NA",
                        "sequence_hash": self.module.sequence_hash("NA_EXISTING"),
                        "sequence": "NA_EXISTING",
                        "embedding_path": str(root / "data" / "embedding" / "files" / "matrix_NA_1.pt"),
                        "embedding_exists": "true",
                    }
                )
            files_dir = root / "data" / "embedding" / "files"
            files_dir.mkdir()
            (files_dir / "matrix_HA_1.pt").write_bytes(b"ha")
            (files_dir / "matrix_NA_1.pt").write_bytes(b"na")
            aligned_fasta = root / "HA_aligned.fasta"
            aligned_fasta.write_text(">HA_1\nHA-EXISTING\n>HA_2\nHA-NEW\n", encoding="utf-8")

            argv = [
                "prepare_dataset_processed.py",
                "--dataset-dir",
                str(dataset_dir),
                "--registry-csv",
                str(registry),
                "--embedding-files-dir",
                str(files_dir),
                "--pending-dir",
                str(root / "data" / "embedding" / "registry" / "pending"),
                "--aligned-ha-fasta",
                str(aligned_fasta),
                "--timestamp",
                "20260616_210000",
            ]
            with patch("sys.argv", argv):
                self.module.main()

            processed = pd.read_csv(dataset_dir / "processed" / "source.csv")
            self.assertEqual(processed["seq_id_a"].tolist(), ["HA_1", "HA_1"])
            self.assertEqual(processed["seq_id_c"].tolist(), ["HA_2", "HA_2"])
            self.assertEqual(processed["seq_id_b"].tolist(), ["NA_1", "NA_1"])
            self.assertEqual(processed["serumPassCat"].tolist(), ["<NONE>", "<NONE>"])
            self.assertEqual(processed["serumHA"].tolist(), ["HA-EXISTING", "HA-EXISTING"])
            self.assertEqual(processed["virusHA"].tolist(), ["HA-NEW", "HA-NEW"])

            updated_registry = pd.read_csv(registry)
            new_row = updated_registry[updated_registry["seq_id"] == "HA_2"].iloc[0]
            self.assertEqual(new_row["segment"], "HA")
            self.assertEqual(new_row["embedding_exists"], False)

            pending = root / "data" / "embedding" / "registry" / "pending" / "20260616_210000.fasta"
            self.assertEqual(pending.read_text(encoding="utf-8"), ">HA_2\nHA_NEW\n")

            qc = json.loads((dataset_dir / "processed" / "qc_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(qc["rows_input"], 4)
            self.assertEqual(qc["rows_output"], 2)
            self.assertEqual(qc["rows_dropped"], 2)
            self.assertEqual(qc["new_sequences"]["HA"], 1)
            self.assertEqual(qc["missing_embedding_count"], 1)
            self.assertEqual(qc["pending_fasta"], str(pending))
            steps = {step["name"]: step for step in qc["steps"]}
            self.assertEqual(steps["drop_missing_label"]["rows_removed"], 1)
            self.assertEqual(steps["drop_missing_sequences"]["rows_removed"], 1)
            self.assertEqual(steps["normalize_passage"]["details"]["serumPassCat_changed"], 2)
            self.assertEqual(steps["assign_sequence_ids"]["details"]["new_sequences"]["HA"], 1)
            self.assertEqual(steps["write_pending_fasta"]["details"]["missing_embedding_count"], 1)


if __name__ == "__main__":
    unittest.main()
