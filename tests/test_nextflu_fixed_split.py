import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "benchmark_pairwise" / "train_nextflu_fixed_split.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_nextflu_fixed_split", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(subtype: str, serum_name: str, virus_name: str, label: float):
    return {
        "serumHA": "AAAA",
        "virusHA": "AAAT",
        "serumName": serum_name,
        "virusName": virus_name,
        "Type": subtype,
        "label": label,
    }


def test_run_nextflu_fixed_split_writes_predictions_models_and_metrics():
    module = _load_module()
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "split"
        output_dir = root / "out"
        data_dir.mkdir()

        train = pd.DataFrame(
            [
                _row("H1N1", "h1_serum", "h1_virus", 1.0),
                _row("H3N2", "h3_serum", "h3_virus", 2.0),
            ]
        )
        test = pd.DataFrame(
            [
                _row("H1N1", "h1_serum", "h1_virus", 1.0),
                _row("H3N2", "h3_serum", "h3_virus", 2.0),
            ]
        )
        valid = test.copy()
        train.to_csv(data_dir / "train.csv", index=False)
        valid.to_csv(data_dir / "valid.csv", index=False)
        test.to_csv(data_dir / "test.csv", index=False)

        module.run(data_dir=data_dir, output_dir=output_dir, subtypes=["H1N1", "H3N2"])

        predictions = pd.read_csv(output_dir / "predictions.csv")
        assert {"pred_with_name", "pred_without_name", "reference"}.issubset(predictions.columns)
        assert len(predictions) == 2
        assert (output_dir / "H1N1_model.json").is_file()
        assert (output_dir / "H3N2_model.json").is_file()
        assert (output_dir / "metrics.json").is_file()
        assert (output_dir / "run_config.json").is_file()


def test_run_nextflu_can_merge_valid_into_train():
    module = _load_module()
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "split"
        output_dir = root / "out"
        data_dir.mkdir()

        train = pd.DataFrame([_row("H1N1", "h1_serum", "h1_virus", 1.0)])
        valid = pd.DataFrame([_row("H1N1", "h1_serum_valid", "h1_virus_valid", 2.0)])
        test = pd.DataFrame([_row("H1N1", "h1_serum", "h1_virus", 1.0)])
        train.to_csv(data_dir / "train.csv", index=False)
        valid.to_csv(data_dir / "valid.csv", index=False)
        test.to_csv(data_dir / "test.csv", index=False)

        module.run(
            data_dir=data_dir,
            output_dir=output_dir,
            subtypes=["H1N1"],
            merge_valid_into_train=True,
        )

        run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
        assert run_config["merge_valid_into_train"] is True
        assert run_config["input_row_counts"] == {"train": 1, "valid": 1, "test": 1}
        assert run_config["training_row_count"] == 2


def test_resolve_split_data_dir_appends_test_season():
    module = _load_module()
    root = Path("data/dataset/H1H3_new/splited/v1/H1H3_new/season")

    resolved = module.resolve_split_data_dir(root, "26")

    assert resolved == root / "26"
    assert module.resolve_split_data_dir(root / "26", "26") == root / "26"
