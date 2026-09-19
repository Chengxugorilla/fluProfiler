import importlib
import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "predict_test_csv.py"


def load_module():
    spec = importlib.util.spec_from_file_location("predict_test_csv", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_configures_legacy_model_import_path_for_pickled_checkpoints():
    module = load_module()
    legacy_path = str(REPO_ROOT / "src" / "fluprofiler")
    sys.path = [path for path in sys.path if path != legacy_path]
    sys.modules.pop("models", None)
    sys.modules.pop("models.architectures", None)

    module.configure_pickle_import_paths()

    architectures = importlib.import_module("models.architectures")
    assert hasattr(architectures, "fluProfiler_Config")


def test_predict_one_pair_supports_v2_batch_input_contract():
    module = load_module()

    class V2Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_batch = None

        def forward(self, batch):
            self.seen_batch = batch
            return module.ModelOutput(logits=torch.tensor([[2.5]]), pred=torch.tensor([2.5]))

    model = V2Model()
    ma = torch.ones(1, 2, 3)
    mc = torch.full((1, 2, 3), 2.0)
    mask = torch.ones(1, 2)
    passage = torch.tensor([[0, 2, 2, 2, 1]])

    pred = module.predict_one_pair(
        model=model,
        matrices={"serum_HA": ma, "virus_HA": mc},
        matrix_masks={"serum_HA": mask, "virus_HA": mask},
        passage_tokens=passage,
    )

    assert pred == 2.5
    assert model.seen_batch.matrices["serum_HA"] is ma
    assert model.seen_batch.matrices["virus_HA"] is mc
    assert model.seen_batch.passage_tokens is passage
