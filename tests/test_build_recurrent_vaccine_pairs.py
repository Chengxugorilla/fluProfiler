import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_recurrent_vaccine_pairs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_recurrent_vaccine_pairs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_fasta_ids_uses_pipe_policy(tmp_path):
    fasta = tmp_path / "sample.fasta"
    fasta.write_text(
        ">REC001 extra metadata\n"
        "ACDE\n"
        ">EPI123|EPI_ISL_456\n"
        "FGHI\n",
        encoding="utf-8",
    )
    module = load_module()

    assert module.parse_fasta_ids(fasta, pipe_part="first") == ["REC001", "EPI123"]
    assert module.parse_fasta_ids(fasta, pipe_part="second") == ["REC001", "EPI_ISL_456"]


def test_build_pairs_places_recurrent_as_serum_and_vaccine_as_virus():
    module = load_module()

    rows = module.build_pair_rows(["r1", "r2"], ["v1"], passage="<EGG>")

    assert rows == [
        {"seq_id_a": "r1", "seq_id_c": "v1", "serumPassCat": "<EGG>", "virusPassCat": "<EGG>"},
        {"seq_id_a": "r2", "seq_id_c": "v1", "serumPassCat": "<EGG>", "virusPassCat": "<EGG>"},
    ]
