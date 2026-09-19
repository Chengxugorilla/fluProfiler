import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_antigen_space_targets_vs_vaccines.py"


def load_module():
    spec = importlib.util.spec_from_file_location("plot_antigen_space_targets_vs_vaccines", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedding_id_prefers_existing_epi_candidate(tmp_path):
    module = load_module()
    embedding_dir = tmp_path / "embedding"
    embedding_dir.mkdir()
    (embedding_dir / "matrix_EPI4496474.pt").write_bytes(b"")

    record_id = "A/Missouri/11/2025|EPI_ISL_20066835|HA|EPI4496474"

    assert module.embedding_id(record_id, embedding_dir) == "EPI4496474"


def test_build_plot_rows_keeps_target_label_for_duplicate_vaccine(tmp_path):
    module = load_module()
    embedding_dir = tmp_path / "embedding"
    embedding_dir.mkdir()
    for seq_id in ["EPI241025", "EPI4496474", "EPI123"]:
        (embedding_dir / f"matrix_{seq_id}.pt").write_bytes(b"")

    target_fasta = tmp_path / "targets.fasta"
    target_fasta.write_text(
        ">A/New Jersey/11/1976|EPI241025\n"
        "ACDE\n"
        ">A/Missouri/11/2025|EPI4496474\n"
        "FGHI\n",
        encoding="utf-8",
    )
    vaccine_fasta = tmp_path / "vaccines.fasta"
    vaccine_fasta.write_text(
        ">A/Missouri/11/2025|EPI4496474\n"
        "FGHI\n"
        ">Historical vaccine|EPI123\n"
        "KLMN\n",
        encoding="utf-8",
    )

    rows = module.build_plot_rows(target_fasta, vaccine_fasta, embedding_dir)

    assert rows == [
        {
            "seq_id": "EPI241025",
            "plot_label": "1976 New Jersey",
            "group": "target",
            "record_id": "A/New Jersey/11/1976|EPI241025",
        },
        {
            "seq_id": "EPI4496474",
            "plot_label": "2025 Missouri / 2026 vaccine",
            "group": "target",
            "record_id": "A/Missouri/11/2025|EPI4496474",
        },
        {
            "seq_id": "EPI123",
            "plot_label": "EPI123",
            "group": "vaccine",
            "record_id": "Historical vaccine|EPI123",
        },
    ]
