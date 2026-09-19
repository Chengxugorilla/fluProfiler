from pathlib import Path

import nbformat


NOTEBOOK = Path(__file__).parents[1] / "paper/Code/gradient_x_input_analysis.ipynb"


def test_notebook_has_code_driven_top25_prior_alignment_cell():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = notebook.cells[-1].source

    assert "ABCDE.csv" in source
    assert "combined_summary" in source
    assert "site_annotations" in source
    assert "top25_annotations" in source
    assert "prior_alignment_summary" in source
    assert "fisher_exact" in source
    assert "savefig" not in source
    assert "to_csv" not in source
