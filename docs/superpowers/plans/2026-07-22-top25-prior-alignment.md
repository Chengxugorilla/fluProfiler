# Top-25 Prior Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fully code-driven notebook section that compares the combined mean-absolute-attribution Top-25 with overlapping labels read from `paper/data/ABCDE.csv`.

**Architecture:** The new final notebook code cell consumes `combined_summary` from the existing global-attribution cell and reads the CSV directly. It validates the label schema, creates a 329-position annotation table with independent epitope/RBS/glycosylation flags, then derives all Top-25 hit and aggregate statistics from that table. It displays results inline with Pandas and Matplotlib only.

**Tech Stack:** Python 3, Pandas, NumPy, SciPy (`fisher_exact`), Matplotlib, Jupyter/nbformat.

## Global Constraints

- Modify only `/home/chenyh/workspace/fluProfiler/paper/Code/gradient_x_input_analysis.ipynb` for the analysis implementation.
- Read labels from `/home/chenyh/workspace/fluProfiler/paper/data/ABCDE.csv`; do not encode site hits manually.
- Use `combined_summary['mean_abs']` for ranking and choose the descending Top-25 HA1 positions.
- Treat `Antigenic epitope`, `Receptor binding site`, and `glycosylation site` as overlapping labels.
- Do not add biological interpretation; output only data-derived labels, counts, enrichment statistics, ranks, and attribution summaries.
- Show figures inline with `plt.show()` and immediately `plt.close(fig)`; do not call `savefig`, `to_csv`, `to_excel`, or write image files.
- Avoid Seaborn and preserve all existing cells and results.

---

### Task 1: Add a regression test for the Top-25 label-alignment contract

**Files:**
- Create: `/home/chenyh/workspace/fluProfiler/tests/test_top25_prior_alignment_contract.py`
- Test: `/home/chenyh/workspace/fluProfiler/tests/test_top25_prior_alignment_contract.py`

**Interfaces:**
- Consumes: `/home/chenyh/workspace/fluProfiler/paper/data/ABCDE.csv` and `/home/chenyh/workspace/fluProfiler/paper/Code/gradient_x_input_analysis.ipynb`.
- Produces: Assertions specifying that the notebook's final cell reads the CSV, uses `combined_summary`, creates `site_annotations`, `top25_annotations`, and `prior_alignment_summary`, calls `fisher_exact`, and does not save figures/files.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n fluProfiler python -m pytest tests/test_top25_prior_alignment_contract.py -q`

Expected: FAIL because the current final notebook cell does not contain the new analysis.

- [ ] **Step 3: Write minimal implementation**

Append one markdown cell titled `## 4. Top-25 与 CSV 先验标签的代码对齐` and one code cell. The code cell must:

```python
from pathlib import Path
from scipy.stats import fisher_exact

prior_csv_path = PROJECT_ROOT / "paper/data/ABCDE.csv"
prior_raw = pd.read_csv(prior_csv_path)
required_columns = [
    "Amino acid residues of HA1",
    "Antigenic epitope",
    "Receptor binding site",
    "glycosylation site",
]
assert list(prior_raw.columns) == required_columns
assert prior_raw[required_columns[0]].is_unique

top25_positions = combined_summary.sort_values("mean_abs", ascending=False).head(25).index.astype(int)
```

Build `site_annotations` for all 329 positions; map CSV epitope strings and `√` flags with `reindex`, preserve independent Boolean columns, join combined rank/mean-absolute attribution, create `top25_annotations`, compute groups `Any epitope`, `Epitope A` through `Epitope E`, `Epitope Un`, `RBS`, and `Glycosylation`. For each group, construct the 2x2 count table from membership versus Top-25 status, call `fisher_exact(table, alternative='greater')`, and store hit count, group size, Top-25 rate, background rate, odds ratio, p-value, group/complement mean attribution, and group/complement mean rank in `prior_alignment_summary`. Display the tables and one Matplotlib horizontal bar chart of Top-25 mean-absolute attribution, epitope color, and independent RBS/glycosylation marker layers.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n fluProfiler python -m pytest tests/test_top25_prior_alignment_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add paper/Code/gradient_x_input_analysis.ipynb tests/test_top25_prior_alignment_contract.py
git commit -m "feat: add Top-25 prior label alignment"
```

### Task 2: Execute and verify the notebook analysis end-to-end

**Files:**
- Modify: `/home/chenyh/workspace/fluProfiler/paper/Code/gradient_x_input_analysis.ipynb`
- Test: `/home/chenyh/workspace/fluProfiler/tests/test_top25_prior_alignment_contract.py`

**Interfaces:**
- Consumes: final notebook with the alignment code cell and the input artifacts referenced by its initial setup cell.
- Produces: an executed notebook with displayed tables and an embedded inline figure, plus no outputs written outside the notebook.

- [ ] **Step 1: Write the failing execution check**

Add these runtime assertions to the new analysis code cell before displaying results:

```python
assert len(top25_positions) == 25
assert top25_annotations.index.is_unique
assert set(top25_annotations.index) == set(top25_positions)
assert prior_alignment_summary["top25_hits"].between(0, 25).all()
assert site_annotations[["is_rbs", "is_glycosylation"]].dtypes.eq(bool).all()
```

- [ ] **Step 2: Run notebook execution to verify it fails before the cell implementation**

Run the project kernel-execution helper used for this notebook and inspect errors.

Expected: prior to Task 1 implementation, the new objects and assertions are absent, so the contract cannot be satisfied.

- [ ] **Step 3: Execute the implemented notebook**

Run all notebook cells with the `fluProfiler` environment kernel. Preserve only notebook outputs, including tables and PNG display data generated by `plt.show()`.

- [ ] **Step 4: Run verification commands**

Run:

```bash
conda run -n fluProfiler python -m pytest tests/test_top25_prior_alignment_contract.py tests/test_run_gradient_x_input.py -q
conda run -n fluProfiler python - <<'PY'
from pathlib import Path
import nbformat

path = Path('paper/Code/gradient_x_input_analysis.ipynb')
nb = nbformat.read(path, as_version=4)
assert not any(
    output.output_type == 'error'
    for cell in nb.cells if cell.cell_type == 'code'
    for output in cell.get('outputs', [])
)
source = nb.cells[-1].source
assert 'plt.show()' in source and 'plt.close(fig)' in source
assert 'savefig' not in source and 'to_csv' not in source
print({'cells': len(nb.cells), 'last_cell_outputs': len(nb.cells[-1].get('outputs', []))})
PY
```

Expected: pytest passes; the executed notebook has no error outputs, its final cell has displayed output, and no file/image saving statements.

- [ ] **Step 5: Commit executed notebook and contract test**

```bash
git add paper/Code/gradient_x_input_analysis.ipynb tests/test_top25_prior_alignment_contract.py
git commit -m "feat: add Top-25 prior label alignment"
```
