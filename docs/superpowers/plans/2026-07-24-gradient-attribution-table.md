# Gradient Attribution Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mixed exploratory notebook with a small, reproducible pipeline that produces `attribution_with_dates` and query-site summaries.

**Architecture:** The notebook will have a single data path: derive stable records from `whole.csv`, load per-sample attribution, then merge them through the existing five-key one-to-one join. One summary helper will consume that merged table and return ranked mean absolute query attribution by token position.

**Tech Stack:** Python, pandas, NumPy, Jupyter Notebook, `fluProfiler` Conda environment.

## Global Constraints

- Modify only `paper/Code/gradient_x_input_analysis.ipynb` plus this plan and the approved design document.
- Run project Python with `conda run -n fluProfiler`.
- Preserve the H3N2 filter, 329-amino-acid rule, deduplication keys, record order, and five merge keys.
- Preserve `validate="one_to_one"` in the metadata merge.
- Do not retain NPZ/metrics/Top-25/prior-label/plot workflows.

---

### Task 1: Replace the notebook with the focused analysis pipeline

**Files:**

- Modify: `paper/Code/gradient_x_input_analysis.ipynb`

**Interfaces:**

- Consumes: `whole.csv` and `attribution_by_sample.csv` at the configured absolute paths.
- Produces: `processed`, `record_dates`, `attribution`, `attribution_with_dates`, and `query_site_summary` in notebook scope.
- Defines:
  - `build_records(whole_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `load_attributions(attribution_csv: Path, processed: pd.DataFrame) -> pd.DataFrame`
  - `build_analysis_table(attribution: pd.DataFrame, record_dates: pd.DataFrame) -> pd.DataFrame`
  - `summarize_query_sites(table: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Write the focused notebook cells**

Create five cells: title/intent, imports and paths, helper definitions, table construction/integrity checks, and the serum-filtered summary example. The merge helper must contain:

```python
def build_analysis_table(attribution, record_dates):
    return attribution.merge(
        record_dates,
        on=["sample_index", "seq_id_a", "seq_id_c", "serumName", "virusName"],
        how="left",
        validate="one_to_one",
    )
```

The summary helper must select `query_token_*` columns, calculate their absolute column means, extract the numeric suffix as `aa_position`, and sort by that position.

- [ ] **Step 2: Add explicit notebook integrity checks**

After constructing `attribution_with_dates`, assert that its row count equals the attribution row count, that `sample_index` preserves its order, and that the merged `serumDate` and `virusDate` are not missing. In the example, assert the summary has 329 rows and positions 1–329.

- [ ] **Step 3: Validate notebook structure without execution**

Run:

```bash
conda run -n fluProfiler python -c "import nbformat; p='paper/Code/gradient_x_input_analysis.ipynb'; nb=nbformat.read(p, as_version=4); assert len(nb.cells)==5; assert 'attribution_with_dates = attribution.merge(' in ''.join(nb.cells[3].source); assert 'validate=\"one_to_one\"' in ''.join(nb.cells[3].source)"
```

Expected: exit code 0.

- [ ] **Step 4: Execute notebook code in order**

Run the notebook using `nbconvert --execute` with the `fluProfiler` kernel:

```bash
conda run -n fluProfiler jupyter nbconvert --to notebook --execute --inplace paper/Code/gradient_x_input_analysis.ipynb --ExecutePreprocessor.kernel_name=fluProfiler --ExecutePreprocessor.timeout=180
```

Expected: exit code 0; the output notebook displays a merged-table preview and a 329-row query-site summary.

- [ ] **Step 5: Inspect the result and commit the focused refactor**

Run `git diff --check -- paper/Code/gradient_x_input_analysis.ipynb`, then `git diff --stat -- paper/Code/gradient_x_input_analysis.ipynb`. Stage only the notebook, design, and plan files and commit with message `refactor: focus gradient attribution analysis table`.

Expected: no whitespace errors and one commit containing only the notebook, design, and plan files.

## Self-Review

- Spec coverage: Task 1 retains the stable record construction, five-key metadata merge, and query-token aggregation; it removes every out-of-scope analysis workflow.
- Placeholder scan: no TODO/TBD markers or unspecified validation commands.
- Type consistency: each helper consumes and returns pandas DataFrames; the main cell produces every name used by the example.
