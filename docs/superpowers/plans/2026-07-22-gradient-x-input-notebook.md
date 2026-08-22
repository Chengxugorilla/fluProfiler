# Gradient × Input Analysis Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and execute `paper/Code/gradient_x_input_analysis.ipynb` so it reproduces the global and seasonal H3N2 Gradient × Input analyses with all tables and figures embedded and no external image output.

**Architecture:** Build one self-contained notebook whose first code cell owns every input path and analysis constant. Later cells validate and load the existing attribution artifacts, compute global and seasonal summaries from the NPZ arrays plus source metadata, and render all figures inline. Execute the notebook with the project Conda kernel, then validate its JSON, outputs, numerical invariants, and absence of image writes.

**Tech Stack:** Python 3, Jupyter nbformat/nbclient, NumPy, Pandas, Matplotlib, Seaborn, SciPy, IPython display.

## Global Constraints

- Create only `paper/Code/gradient_x_input_analysis.ipynb` as the analysis artifact.
- Store executed cell outputs and figures in the notebook.
- Do not call `savefig` or write image files.
- Use aligned HA1 positions 1–329 and exclude BOS/EOS from residue rankings.
- Use `mean(abs(reference + query))` for seasonal residue weights.
- Use the 2014NH–2023SH display range and the project's existing NH/SH date convention.
- Join source metadata with the attribution export deduplication key: `seq_a`, `seq_c`, `serumPassCat`, `virusPassCat`.
- Do not add `#` or `*` suffixes to residue labels.

---

### Task 1: Create the self-contained analysis notebook

**Files:**
- Create: `paper/Code/gradient_x_input_analysis.ipynb`
- Reference: `paper/Code/Fig.3.ipynb`
- Reference: `experiments/serum_gate/run_gradient_x_input.py`

**Interfaces:**
- Consumes: the result directory configured as `RESULT_DIR`, with the four Gradient × Input artifacts; sibling `metrics.csv` and `run_config.json`; source CSV from `analysis_config["data_csv"]`.
- Produces: an nbformat v4 notebook containing deterministic code and markdown cells, with named in-memory objects `config`, `arrays`, `samples`, `site_summary`, `core_summary`, `combined`, `mutation_mask`, `seasonal_weights`, `seasonal_top_sites`, and `seasonal_site_table`.

- [ ] **Step 1: Run a failing artifact check**

Run:

```bash
test -f paper/Code/gradient_x_input_analysis.ipynb
```

Expected: exit status 1 because the notebook does not yet exist.

- [ ] **Step 2: Create notebook metadata, imports, paths, and validations**

Create an nbformat v4 notebook with kernel metadata for the `fluProfiler` environment. The first cells must define:

```python
RESULT_DIR = Path(
    "/home/chenyh/workspace/fluProfiler/results/H1H3_HA1_v1.0/20260717_164256/"
    "SerumGate-Minus-latent8/titer/seed_0/subtype/H3N2/gradient_x_input"
)
MODEL_DIR = RESULT_DIR.parent
EXPECTED_SAMPLE_COUNT = 24_740
EXPECTED_TOKEN_COUNT = 331
CANONICAL_H3_SITES = [145, 155, 156, 158, 159, 189, 193]
SEASONS = [f"{year}{hemisphere}" for year in range(2014, 2024) for hemisphere in ("NH", "SH")]
```

Validate all required paths, `config["method"] == "gradient_x_input"`, sample count 24,740, token count 331, exact NPZ shapes, finite values, and `sample_index == arange(24740)`.

- [ ] **Step 3: Add global analysis cells**

Compute:

```python
reference = arrays["reference"][:, 1:330]
query = arrays["query"][:, 1:330]
combined = reference + query
positions = np.arange(1, 330)
```

Add separate markdown and code cells for prediction/label distributions, final test metrics, global attribution curves, top-30 combined sites, canonical-site ranks, region enrichment, reference/query cancellation, mutation-versus-conserved comparisons, and attribution completeness. Every plot must end with `plt.tight_layout(); plt.show(); plt.close()`.

The mutation cell creates `mutation_mask` from character-wise comparison of `seq_a` and `seq_c` and asserts shape `(24_740, 329)`. The completeness cell reports Pearson correlations of reference, query, and combined attribution sums with `arrays["mean"]`, plus combined sum MAE.

- [ ] **Step 4: Add seasonal metadata and weight computation cells**

Read only the deduplication key plus `virusDate` from the source CSV. Reject conflicting dates per join key. Convert valid dates with:

```python
def assign_season(value):
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return pd.NA
    hemisphere = "NH" if date.month <= 6 else "SH"
    return f"{date.year}{hemisphere}"
```

Join with `samples` using `validate="one_to_one"`, report missing dates, and compute:

```python
seasonal_weights = pd.DataFrame(index=positions, columns=SEASONS, dtype=float)
for season in SEASONS:
    mask = joined["season"].eq(season).fillna(False).to_numpy()
    if mask.any():
        seasonal_weights.loc[:, season] = np.abs(combined[mask]).mean(axis=0)

nonempty_seasons = [season for season in SEASONS if seasonal_weights[season].notna().any()]
if not nonempty_seasons:
    raise ValueError("No observations fall in the configured 2014NH–2023SH seasons")
seasonal_top_sites = {
    season: seasonal_weights[season].nlargest(15).index.astype(int).tolist()
    for season in nonempty_seasons
}
```

- [ ] **Step 5: Add the requested aggregation table and heatmap**

Define the exact Epitope A–E and Unknown sets from `paper/Code/Fig.3.ipynb`. Classify all remaining positions as Unreported. Build the top-site union deterministically, ordered by group and numeric position.

Render a Matplotlib table with columns `Epitope` and `Key antigenic HA1 sites across seasons`, using the established green/cyan/yellow/salmon/blue palette and white for Unreported. Render a grayscale Seaborn heatmap of raw `seasonal_weights` for that ordered union, with epitope-colored y-label bounding boxes. Both figures call `plt.show()`; neither writes a file.

- [ ] **Step 6: Add interpretation and limitations cells**

End with generated Markdown listing the observed leading sites and stating that Gradient × Input is a local first-order approximation, token sums are not exact output decompositions, the analysis includes training and test rows, and the titer split measures within-serum interpolation rather than unseen-serum generalization.

- [ ] **Step 7: Validate structure before execution**

Run a JSON check asserting nbformat 4, at least 20 cells, no `savefig`, and presence of `seasonal_top_sites` and `mutation_mask`.

Expected: exit status 0.

---

### Task 2: Execute and verify the notebook

**Files:**
- Modify: `paper/Code/gradient_x_input_analysis.ipynb` by embedding execution counts and outputs only.

**Interfaces:**
- Consumes: the unexecuted notebook from Task 1 and the `fluProfiler` Conda environment.
- Produces: the same notebook with all execution counts and inline display outputs populated.

- [ ] **Step 1: Record pre-execution image artifacts**

Run:

```bash
find paper/Code -maxdepth 1 -type f \( -name '*.png' -o -name '*.svg' -o -name '*.pdf' \) -printf '%f\n' | sort
```

Store the output for comparison after execution.

- [ ] **Step 2: Execute the notebook in place**

Run:

```bash
/home/chenyh/miniconda3/envs/fluProfiler/bin/jupyter nbconvert \
  --to notebook \
  --execute paper/Code/gradient_x_input_analysis.ipynb \
  --output gradient_x_input_analysis.ipynb \
  --output-dir paper/Code \
  --ExecutePreprocessor.timeout=900 \
  --ExecutePreprocessor.kernel_name=python3
```

Expected: exit status 0 and a message writing the executed notebook.

- [ ] **Step 3: Validate outputs and numerical invariants**

Parse notebook JSON and assert no error outputs, every code cell has an execution count, at least eight display outputs, at least six embedded images, and text outputs include `24740`, `154`, and `190`.

Expected: print code-cell, display-output, and image counts and exit status 0.

- [ ] **Step 4: Confirm no image files were created**

Repeat the image-artifact command from Step 1 and compare output byte-for-byte with the pre-execution list.

Expected: no new PNG, SVG, or PDF files.

- [ ] **Step 5: Run final source-policy check**

Run:

```bash
rg -n "savefig|to_csv|to_excel|imwrite" paper/Code/gradient_x_input_analysis.ipynb
```

Expected: no matches.

- [ ] **Step 6: Inspect diff scope**

Run `git status --short -- paper/Code/gradient_x_input_analysis.ipynb` and `git diff --stat -- paper/Code/gradient_x_input_analysis.ipynb`.

Expected: only the requested notebook is new or modified for implementation.

- [ ] **Step 7: Leave implementation uncommitted**

Do not commit the notebook unless the user explicitly requests a commit.

