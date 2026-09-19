# Gradient Seasonal Top-15 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NH/SH virus-date seasons, per-season query-site Top-15 results, and their union to the gradient-attribution notebook.

**Architecture:** Keep `attribution_with_dates` unchanged as the source table. Helpers derive one season label from each valid `virusDate`, summarize query sites for each group, and consolidate the resulting Top-15 rankings.

**Tech Stack:** Python, pandas, NumPy, Jupyter Notebook, fluProfiler Conda environment.

## Global Constraints

- Use `virusDate` only.
- `YYYYNH` is 1 February–31 August; `YYYSH` is 1 September–31 January of the following year.
- Exclude invalid/missing dates only from seasonal ranking, not from the main table.
- Keep exactly 15 positions per populated season.

---

### Task 1: Add seasonal ranking helpers and outputs

**Files:**
- Modify: `paper/Code/gradient_x_input_analysis.ipynb`

**Interfaces:**
- Consumes: `attribution_with_dates` with `virusDate` and `query_token_*` columns.
- Produces: `season_sample_counts`, `season_top15`, and `season_top15_union`.

- [ ] **Step 1: Write the failing structural check**

Run a notebook JSON check requiring `add_flu_season`, `build_season_top15`, and `season_top15_union`; expect it to fail before implementation.

- [ ] **Step 2: Implement the helpers and seasonal output cell**

Add `add_flu_season(table)`, assign NH/SH labels from parsed `virusDate`, then calculate each season's Top-15 from `summarize_query_sites`. Build the union with selection count, sorted season list, best rank, and per-season rank mapping.

- [ ] **Step 3: Run all code cells in fluProfiler**

Execute each code cell sequentially and assert every populated season has 15 rows and the union position set equals the seasonal Top-15 position set.

- [ ] **Step 4: Inspect changed files**

Run `git diff --check --no-index /dev/null paper/Code/gradient_x_input_analysis.ipynb` and inspect the three seasonal result tables.

## Self-Review

This plan covers date assignment, missing-date handling, per-season ranks, and the cross-season union. Helper names and output names match the design document.
