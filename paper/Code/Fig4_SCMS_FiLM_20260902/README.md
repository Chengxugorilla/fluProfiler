# SCMS-FiLM Figure 4 re-analysis

This independent workflow leaves `Fig.4 (Vaccine_selection).ipynb` untouched.
It evaluates each rolling season's SCMS-FiLM checkpoint on the complete
candidate-serum × target-virus grid, then ranks candidates by mean predicted
antigenic distance (lower is better).

`A/DARWIN/6/2021 CELL` is included once its dedicated HA1 embedding exists at
`assets/embeddings/matrix_DARWIN6_2021_CELL_HA1.pt`.  Its exact HA1 FASTA is
included in `assets/darwin6_2021_cell_ha1.fasta`; generate that single matrix
with LucaVirus before running the notebook.  Other legacy supplementary
records without HA1 assets continue to be reported and skipped.

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n fluProfiler python \
  paper/Code/Fig4_SCMS_FiLM_20260902/run_vaccine_selection.py \
  --device cuda:0 \
  --gpu-cache-gb 20
```

For a smoke test, use one season:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n fluProfiler python \
  paper/Code/Fig4_SCMS_FiLM_20260902/run_vaccine_selection.py \
  --seasons 39 --device cuda:0 --gpu-cache-gb 20
```
