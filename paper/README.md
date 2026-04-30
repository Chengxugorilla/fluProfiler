# Paper Reproduction

This directory contains the notebooks and inputs needed to reproduce the paper figures.

## Quick Start

```bash
conda create -n fluprofiler-paper python=3.10 -y
conda activate fluprofiler-paper
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Run notebooks from `paper/Code/` so the relative paths resolve correctly.

## Notebooks

### `Fig.S1.ipynb`
- Input: project code and local figure data used by the notebook
- Output: `paper/Figure/Fig.S1B1.svg`, `paper/Figure/Fig.S1B2.svg`

### `Fig.2BCD.ipynb`
- Input: `paper/data/Fig2/*.csv`, `paper/data/Fig2/vaccine_strain_sequences.fasta`
- Output: `paper/Figure/Fig2BCD.svg`

### `Fig.2E.ipynb`
- Input: project code and local Fig. 2 data
- Output: `paper/Figure/Fig2E.svg`

### `Fig.3.ipynb`
- Input: project code and local data used by the notebook
- Output: Fig. 3 figure panels under `paper/Figure/`

### `Fig.4 (Vaccine_selection).ipynb`
- Input: project code and vaccine selection data
- Output: Fig. 4 figure panels under `paper/Figure/`

### `Fig.5AB (viral evolution map).ipynb`
- Input: project code and evolution data
- Output: Fig. 5A-B figure panels under `paper/Figure/`

### `Fig.5C.ipynb`
- Input: project code and sampling data
- Output: `paper/Figure/Fig5C_PCA_sampling.svg`, `paper/Figure/Fig5C_sampling_Type.svg`

### `Fig.5CD (antigen_space).ipynb`
- Input: project code and antigen-space data
- Output: Fig. 5C-D figure panels under `paper/Figure/`

### `Extended_Fig5.ipynb`
- Input: `paper/data/Fig2/vaccine_strain_sequences.fasta`
- Output: Extended Fig. 5 panels under `paper/Figure/`

