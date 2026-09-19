# fluProfiler Project Instructions

## Mandatory Conda environment

All code in this project must run in the Conda environment named
`fluProfiler`. This includes Python commands, scripts, tests, data-processing
jobs, training and evaluation jobs, and Jupyter notebooks.

- For non-interactive commands, prefer `conda run -n fluProfiler <command>`.
- For an interactive shell, run `conda activate fluProfiler` before executing
  project code.
- Do not use the base Conda environment, the system Python, or another virtual
  environment for this project.
- Do not install project dependencies outside the `fluProfiler` environment.
- Configure notebook kernels to use the Python interpreter from the
  `fluProfiler` environment.
- Verification and test commands must use the same environment as production
  runs, for example `conda run -n fluProfiler pytest`.
