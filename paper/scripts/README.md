# Benchmark Reproducibility

The scripts in this directory reproduce benchmark results used in the paper.

## Additional dependencies

The benchmark script (`benchmark.py`) additionally imports:

- `xgboost`
- `catboost`
- `tabpfn`

These are **benchmark-only** dependencies. They are intentionally not required by
the core `sklearn-contrib-bde` package and are therefore not listed under the
main project dependencies.

## Setup

Run from the repository root unless noted otherwise.

1. Create/install the base project environment as specified in the main project 
documentation.

2. Install benchmark-only Python packages into the pixi environment:

```bash
pixi run python -m pip install xgboost catboost tabpfn
```

3. Run the benchmark script `paper/scripts/run_bench.sh` to reproduce the benchmark.
