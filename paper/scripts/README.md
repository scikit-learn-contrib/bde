# Benchmark Reproducibility

The scripts in this directory reproduce benchmark results used in the paper.

## Benchmark environment

The benchmark script (`benchmark.py`) additionally imports:

- `xgboost`
- `catboost`
- `tabpfn`

These are **benchmark-only** dependencies. They are intentionally kept out of
the core `sklearn-contrib-bde` package and provided through the optional pixi
`benchmark` environment instead.

## Setup

Run from the repository root unless noted otherwise.

1. Install the benchmark environment:

```bash
pixi install -e benchmark
```

This installs the locked benchmark dependencies, including `pip`, `xgboost`,
`catboost`, and `tabpfn`; no manual `pip install` or `ensurepip` step is needed.

2. Run a benchmark through the pixi task. For example:

```bash
pixi run -e benchmark benchmark --dataset airfoil --models linear rf --n-runs 5
```

The task invokes `paper/scripts/run_bench.sh`. The wrapper resolves its paths
relative to its own location, so this equivalent command also works from the
repository root:

```bash
pixi run -e benchmark bash paper/scripts/run_bench.sh \
  --dataset airfoil --models linear rf --n-runs 5
```
