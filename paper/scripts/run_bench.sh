#!/usr/bin/env bash
set -euo pipefail
set -a
source scripts/bench.env
set +a
python scripts/benchmark.py "$@"
