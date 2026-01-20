#!/usr/bin/env bash
set -euo pipefail
set -a
source bench.env
set +a
python scripts/benchmark.py "$@"
