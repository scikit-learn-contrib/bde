#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

set -a
source "${script_dir}/bench.env"
set +a

python "${script_dir}/benchmark.py" "$@"
