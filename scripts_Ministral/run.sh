#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate qwen38
export QWEN38_RAM_HARD_FLOOR_GIB="${QWEN38_RAM_HARD_FLOOR_GIB:-2}"
export QWEN38_RAM_RESERVE_GIB="${QWEN38_RAM_RESERVE_GIB:-3}"
export QWEN38_GPU_RESERVE_GIB="${QWEN38_GPU_RESERVE_GIB:-3}"
export QWEN38_STREAM_OUTPUT="${QWEN38_STREAM_OUTPUT:-1}"
cd "$ROOT"
exec python "$HERE/08_run_evaluation.py" "$@"
