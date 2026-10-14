#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/training:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=glfw
exec "${AGV_PYTHON:-$ROOT/.venv/bin/python}" training/preview_best_model_release.py \
  --bundle models/quality_recovery_20261014/portable \
  --baseline models/quality_recovery_20261014/portable \
  --manifest models/quality_recovery_20261014/reports/cases.json \
  --case independent_dynamic_000 \
  --output training/runs/manual_best_20261014 "$@"
