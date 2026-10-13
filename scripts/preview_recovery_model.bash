#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/training:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=glfw
[[ -f training/navrl/glfw_capture.so ]] || bash scripts/build_viewer.bash
exec .venv/bin/python -B training/preview_sensor_navigation.py \
  --bundle models/recovery_20261012/portable \
  --baseline models/recovery_20261012/baseline_portable \
  --manifest training/config/navigation_stage3_regression.json \
  --case independent_dynamic_000 \
  --output training/runs/manual_recovery_release \
  --interactive-goals --pick-place-obstacles "$@"
