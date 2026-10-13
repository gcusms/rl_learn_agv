#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/training:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec nice -n 15 .venv/bin/python -B -m prototypes.train_continued_recovery \
  --checkpoint models/recovery_20261012/parent/policy.zip \
  --output "${1:-training/runs/recovery_reproduction_10000}" \
  --failure-pool models/recovery_20261012/training_inputs/failure_pool.json \
  --training-log models/recovery_20261012/training_inputs/replay_episode.jsonl \
  --replay-step 10458 --steps "${2:-10000}" --seed 92612 \
  --resume-quality --scheduled-recovery --sensor-core time-consistent
