#!/usr/bin/env bash
# Combine all folds' metrics once the jobs have finished.
set -euo pipefail

DIM=${1:-2d}
FINAL_DIR=${2:-./outputs}

python tools/aggregate_kfold_results.py \
  --dim "${DIM}" \
  --final-dir "${FINAL_DIR}" \
  --out "${FINAL_DIR}/aggregate_${DIM}.csv"
