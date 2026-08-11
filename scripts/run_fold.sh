#!/usr/bin/env bash
# Run one fold locally.
#   ./scripts/run_fold.sh 2d 0 "late_fusion meta_learner"
#
# Set CONFIG to your own copy of configs/example.yaml (with your data paths filled in),
# or export XRAY/NEUTRON/SEG instead.
set -euo pipefail

DIM=${1:-2d}
FOLD=${2:-0}
METHODS=${3:-all}
K_FOLDS=${K_FOLDS:-3}
CONFIG=${CONFIG:-configs/my_config.yaml}

ARGS=(--dim "${DIM}" --fold "${FOLD}" --k-folds "${K_FOLDS}" --methods ${METHODS})

if [[ -f "${CONFIG}" ]]; then
  ARGS+=(--config "${CONFIG}")
else
  ARGS+=(--xray "${XRAY:?set XRAY or provide a config file}" \
         --neutron "${NEUTRON:?set NEUTRON or provide a config file}" \
         --seg "${SEG:?set SEG or provide a config file}")
fi

python run.py "${ARGS[@]}"
