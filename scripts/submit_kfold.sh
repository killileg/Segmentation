#!/usr/bin/env bash
# Submit every fold as a separate LSF job (one GPU each).
# Adapt the #BSUB header to your cluster before use.
#   ./scripts/submit_kfold.sh 3d 5 "all"
set -euo pipefail

DIM=${1:-2d}
K_FOLDS=${2:-3}
METHODS=${3:-all}
WALLTIME=${WALLTIME:-24:00}
CONFIG=${CONFIG:-configs/my_config.yaml}
QUEUE=${QUEUE:-gpuv100}

mkdir -p logs

for (( FOLD=0; FOLD<K_FOLDS; FOLD++ )); do
  JOB="msfusion_${DIM}_fold${FOLD}"
  bsub <<LSF
#BSUB -J ${JOB}
#BSUB -q ${QUEUE}
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W ${WALLTIME}
#BSUB -o logs/${JOB}.out
#BSUB -e logs/${JOB}.err

cd \$LS_SUBCWD
python run.py --dim ${DIM} --fold ${FOLD} --k-folds ${K_FOLDS} --methods ${METHODS} --config ${CONFIG}
LSF
  echo "submitted ${JOB}"
done
