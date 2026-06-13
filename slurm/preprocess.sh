#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

set -euo pipefail
: "${MONTAGE:?set MONTAGE env var}"
: "${EPOCH:?set EPOCH env var}"
: "${DATA_ROOT:=data/raw/brainclinics}"
: "${OUTPUT_ROOT:=data/preprocessed}"
mkdir -p logs

python scripts/preprocess.py \
    --montage "${MONTAGE}" --epoch "${EPOCH}" \
    --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}"
