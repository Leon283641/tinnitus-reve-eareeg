#!/bin/bash
#SBATCH --job-name=lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/lora_%j.out
#SBATCH --error=logs/lora_%j.err

set -euo pipefail
: "${MONTAGE:?set MONTAGE env var}"
: "${EPOCH:?set EPOCH env var}"
: "${PREPROCESSED_ROOT:=data/preprocessed}"
: "${RESULTS_ROOT:=results}"
: "${MODEL_CACHE:=model_cache}"
mkdir -p logs "${RESULTS_ROOT}"

python scripts/run_lora.py \
    --input  "${PREPROCESSED_ROOT}/full_equal/${MONTAGE}/EO/epochs_${EPOCH}.npz" \
    --output "${RESULTS_ROOT}/full_equal_${MONTAGE}_EO_${EPOCH}.lora.json" \
    --model-cache "${MODEL_CACHE}"
