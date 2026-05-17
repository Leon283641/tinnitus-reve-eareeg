#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

set -euo pipefail
: "${MONTAGE:?set MONTAGE env var}"
: "${EPOCH:?set EPOCH env var}"
: "${PREPROCESSED_ROOT:=data/preprocessed}"
: "${EMBEDDINGS_ROOT:=data/embeddings}"
: "${MODEL_CACHE:=model_cache}"
mkdir -p logs

python scripts/extract_embeddings.py \
    --input  "${PREPROCESSED_ROOT}/full_equal/${MONTAGE}/EO/epochs_${EPOCH}.npz" \
    --output "${EMBEDDINGS_ROOT}/full_equal/${MONTAGE}/EO/embeddings_${EPOCH}.npz" \
    --model-cache "${MODEL_CACHE}"
