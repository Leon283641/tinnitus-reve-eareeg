#!/bin/bash
#SBATCH --job-name=logreg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/logreg_%j.out
#SBATCH --error=logs/logreg_%j.err

set -euo pipefail
: "${MONTAGE:?set MONTAGE env var}"
: "${EPOCH:?set EPOCH env var}"
: "${EMBEDDINGS_ROOT:=data/embeddings}"
: "${RESULTS_ROOT:=results}"
mkdir -p logs "${RESULTS_ROOT}"

python scripts/run_logreg.py \
    --embeddings "${EMBEDDINGS_ROOT}/full_equal/${MONTAGE}/EO/embeddings_${EPOCH}.npz" \
    --output     "${RESULTS_ROOT}/full_equal_${MONTAGE}_EO_${EPOCH}.logreg.json"
