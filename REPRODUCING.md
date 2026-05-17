# Reproducing the thesis results

## Environment

```
conda env create -f environment.yml
conda activate tinnitus-reve
```

## Data

Request the BrainClinics TDBRAIN dataset and place the raw CSVs at:

```
data/raw/brainclinics/
├── tinnitus/sub-*/ses-*/eeg/sub-*_ses-*_task-restEO_eeg.csv
└── control/sub-*/ses-*/eeg/sub-*_ses-*_task-restEO_eeg.csv
```

The subject lists used in this work are encoded in `scripts/preprocess.py` (31 tinnitus + 31 controls, the `full_equal` set). Sixteen controls were excluded after visual inspection of recording quality and are listed in `COLAB_REMOVED_CONTROLS` in that script.

## Model cache (optional, for offline cluster nodes)

```
huggingface-cli download brain-bzh/reve-base --local-dir model_cache/reve-base
huggingface-cli download brain-bzh/reve-positions --local-dir model_cache/reve-positions
```

## The pipeline (one cell at a time)

A *cell* is defined by `(montage, epoch)`. The 40 cells in the thesis are listed in `cells/grid_40.csv`. For a single cell, e.g. `4ch_contralateral × 4sec`:

```
# 1. Preprocess raw recordings -> epoched .npz
python scripts/preprocess.py --montage 4ch_contralateral --epoch 4sec \
    --data-root data/raw/brainclinics --output-root data/preprocessed

# 2. Extract REVE embeddings from the epoched .npz
python scripts/extract_embeddings.py \
    --input  data/preprocessed/full_equal/4ch_contralateral/EO/epochs_4sec.npz \
    --output data/embeddings/full_equal/4ch_contralateral/EO/embeddings_4sec.npz \
    --model-cache model_cache

# 3a. Linear probe on pooled embeddings (CPU)
python scripts/run_logreg.py \
    --embeddings data/embeddings/full_equal/4ch_contralateral/EO/embeddings_4sec.npz \
    --output results/4ch_contralateral_4sec.logreg.json

# 3b. LoRA fine-tune (GPU)
python scripts/run_lora.py \
    --input  data/preprocessed/full_equal/4ch_contralateral/EO/epochs_4sec.npz \
    --output results/4ch_contralateral_4sec.lora.json \
    --model-cache model_cache
```

## Batch submission on a SLURM cluster

The `slurm/` directory contains submission templates parameterised by environment variables. Example:

```
MONTAGE=4ch_contralateral EPOCH=4sec sbatch slurm/preprocess.sh
MONTAGE=4ch_contralateral EPOCH=4sec sbatch slurm/extract_embeddings.sh
MONTAGE=4ch_contralateral EPOCH=4sec sbatch slurm/run_lora.sh
```

To reproduce the full 40-cell grid, loop over rows of `cells/grid_40.csv` and submit each as a separate job. Logistic-regression cells are CPU-bound and run in seconds; LoRA cells take roughly 1–10 hours per cell on an H100 GPU depending on epoch count.

## Output schema

Each `run_*.py` script writes a JSON containing:

- `method`, `cv_mode`, `config`
- `subject_results` — per-subject `{true_label, pred_label, mean_prob, n_epochs, epoch_accuracy}`
- `metrics` — aggregate `{accuracy, balanced_accuracy, f1_weighted, auroc, cohens_kappa, sensitivity, specificity, n_total, n_tinnitus, n_control}`

The aggregated results from the thesis are in `results/RESULTS.csv` (40 rows).

## References

- *REVE: a foundation model for EEG.* `brain-bzh/reve-base` on Hugging Face.
- *BrainClinics TDBRAIN.* van Dijk et al. (2022), *Scientific Data*.
- *LoRA: Low-Rank Adaptation of Large Language Models.* Hu et al. (2022), ICLR.
