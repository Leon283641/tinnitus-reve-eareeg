# tinnitus-reve-eareeg

Source code for the MSc thesis *Approaching full-cap EEG performance with ear-EEG and a foundation-model representation: a study on tinnitus classification*. Trinity College Dublin, 2026.

## Overview

The codebase reproduces a 40-cell experiment grid in which the REVE EEG foundation model (Brain-bzh) is applied to resting-state recordings from the BrainClinics TDBRAIN dataset to classify tinnitus subjects against age-matched controls. The grid spans four montages (full 26-channel cap, 4-channel contralateral re-reference, 4-channel cross-bipolar, 2-channel bipolar) and five epoch configurations (1 s, 2 s, 4 s, 4 s + 3 s overlap, 8 s) under two methods: a logistic-regression linear probe on the mean-pooled REVE embedding, and parameter-efficient fine-tuning of the REVE encoder via LoRA adapters. All experiments use leave-one-subject-out cross-validation with subject-level majority vote.

## Repository structure

```
.
├── scripts/
│   ├── preprocess.py            # raw CSV -> filtered, montaged, epoched .npz
│   ├── extract_embeddings.py    # epochs.npz -> REVE pooled+flat embeddings.npz
│   ├── run_logreg.py            # linear probe on pooled embeddings (LOSO)
│   └── run_lora.py              # LoRA fine-tune of REVE (LOSO)
├── slurm/                       # SLURM submission templates
├── results/RESULTS.csv          # final results for the 40 experiments (acc, F1, AUROC, sens, spec)
├── environment.yml              # conda environment
└── REPRODUCING.md               # end-to-end reproduction instructions
```

## Data and model

The BrainClinics TDBRAIN dataset is restricted and must be requested from the Brainclinics Foundation: <https://brainclinics.com/tdbrain>. Once obtained, place the raw CSV files at `data/raw/brainclinics/` following the original directory layout (`tinnitus/sub-*/ses-*/eeg/*.csv` and `control/sub-*/ses-*/eeg/*.csv`).

The REVE foundation model (`brain-bzh/reve-base`) and its companion position bank (`brain-bzh/reve-positions`) are loaded from the Hugging Face Hub on first use. 

## Reproducing the experiments

See `REPRODUCING.md` for the full pipeline. A summary:

1. `preprocess.py` produces one `.npz` per (montage, epoch) combination.
2. `extract_embeddings.py` produces a corresponding embeddings `.npz`.
3. `run_logreg.py` runs the linear probe; `run_lora.py` runs LoRA fine-tuning.
4. All four scripts accept a single cell at a time; the 40 `(montage, epoch, method)` triples used in the thesis are recorded in `results/RESULTS.csv`.

The full grid can be reproduced on a single H100 GPU in roughly five days of wall-clock time. Preprocessing and the linear-probe step run on CPU; the LoRA step requires a GPU.

## References

The REVE foundation model and the BrainClinics TDBRAIN dataset should be cited per their original publications; see `REPRODUCING.md`.
