"""Preprocess BrainClinics raw CSV recordings into epoched .npz files.

Pipeline: crop 3 s ends -> 0.5 Hz high-pass -> 50 Hz notch -> montage
re-reference -> resample to 200 Hz -> fixed-length epoching -> peak-to-peak
artifact rejection -> per-channel z-score (clip at +/- 15 SD).

Subject set: full_equal (31 tinnitus + 31 controls). Condition: eyes-open (EO).
Montages: full26ch | 4ch_contralateral | 4ch_xbipolar | 2ch_bipolar.
Epochs:   1sec | 2sec | 4sec | 4sec_overlap3sec | 8sec.
"""
import argparse
import json
import os
import warnings
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

warnings.filterwarnings("ignore", category=RuntimeWarning)

ORIGINAL_SFREQ = 500.0
TARGET_SFREQ = 200.0
CROP_SECONDS = 3.0
HIGHPASS_FREQ = 0.5
NOTCH_FREQ = 50.0
CLIP_SD = 15.0
REJECT_PTP_PERCENTILE = 99
REJECT_PTP_MIN = 150.0
REJECT_PTP_MAX = 500.0

EEG_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8",
    "CP3", "CPz", "CP4",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]

#Brainclinics subject IDs are converted to standard names used throughout. 

TINNITUS_SUBJECTS = {
    "T01": ("sub-88011025", 1), "T02": ("sub-88011649", 1), "T03": ("sub-88011741", 1),
    "T04": ("sub-88012685", 1), "T05": ("sub-88012865", 1), "T06": ("sub-88014125", 1),
    "T07": ("sub-88016417", 1), "T08": ("sub-88019613", 1), "T09": ("sub-88021637", 2),
    "T10": ("sub-88021825", 1), "T11": ("sub-88022317", 1), "T12": ("sub-88022589", 1),
    "T13": ("sub-88022897", 1), "T14": ("sub-88023081", 1), "T15": ("sub-88023257", 1),
    "T16": ("sub-88023305", 1), "T17": ("sub-88023621", 1), "T18": ("sub-88023801", 1),
    "T19": ("sub-88023845", 1), "T20": ("sub-88024025", 1), "T21": ("sub-88024157", 1),
    "T22": ("sub-88024249", 1), "T23": ("sub-88024293", 1), "T24": ("sub-88024341", 1),
    "T25": ("sub-88024429", 1), "T26": ("sub-88024473", 1), "T27": ("sub-88024517", 1),
    "T28": ("sub-88024609", 1), "T29": ("sub-88024653", 1), "T30": ("sub-88025369", 1),
    "T31": ("sub-88065877", 1),
}

CONTROL_SUBJECTS = {
    "C01": ("sub-87974617", 1), "C02": ("sub-87974621", 1), "C03": ("sub-87974665", 1),
    "C04": ("sub-87974709", 1), "C05": ("sub-87974841", 1), "C06": ("sub-87974973", 1),
    "C07": ("sub-87976193", 1), "C08": ("sub-87976369", 1), "C09": ("sub-87976413", 1),
    "C10": ("sub-87976457", 1), "C11": ("sub-87976461", 1), "C12": ("sub-87976505", 1),
    "C13": ("sub-87976641", 1), "C14": ("sub-87976773", 1), "C15": ("sub-87976817", 1),
    "C16": ("sub-87976953", 1), "C17": ("sub-87977045", 1), "C18": ("sub-87980197", 1),
    "C19": ("sub-87980241", 1), "C20": ("sub-87980329", 1), "C21": ("sub-87980373", 1),
    "C22": ("sub-87980417", 1), "C23": ("sub-87980689", 1), "C24": ("sub-87980869", 1),
    "C25": ("sub-87980913", 1), "C26": ("sub-87982225", 1), "C27": ("sub-87982849", 1),
    "C28": ("sub-88008997", 1), "C29": ("sub-88041893", 1), "C30": ("sub-88041941", 1),
    "C31": ("sub-88048729", 1), "C32": ("sub-88049585", 1), "C33": ("sub-88051073", 1),
    "C34": ("sub-88053453", 1), "C35": ("sub-88053545", 1), "C36": ("sub-88055121", 1),
    "C37": ("sub-88055301", 1), "C38": ("sub-88057461", 1), "C39": ("sub-88057869", 1),
    "C40": ("sub-88058001", 1), "C41": ("sub-88058633", 1), "C42": ("sub-88059397", 1),
    "C43": ("sub-88067357", 1), "C44": ("sub-88067853", 1), "C45": ("sub-88068841", 1),
    "C46": ("sub-88073029", 1), "C47": ("sub-88075053", 1),
}

# Controls excluded after visual inspection of recording quality (n=16) leaving 31C.
COLAB_REMOVED_CONTROLS = {
    "C02", "C04", "C07", "C11", "C16", "C25", "C27", "C29",
    "C33", "C34", "C35", "C36", "C41", "C42", "C44", "C45",
}

MONTAGES = ["full26ch", "4ch_contralateral", "4ch_xbipolar", "2ch_bipolar"]

EPOCH_CONFIGS = {
    "1sec":             {"duration": 1.0, "overlap": 0.0},
    "2sec":             {"duration": 2.0, "overlap": 0.0},
    "4sec":             {"duration": 4.0, "overlap": 0.0},
    "4sec_overlap3sec": {"duration": 4.0, "overlap": 3.0},
    "8sec":             {"duration": 8.0, "overlap": 0.0},
}


def load_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    return df.values.T.astype(np.float64), list(df.columns)


def find_csv(data_root, sub_id, session, group):
    group_dir = "tinnitus" if group == "tinnitus" else "control"
    base = Path(data_root) / group_dir
    candidates = [
        base / sub_id / f"ses-{session}" / "eeg" / f"{sub_id}_ses-{session}_task-restEO_eeg.csv",
        base / f"{sub_id}_ses-{session}_task-restEO_eeg.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in base.glob(f"*{sub_id}*restEO*"):
        return p
    return None


def bandpass(data, low, high, fs, order=4):
    if low is not None and high is not None:
        b, a = butter(order, [low, high], btype="band", fs=fs)
    elif low is not None:
        b, a = butter(order, low, btype="high", fs=fs)
    elif high is not None:
        b, a = butter(order, high, btype="low", fs=fs)
    else:
        return data
    return filtfilt(b, a, data, axis=-1)


def notch(data, freq, fs, q=30):
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, data, axis=-1)


def resample(data, orig_fs, target_fs):
    if orig_fs == target_fs:
        return data
    up, down = int(target_fs), int(orig_fs)
    g = gcd(up, down)
    return resample_poly(data, up // g, down // g, axis=-1)


def epoch(data, sfreq, epoch_s, overlap_s):
    n_ch, n_samp = data.shape
    win = int(epoch_s * sfreq)
    step = int((epoch_s - overlap_s) * sfreq)
    if step <= 0:
        raise ValueError(f"overlap >= epoch_s ({overlap_s} >= {epoch_s})")
    epochs = []
    s = 0
    while s + win <= n_samp:
        epochs.append(data[:, s:s + win])
        s += step
    return np.array(epochs) if epochs else np.zeros((0, n_ch, win))


def reject_ptp(epochs):
    if len(epochs) == 0:
        return epochs, 0
    ptp = np.ptp(epochs, axis=-1).max(axis=-1)
    threshold = np.clip(np.percentile(ptp, REJECT_PTP_PERCENTILE),
                        REJECT_PTP_MIN, REJECT_PTP_MAX)
    mask = ptp <= threshold
    return epochs[mask], int((~mask).sum())


def zscore(epochs):
    if len(epochs) == 0:
        return epochs
    n_ep, n_ch, n_samp = epochs.shape
    flat = epochs.transpose(1, 0, 2).reshape(n_ch, -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    out = np.clip((flat - mean) / std, -CLIP_SD, CLIP_SD)
    return out.reshape(n_ch, n_ep, n_samp).transpose(1, 0, 2).astype(np.float32)


def apply_montage(data, ch_names, montage):
    ch_idx = {n: i for i, n in enumerate(ch_names)}
    if montage == "full26ch":
        idx = [ch_idx[c] for c in EEG_CHANNELS if c in ch_idx]
        return data[idx], [c for c in EEG_CHANNELS if c in ch_idx]
    t7, t8 = data[ch_idx["T7"]], data[ch_idx["T8"]]
    p7, p8 = data[ch_idx["P7"]], data[ch_idx["P8"]]
    if montage == "4ch_contralateral":
        right = (t8 + p8) / 2
        left = (t7 + p7) / 2
        out = np.array([t7 - right, t8 - left, p7 - right, p8 - left])
        return out, ["T7", "T8", "P7", "P8"]
    if montage == "2ch_bipolar":
        return np.array([t7 - t8, p7 - p8]), ["T7-T8", "P7-P8"]
    if montage == "4ch_xbipolar":
        return np.array([t7 - t8, t7 - p8, p7 - t8, p7 - p8]), \
               ["T7-T8", "T7-P8", "P7-T8", "P7-P8"]
    raise ValueError(f"Unknown montage: {montage}")


def process_subject(data_root, sub_id, session, group, montage):
    p = find_csv(data_root, sub_id, session, group)
    if p is None:
        return None, None
    data, ch_names = load_csv(p)
    crop = int(CROP_SECONDS * ORIGINAL_SFREQ)
    if data.shape[1] <= 2 * crop:
        return None, None
    data = data[:, crop:-crop]
    idx = {n: i for i, n in enumerate(ch_names)}
    present = [c for c in EEG_CHANNELS if c in idx]
    data = data[[idx[c] for c in present]]
    data = bandpass(data, HIGHPASS_FREQ, None, ORIGINAL_SFREQ)
    data = notch(data, NOTCH_FREQ, ORIGINAL_SFREQ)
    data, ch_names = apply_montage(data, present, montage)
    data = resample(data, ORIGINAL_SFREQ, TARGET_SFREQ)
    return data, ch_names


def full_equal_subjects():
    tin = [(k, v[0], v[1]) for k, v in TINNITUS_SUBJECTS.items()]
    ctrl = [(k, v[0], v[1]) for k, v in CONTROL_SUBJECTS.items() if k not in COLAB_REMOVED_CONTROLS]
    assert len(tin) == 31 and len(ctrl) == 31
    return tin, ctrl


def process_cell(data_root, output_root, montage, epoch_name, epoch_cfg):
    out_dir = Path(output_root) / "full_equal" / montage / "EO"
    out_file = out_dir / f"epochs_{epoch_name}.npz"
    if out_file.exists():
        print(f"  SKIP exists: {out_file}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    tin, ctrl = full_equal_subjects()
    all_epochs, all_labels, all_subjects = [], [], []
    n_rejected = 0
    ch_names_final = None

    for short_id, sub_id, session in tin + ctrl:
        group = "tinnitus" if short_id.startswith("T") else "control"
        label = 1 if group == "tinnitus" else 0
        data, ch_names = process_subject(data_root, sub_id, session, group, montage)
        if data is None:
            print(f"    MISSING: {short_id}")
            continue
        ch_names_final = ch_names
        ep = epoch(data, TARGET_SFREQ, epoch_cfg["duration"], epoch_cfg["overlap"])
        if len(ep) == 0:
            continue
        ep, n_rej = reject_ptp(ep)
        n_rejected += n_rej
        if len(ep) == 0:
            print(f"    ALL REJECTED: {short_id}")
            continue
        ep = zscore(ep)
        all_epochs.append(ep)
        all_labels.extend([label] * len(ep))
        all_subjects.extend([sub_id] * len(ep))

    if not all_epochs:
        print(f"  ERROR: no data")
        return

    X = np.concatenate(all_epochs, axis=0)
    y = np.array(all_labels, dtype=np.int64)
    subjects = np.array(all_subjects)
    metadata = {
        "variant": "full_equal", "montage": montage, "condition": "EO",
        "epoch_config": epoch_name, "epoch_duration_s": epoch_cfg["duration"],
        "epoch_overlap_s": epoch_cfg["overlap"], "sfreq": TARGET_SFREQ,
        "ch_names": ch_names_final, "shape": list(X.shape),
        "n_rejected_epochs": n_rejected,
    }
    tmp = out_file.with_suffix(".tmp" + out_file.suffix)
    np.savez_compressed(tmp, X=X, y=y, subjects=subjects, metadata=json.dumps(metadata))
    os.replace(tmp, out_file)
    n_tin = int((y == 1).sum())
    n_ctrl = int((y == 0).sum())
    print(f"  SAVED: {out_file}  X={X.shape}  tinnitus_epochs={n_tin}  control_epochs={n_ctrl}  rejected={n_rejected}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", choices=MONTAGES, required=True)
    ap.add_argument("--epoch", choices=list(EPOCH_CONFIGS), required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()
    print(f"full_equal / {args.montage} / EO / {args.epoch}")
    process_cell(args.data_root, args.output_root, args.montage,
                 args.epoch, EPOCH_CONFIGS[args.epoch])


if __name__ == "__main__":
    main()
