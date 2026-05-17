"""Extract REVE embeddings from a preprocessed epochs .npz file.

Saves both per-channel concatenated (flat) and channel-mean-pooled embeddings.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

REVE_MODEL = "brain-bzh/reve-base"
REVE_POSITIONS = "brain-bzh/reve-positions"
BATCH_SIZE = 16

POSITION_MAP = {
    "T7": "T7", "T8": "T8", "P7": "P7", "P8": "P8",
    "T7-T8": "T7", "T7-P8": "T7", "P7-T8": "P7", "P7-P8": "P7",
    "Fp1": "Fp1", "Fp2": "Fp2", "F7": "F7", "F3": "F3", "Fz": "Fz",
    "F4": "F4", "F8": "F8", "FC3": "FC3", "FCz": "FCz", "FC4": "FC4",
    "C3": "C3", "Cz": "Cz", "C4": "C4",
    "CP3": "CP3", "CPz": "CPz", "CP4": "CP4",
    "P3": "P3", "Pz": "Pz", "P4": "P4",
    "O1": "O1", "Oz": "Oz", "O2": "O2",
}


def load_positions(ch_names, device, cache_dir):
    from transformers import AutoModel
    bank = AutoModel.from_pretrained(REVE_POSITIONS, trust_remote_code=True,
                                     cache_dir=cache_dir)
    available = bank.get_all_positions()
    reve_names = []
    for ch in ch_names:
        name = POSITION_MAP.get(ch, ch)
        if name not in available:
            raise ValueError(f"channel {ch!r} (mapped to {name!r}) not in position bank")
        reve_names.append(name)
    return bank(reve_names).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-cache", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.model_cache:
        os.environ["HF_HUB_CACHE"] = args.model_cache
        os.environ["TRANSFORMERS_CACHE"] = args.model_cache

    from transformers import AutoModel
    model = AutoModel.from_pretrained(REVE_MODEL, trust_remote_code=True,
                                      cache_dir=args.model_cache)
    model.eval().to(device)

    data = np.load(args.input, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]
    metadata = json.loads(str(data["metadata"]))
    ch_names = metadata["ch_names"]
    print(f"input: {X.shape}  channels: {ch_names}")

    positions = load_positions(ch_names, device, args.model_cache)
    use_amp = device.type == "cuda"

    flat_chunks, pooled_chunks = [], []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(X), BATCH_SIZE):
            bx = torch.from_numpy(X[i:i + BATCH_SIZE]).to(device)
            bp = positions.unsqueeze(0).repeat(bx.shape[0], 1, 1)
            if use_amp:
                with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                    out = model(bx, bp)
            else:
                out = model(bx, bp)
            flat_chunks.append(out.reshape(bx.shape[0], -1).float().cpu().numpy())
            pooled_chunks.append(out.mean(dim=(1, 2)).float().cpu().numpy())

    embeddings_flat = np.concatenate(flat_chunks, axis=0)
    embeddings_pooled = np.concatenate(pooled_chunks, axis=0)
    elapsed = time.time() - t0
    print(f"flat: {embeddings_flat.shape}  pooled: {embeddings_pooled.shape}  elapsed: {elapsed:.1f}s")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp" + out.suffix)
    np.savez_compressed(
        tmp,
        embeddings_flat=embeddings_flat,
        embeddings_pooled=embeddings_pooled,
        y=y, subjects=subjects,
        metadata=json.dumps({**metadata, "reve_model": REVE_MODEL,
                             "flat_dim": embeddings_flat.shape[1],
                             "pooled_dim": embeddings_pooled.shape[1]}),
    )
    os.replace(tmp, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
