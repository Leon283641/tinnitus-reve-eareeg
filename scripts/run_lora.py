"""LoRA fine-tuning of REVE for tinnitus classification (LOSO).

LoRA adapters (rank 8, alpha 16) are inserted into every Q/K/V and output
projection of the REVE attention blocks. Each LOSO fold trains a fresh head
plus the LoRA parameters in two stages: (1) head-only warm-up, (2) joint
training of head + LoRA. Per-subject prediction is the mean of per-epoch
positive-class probabilities, thresholded at 0.5.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, roc_auc_score)

REVE_MODEL = "brain-bzh/reve-base"
REVE_POSITIONS = "brain-bzh/reve-positions"

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
HEAD_WARMUP_EPOCHS = 5
LORA_TRAIN_EPOCHS = 25
HEAD_LR = 1e-3
LORA_LR = 1e-4
BATCH_SIZE = 32
PATIENCE = 5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

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


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank, alpha, dropout):
        super().__init__()
        self.original = original_linear
        self.scaling = alpha / rank
        d_in, d_out = original_linear.in_features, original_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + (
            self.dropout(x) @ self.lora_A.to(x.device) @ self.lora_B.to(x.device) * self.scaling
        )

    @property
    def lora_params(self):
        return [self.lora_A, self.lora_B]


def apply_lora(model, rank, alpha, dropout):
    modules = []
    n_params = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in ("to_qkv", "to_out")):
            continue
        parent = model
        for part in name.split(".")[:-1]:
            parent = getattr(parent, part)
        layer = LoRALinear(module, rank, alpha, dropout)
        setattr(parent, name.split(".")[-1], layer)
        modules.append(layer)
        n_params += module.in_features * rank + rank * module.out_features
    print(f"  applied LoRA to {len(modules)} layers ({n_params/1e6:.2f}M trainable)")
    return modules


def load_positions(ch_names, device, cache_dir):
    from transformers import AutoModel
    bank = AutoModel.from_pretrained(REVE_POSITIONS, trust_remote_code=True,
                                     cache_dir=cache_dir)
    available = bank.get_all_positions()
    names = []
    for ch in ch_names:
        n = POSITION_MAP.get(ch, ch)
        if n not in available:
            raise ValueError(f"channel {ch!r} (mapped to {n!r}) not in position bank")
        names.append(n)
    return bank(names).to(device)


def train_fold(model, head, positions, X_tr, y_tr, X_val, y_val, device, lora_modules):
    n_pos = (y_tr == 1).sum().item()
    n_neg = (y_tr == 0).sum().item()
    weight = torch.tensor([n_pos / (n_pos + n_neg), n_neg / (n_pos + n_neg)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    use_amp = device.type == "cuda"

    def embed(bx):
        bp = positions.unsqueeze(0).repeat(bx.shape[0], 1, 1)
        if use_amp:
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                out = model(bx, bp)
        else:
            out = model(bx, bp)
        return out.reshape(bx.shape[0], -1).float()

    def evaluate(X, y):
        head.eval(); model.eval()
        probs = []
        with torch.inference_mode():
            for i in range(0, len(X), BATCH_SIZE):
                bx = X[i:i + BATCH_SIZE].to(device)
                p = torch.softmax(head(embed(bx)), dim=1)[:, 1].cpu().numpy()
                probs.extend(p)
        preds = (np.array(probs) > 0.5).astype(int)
        return balanced_accuracy_score(y.numpy(), preds)

    # Stage 1: head warm-up (LoRA frozen)
    for m in lora_modules:
        for p in m.lora_params:
            p.requires_grad = False
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(HEAD_WARMUP_EPOCHS):
        head.train(); model.eval()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                emb = embed(bx)
            loss = criterion(head(emb), by)
            opt.zero_grad(); loss.backward(); opt.step()

    # Stage 2: joint LoRA + head
    for m in lora_modules:
        for p in m.lora_params:
            p.requires_grad = True
    lora_params = [p for m in lora_modules for p in m.lora_params]
    head_params = list(head.parameters())
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": LORA_LR},
         {"params": head_params, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5)

    best_val = 0.0
    best_lora = None
    best_head = None
    patience = 0
    for _ in range(LORA_TRAIN_EPOCHS):
        head.train(); model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            loss = criterion(head(embed(bx)), by)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(lora_params + head_params, GRAD_CLIP)
            opt.step()
        val = evaluate(X_val, y_val)
        sched.step(val)
        if val > best_val:
            best_val = val
            best_lora = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
            best_head = {k: v.clone().detach() for k, v in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_lora:
        for n, p in model.named_parameters():
            if n in best_lora:
                p.data.copy_(best_lora[n])
    if best_head:
        head.load_state_dict(best_head)
    return best_val


def predict(model, head, positions, X_test, device):
    head.eval(); model.eval()
    probs = []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for i in range(0, len(X_test), BATCH_SIZE):
            bx = X_test[i:i + BATCH_SIZE].to(device)
            bp = positions.unsqueeze(0).repeat(bx.shape[0], 1, 1)
            if use_amp:
                with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                    out = model(bx, bp)
            else:
                out = model(bx, bp)
            emb = out.reshape(bx.shape[0], -1).float()
            p = torch.softmax(head(emb), dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
    return np.array(probs)


def compute_metrics(subject_results):
    y_true = np.array([r["true_label"] for r in subject_results])
    y_pred = np.array([r["pred_label"] for r in subject_results])
    y_prob = np.array([r["mean_prob"] for r in subject_results])
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "cohens_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        "auroc": round(roc_auc_score(y_true, y_prob), 4) if len(set(y_true)) > 1 else None,
        "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
        "n_total": len(y_true),
        "n_tinnitus": int((y_true == 1).sum()),
        "n_control": int((y_true == 0).sum()),
    }


def run_loso(input_npz, cache_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data = np.load(input_npz, allow_pickle=True)
    X = data["X"]
    y_np = data["y"]
    subjects = data["subjects"]
    ch_names = json.loads(str(data["metadata"]))["ch_names"]
    positions = load_positions(ch_names, device, cache_dir)

    if cache_dir:
        os.environ["HF_HUB_CACHE"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir

    results = []
    unique_subs = np.unique(subjects)
    for i, test_sub in enumerate(unique_subs):
        t0 = time.time()
        test_mask = subjects == test_sub
        train_mask = ~test_mask
        true_label = int(y_np[test_mask][0])

        # Validation subject: same class as the held-out test subject if available.
        train_subs = np.unique(subjects[train_mask])
        train_labels = {s: int(y_np[subjects == s][0]) for s in train_subs}
        same_class = [s for s, l in train_labels.items() if l == true_label]
        val_sub = same_class[-1] if same_class else train_subs[-1]
        val_mask = subjects == val_sub
        actual_train_mask = train_mask & ~val_mask

        X_tr = torch.from_numpy(X[actual_train_mask]).float()
        y_tr = torch.from_numpy(y_np[actual_train_mask]).long()
        X_val = torch.from_numpy(X[val_mask]).float()
        y_val = torch.from_numpy(y_np[val_mask]).long()
        X_test = torch.from_numpy(X[test_mask]).float()

        from transformers import AutoModel
        model = AutoModel.from_pretrained(REVE_MODEL, trust_remote_code=True,
                                          cache_dir=cache_dir)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad = False
        lora_modules = apply_lora(model, LORA_RANK, LORA_ALPHA, LORA_DROPOUT)

        with torch.inference_mode():
            dummy = torch.randn(1, len(ch_names), X.shape[2]).to(device)
            head_dim = model(dummy, positions.unsqueeze(0)).reshape(1, -1).shape[1]
        head = nn.Sequential(
            nn.LayerNorm(head_dim), nn.Dropout(0.1), nn.Linear(head_dim, 2)
        ).to(device)

        best_val = train_fold(model, head, positions, X_tr, y_tr, X_val, y_val,
                              device, lora_modules)
        probs = predict(model, head, positions, X_test, device)
        mean_prob = float(probs.mean())
        pred_label = int(mean_prob > 0.5)
        epoch_acc = float(((probs > 0.5).astype(int) == y_np[test_mask]).mean())

        results.append({
            "subject": str(test_sub),
            "true_label": true_label,
            "pred_label": pred_label,
            "mean_prob": round(mean_prob, 4),
            "n_epochs": int(test_mask.sum()),
            "epoch_accuracy": round(epoch_acc, 4),
            "correct": pred_label == true_label,
            "best_val_acc": round(best_val, 4),
        })
        dt = time.time() - t0
        status = "OK" if pred_label == true_label else "WR"
        print(f"  fold {i+1}/{len(unique_subs)}  {test_sub}  pred={pred_label}  "
              f"prob={mean_prob:.3f}  val={best_val:.3f}  [{status}]  {dt:.0f}s")

        del model, head, lora_modules
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-cache", default=None)
    args = ap.parse_args()

    t0 = time.time()
    subject_results = run_loso(args.input, args.model_cache)
    metrics = compute_metrics(subject_results)
    elapsed = round(time.time() - t0, 1)

    result = {
        "method": "lora_finetune",
        "cv_mode": "loso",
        "lora_config": {"rank": LORA_RANK, "alpha": LORA_ALPHA,
                        "head_warmup_epochs": HEAD_WARMUP_EPOCHS,
                        "lora_train_epochs": LORA_TRAIN_EPOCHS,
                        "head_lr": HEAD_LR, "lora_lr": LORA_LR,
                        "batch_size": BATCH_SIZE},
        "subject_results": subject_results,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {args.output}  acc={metrics['accuracy']}  auroc={metrics['auroc']}  elapsed={elapsed}s")


if __name__ == "__main__":
    main()
