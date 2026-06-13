"""Logistic-regression linear probe on REVE pooled embeddings (LOSO).

Per-subject prediction is the mean of per-epoch positive-class probabilities,
thresholded at 0.5.
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, roc_auc_score)


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


def run_loso(embeddings_path):
    data = np.load(embeddings_path, allow_pickle=True)
    X = data["embeddings_pooled"]
    y = data["y"]
    subjects = data["subjects"]
    results = []
    for test_sub in np.unique(subjects):
        test_mask = subjects == test_sub
        train_mask = ~test_mask
        scaler = StandardScaler().fit(X[train_mask])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        clf.fit(scaler.transform(X[train_mask]), y[train_mask])
        probs = clf.predict_proba(scaler.transform(X[test_mask]))[:, 1]
        mean_prob = float(probs.mean())
        pred_label = int(mean_prob > 0.5)
        true_label = int(y[test_mask][0])
        epoch_acc = float(((probs > 0.5).astype(int) == y[test_mask]).mean())
        results.append({
            "subject": str(test_sub),
            "true_label": true_label,
            "pred_label": pred_label,
            "mean_prob": round(mean_prob, 4),
            "n_epochs": int(test_mask.sum()),
            "epoch_accuracy": round(epoch_acc, 4),
            "correct": pred_label == true_label,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t0 = time.time()
    subject_results = run_loso(args.embeddings)
    metrics = compute_metrics(subject_results)
    elapsed = round(time.time() - t0, 1)

    result = {
        "method": "logistic_regression_pooled",
        "cv_mode": "loso",
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
