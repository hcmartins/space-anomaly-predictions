"""Regression and classification metrics shared by all three Phase 5 tasks.

Regression (Tasks A, B): MAE, RMSE, MSE, R-squared, bias (mean signed
error, predicted - actual), and Pearson correlation.

Classification (Task C): TSS, HSS, POD, FAR, precision, recall, F1, Brier
score, log score, PR-AUC, ROC-AUC.

FAR here is the meteorological "false alarm RATE" (false positive rate,
FP / (FP + TN)) -- the quantity TSS classically subtracts from POD -- NOT
"false alarm ratio" (FP / (FP + TP), which would be redundant with
precision). See configs/task_storm_detection.yaml's own note.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true

    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    bias = float(np.mean(errors))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        correlation = float("nan")

    return {"mae": mae, "rmse": rmse, "mse": mse, "r2": r2, "bias": bias, "correlation": correlation}


def _confusion_counts(y_true: np.ndarray, y_pred_binary: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred_binary == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred_binary == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_binary == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred_binary == 0)))
    return tp, fp, tn, fn


def classification_metrics(y_true, y_prob, *, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred_binary = (y_prob >= threshold).astype(int)

    tp, fp, tn, fn = _confusion_counts(y_true, y_pred_binary)

    pod = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # a.k.a. recall
    far = fp / (fp + tn) if (fp + tn) > 0 else float("nan")  # false alarm RATE, see module docstring
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = pod
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall) else float("nan")

    tss = (pod - far) if not (np.isnan(pod) or np.isnan(far)) else float("nan")

    hss_denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = (2 * (tp * tn - fp * fn) / hss_denominator) if hss_denominator > 0 else float("nan")

    brier = float(np.mean((y_prob - y_true) ** 2))

    eps = 1e-12
    clipped = np.clip(y_prob, eps, 1 - eps)
    log_score = float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))

    n_classes_present = len(np.unique(y_true))
    if n_classes_present < 2:
        pr_auc = float("nan")
        roc_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))

    return {
        "tss": tss,
        "hss": hss,
        "pod": pod,
        "far": far,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
        "log_score": log_score,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_classes_present_in_y_true": n_classes_present,
    }


def select_threshold(
    y_val_true, y_val_prob, *, grid_min: float = 0.01, grid_max: float = 0.99, grid_steps: int = 99,
) -> tuple[float | None, str]:
    """Eq. 6.1: the decision threshold tau* maximising TSS on VALIDATION data
    only (the test split plays no role). Same fixed grid as the solar pipeline
    (solar/modeling/metrics.py::select_threshold) and the cross-domain fusion
    (cross-domain/evaluation/thresholds.py), so all three chapters use one
    protocol. Returns (None, reason) when validation has < 2 rows, a single
    class, or TSS undefined everywhere -- the caller must then fall back to its
    configured fixed threshold and record that it did."""
    y = np.asarray(y_val_true, dtype=int)
    p = np.asarray(y_val_prob, dtype=float)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None, "validation split has fewer than 2 rows or a single class (TSS undefined)"
    best_t, best = None, -np.inf
    for t in np.linspace(grid_min, grid_max, grid_steps):
        tss = classification_metrics(y, p, threshold=float(t))["tss"]
        if not np.isnan(tss) and tss > best:
            best, best_t = tss, float(t)
    if best_t is None:
        return None, "TSS undefined at every candidate threshold"
    return best_t, f"validation_selected (maximizing tss, val_tss={best:.4f})"
