"""Classification metrics for the flare-occurrence forecasting task(s).

TSS (True Skill Statistic, a.k.a. the Hanssen-Kuipers/Peirce Skill Score)
and HSS (Heidke Skill Score) are the DE FACTO standard verification metrics
in the published solar-flare-forecasting literature (e.g. Bloomfield et al.
2012; Barnes et al. 2016's cross-method comparison uses TSS as its primary
ranking metric) -- included here for that reason, not just ported from
`martian/modeling/metrics.py` for consistency's sake, though the formulas
and the rest of the metric set (POD, FAR, precision, recall, F1, Brier,
log score, PR-AUC, ROC-AUC) are in fact a direct, unmodified port -- fully
generic binary-classification metrics with no martian-specific coupling.

FAR here is the meteorological "false alarm RATE" (false positive rate,
FP / (FP + TN)) -- the quantity TSS classically subtracts from POD -- NOT
"false alarm ratio" (FP / (FP + TP), which would be numerically redundant
with precision). RESEARCHER VERIFICATION REQUIRED if the thesis text uses
"FAR" to mean the ratio instead of the rate.1
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0 and not np.isnan(precision) and not np.isnan(recall)
        else float("nan")
    )

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
    y_val_true, y_val_prob, *, metric: str = "tss", grid_min: float = 0.01, grid_max: float = 0.99, grid_steps: int = 99
) -> tuple[float | None, dict | None]:
    """Selects the decision threshold tau* that maximizes `metric` on
    VALIDATION data only (never train or test) -- Chapter 3.6.2/3.8's
    documented threshold-selection protocol: "conversion to a binary alert
    uses a threshold tau* selected on validation data only... the test set
    plays no role in threshold selection." Sweeps a fixed grid rather than
    the empirical val-set probability values themselves, so the selected
    threshold is a config-driven, reproducible constant rather than an
    artifact of this particular val sample's exact probability values.

    Returns (None, None) if val has fewer than 2 rows or only one class
    present -- TSS (and every other metric here) is undefined without both
    classes, so no meaningful threshold can be selected; the caller must
    fall back to a fixed default threshold in this case, never silently
    pick an arbitrary one."""
    y_val_true = np.asarray(y_val_true, dtype=int)
    if len(y_val_true) < 2 or len(np.unique(y_val_true)) < 2:
        return None, None

    best_threshold = None
    best_score = -np.inf
    best_metrics = None
    for t in np.linspace(grid_min, grid_max, grid_steps):
        m = classification_metrics(y_val_true, y_val_prob, threshold=float(t))
        score = m[metric]
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_threshold = float(t)
            best_metrics = m
    return best_threshold, best_metrics
