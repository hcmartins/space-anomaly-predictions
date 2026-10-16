"""Required classification figures: confusion matrix, reliability diagram
(with bin counts), precision-recall curve, ROC curve. Each returns
(figure, source_dataframe) so the caller can save both via
evaluation.figure_provenance.save_figure_with_provenance.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def confusion_matrix_figure(y_true: np.ndarray, y_pred_binary: np.ndarray):
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted negative", "Predicted positive"])
    ax.set_yticklabels(["Actual negative", "Actual positive"])
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, label="count")

    source_df = pd.DataFrame(
        [
            {"actual": "negative", "predicted": "negative", "count": int(cm[0, 0])},
            {"actual": "negative", "predicted": "positive", "count": int(cm[0, 1])},
            {"actual": "positive", "predicted": "negative", "count": int(cm[1, 0])},
            {"actual": "positive", "predicted": "positive", "count": int(cm[1, 1])},
        ]
    )
    return fig, source_df


def reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10):
    """Calibration curve with EXPLICIT BIN COUNTS -- a bin with zero samples
    is reported (count=0, observed_frequency=NaN), never silently omitted."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        mean_predicted = float(np.mean(y_prob[mask])) if count else float("nan")
        observed_frequency = float(np.mean(y_true[mask])) if count else float("nan")
        rows.append(
            {
                "bin": b,
                "bin_lo": bin_edges[b],
                "bin_hi": bin_edges[b + 1],
                "count": count,
                "mean_predicted_probability": mean_predicted,
                "observed_frequency": observed_frequency,
            }
        )
    source_df = pd.DataFrame(rows)

    fig, (ax_curve, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    valid = source_df["count"] > 0
    ax_curve.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfectly calibrated")
    ax_curve.plot(
        source_df.loc[valid, "mean_predicted_probability"],
        source_df.loc[valid, "observed_frequency"],
        marker="o",
    )
    ax_curve.set_ylabel("Observed frequency")
    ax_curve.set_title("Reliability diagram")
    ax_curve.legend(fontsize=8)

    ax_hist.bar(source_df["bin_lo"], source_df["count"], width=1.0 / n_bins, align="edge", color="steelblue")
    ax_hist.set_xlabel("Mean predicted probability (bin)")
    ax_hist.set_ylabel("Bin count")
    return fig, source_df


def pr_curve_figure(y_true: np.ndarray, y_prob: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")

    source_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(thresholds, np.nan),
        }
    )
    return fig, source_df


def roc_curve_figure(y_true: np.ndarray, y_prob: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")

    source_df = pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds})
    return fig, source_df
