"""Tests for the required classification figures and their source data:
confusion matrix, reliability diagram (with bin counts), PR curve, ROC curve."""
import numpy as np

from martian.evaluation.classification_figures import (
    confusion_matrix_figure,
    pr_curve_figure,
    reliability_diagram,
    roc_curve_figure,
)


def test_confusion_matrix_counts_hand_computed():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred_binary = np.array([1, 0, 0, 1, 1, 0])
    _, src = confusion_matrix_figure(y_true, y_pred_binary)
    by_pair = {(r["actual"], r["predicted"]): r["count"] for _, r in src.iterrows()}
    assert by_pair[("positive", "positive")] == 2  # idx0, idx4
    assert by_pair[("positive", "negative")] == 1  # idx1
    assert by_pair[("negative", "positive")] == 1  # idx3
    assert by_pair[("negative", "negative")] == 2  # idx2, idx5
    assert src["count"].sum() == len(y_true)


def test_reliability_diagram_bin_counts_sum_to_n():
    rng = np.random.default_rng(0)
    y_true = (rng.uniform(size=500) > 0.5).astype(int)
    y_prob = rng.uniform(size=500)
    _, src = reliability_diagram(y_true, y_prob, n_bins=10)
    assert src["count"].sum() == 500
    assert len(src) == 10


def test_reliability_diagram_empty_bin_reports_nan_not_fabricated():
    # All probabilities clustered near 0 -- high-probability bins get zero
    # samples and must show NaN observed_frequency, not a fabricated value.
    y_true = np.array([0, 0, 1, 0])
    y_prob = np.array([0.05, 0.05, 0.06, 0.04])
    _, src = reliability_diagram(y_true, y_prob, n_bins=10)
    empty_bins = src[src["count"] == 0]
    assert len(empty_bins) > 0
    assert empty_bins["observed_frequency"].isna().all()


def test_pr_curve_source_has_precision_recall_columns():
    rng = np.random.default_rng(1)
    y_true = (rng.uniform(size=200) > 0.7).astype(int)
    y_prob = rng.uniform(size=200)
    _, src = pr_curve_figure(y_true, y_prob)
    assert {"precision", "recall", "threshold"} <= set(src.columns)
    assert (src["precision"] >= 0).all() and (src["precision"] <= 1).all()


def test_roc_curve_source_has_fpr_tpr_columns():
    rng = np.random.default_rng(2)
    y_true = (rng.uniform(size=200) > 0.7).astype(int)
    y_prob = rng.uniform(size=200)
    _, src = roc_curve_figure(y_true, y_prob)
    assert {"false_positive_rate", "true_positive_rate", "threshold"} <= set(src.columns)
    assert src["false_positive_rate"].iloc[0] <= src["false_positive_rate"].iloc[-1]
