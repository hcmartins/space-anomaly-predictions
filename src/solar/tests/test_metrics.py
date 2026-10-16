"""Tests for classification metrics, hand-computed."""
import numpy as np
import pytest

from modeling.metrics import classification_metrics, select_threshold


def test_classification_metrics_hand_computed():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.4, 0.2, 0.8, 0.7, 0.1])
    m = classification_metrics(y_true, y_prob, threshold=0.5)
    # pred_binary = [1,0,0,1,1,0] -> TP=2 (idx0,4), FP=1 (idx3), TN=2 (idx2,5), FN=1 (idx1)
    assert m["tp"] == 2 and m["fp"] == 1 and m["tn"] == 2 and m["fn"] == 1
    assert m["pod"] == pytest.approx(2 / 3)
    assert m["far"] == pytest.approx(1 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["tss"] == pytest.approx(2 / 3 - 1 / 3)


def test_classification_metrics_single_class_gives_nan_auc():
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    m = classification_metrics(y_true, y_prob)
    assert np.isnan(m["pr_auc"])
    assert np.isnan(m["roc_auc"])
    assert m["n_classes_present_in_y_true"] == 1


def test_classification_metrics_perfect_prediction():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.99, 0.99, 0.01, 0.01])
    m = classification_metrics(y_true, y_prob)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["brier"] < 0.001


def test_far_is_false_positive_rate_not_false_alarm_ratio():
    # 3 negatives, 1 falsely predicted positive -> FAR (false positive rate)
    # should be 1/3, NOT the false-alarm-RATIO (which would be FP/(FP+TP)).
    y_true = np.array([0, 0, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.1, 0.9])
    m = classification_metrics(y_true, y_prob)
    assert m["fp"] == 1 and m["tn"] == 2
    assert m["far"] == pytest.approx(1 / 3)


def test_select_threshold_finds_the_tss_maximizing_cutoff():
    # y_prob=0.2 (true=0) is the only point separating a poor cutoff from a
    # good one: excluding it (threshold in (0.20, 0.30]) gives TSS=2/3 (the
    # best achievable here); including it (threshold <= 0.20) gives TSS=0.
    # Hand-verified for several thresholds:
    #   0.20 -> pred=[0,1,1,1,1,1] -> TP=3,FP=2,TN=0,FN=0 -> TSS=0
    #   0.21..0.30 -> pred=[0,0,1,1,1,1] -> TP=3,FP=1,TN=2,FN=0 -> TSS=2/3
    #   0.31 -> pred=[0,0,1,0,1,1] -> TP=2,FP=1,TN=2,FN=1 -> TSS=1/3
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.3, 0.7, 0.8])
    threshold, best_metrics = select_threshold(y_true, y_prob, metric="tss", grid_min=0.01, grid_max=0.99, grid_steps=99)
    assert 0.20 < threshold <= 0.30
    assert best_metrics["tss"] == pytest.approx(2 / 3)


def test_select_threshold_returns_none_for_single_class_val():
    threshold, best_metrics = select_threshold(np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9]))
    assert threshold is None and best_metrics is None


def test_select_threshold_returns_none_for_too_few_rows():
    threshold, best_metrics = select_threshold(np.array([1]), np.array([0.9]))
    assert threshold is None and best_metrics is None
