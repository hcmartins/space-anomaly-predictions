"""Tests for regression and classification metrics, hand-computed."""
import numpy as np
import pytest

from martian.modeling.metrics import classification_metrics, regression_metrics


def test_regression_metrics_hand_computed():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 2.5, 4.5])
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.5)
    assert m["mse"] == pytest.approx(0.25)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["bias"] == pytest.approx(0.25)


def test_regression_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["r2"] == pytest.approx(1.0)
    assert m["correlation"] == pytest.approx(1.0)


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
