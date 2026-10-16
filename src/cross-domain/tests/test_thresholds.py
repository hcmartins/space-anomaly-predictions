import numpy as np

from evaluation.thresholds import select_threshold_by_tss


def test_selects_perfect_threshold_for_perfectly_separable_data():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold, source = select_threshold_by_tss(y_true, y_prob)
    assert 0.3 < threshold <= 0.7
    assert "validation_selected" in source


def test_falls_back_to_fixed_threshold_when_single_class_present():
    y_true = np.zeros(10)
    y_prob = np.linspace(0, 1, 10)
    threshold, source = select_threshold_by_tss(y_true, y_prob)
    assert threshold == 0.5
    assert "fixed_fallback" in source
