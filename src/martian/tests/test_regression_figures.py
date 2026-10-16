"""Tests for the required regression figures and their source data:
observed-vs-predicted and residual analysis."""
import numpy as np

from martian.evaluation.regression_figures import (
    observed_vs_predicted_figure,
    residual_analysis_figure,
)


def test_observed_vs_predicted_source_matches_input():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    _, src = observed_vs_predicted_figure(y_true, y_pred)
    assert src["observed"].tolist() == y_true.tolist()
    assert src["predicted"].tolist() == y_pred.tolist()


def test_residual_analysis_hand_computed():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 3.5])
    _, src = residual_analysis_figure(y_true, y_pred)
    # residual = predicted - observed
    assert src["residual"].tolist() == [0.5, -0.5, 0.5]


def test_residual_analysis_source_has_all_columns():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=50)
    y_pred = y_true + rng.normal(scale=0.1, size=50)
    _, src = residual_analysis_figure(y_true, y_pred)
    assert {"observed", "predicted", "residual"} <= set(src.columns)
    assert len(src) == 50
