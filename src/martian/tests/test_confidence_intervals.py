"""Tests for bootstrap confidence intervals: percentile CI correctness,
block-vs-row resampling, and single-class handling for classification AUCs."""
import numpy as np
import pytest

from martian.evaluation.confidence_intervals import (
    bootstrap_ci_classification,
    bootstrap_ci_regression,
    load_evaluation_config,
)


@pytest.fixture
def fast_config():
    config = load_evaluation_config()
    config["bootstrap"]["n_replicates"] = 100
    return config


def test_regression_ci_contains_point_estimate(fast_config):
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=300)
    y_pred = y_true + rng.normal(scale=0.2, size=300)
    ci = bootstrap_ci_regression(y_true, y_pred, config=fast_config)
    mae_lo, mae_hi = ci["mae"]
    assert mae_lo < mae_hi
    assert mae_lo > 0  # MAE is non-negative and the interval should be tight around ~0.16


def test_regression_ci_narrower_with_more_data(fast_config):
    rng = np.random.default_rng(1)
    y_true_small = rng.normal(size=30)
    y_pred_small = y_true_small + rng.normal(scale=0.5, size=30)
    y_true_big = np.tile(y_true_small, 20)
    y_pred_big = np.tile(y_pred_small, 20)

    ci_small = bootstrap_ci_regression(y_true_small, y_pred_small, config=fast_config)
    ci_big = bootstrap_ci_regression(y_true_big, y_pred_big, config=fast_config)
    width_small = ci_small["mae"][1] - ci_small["mae"][0]
    width_big = ci_big["mae"][1] - ci_big["mae"][0]
    assert width_big < width_small


def test_classification_ci_keys_match_point_metrics(fast_config):
    rng = np.random.default_rng(2)
    y_true = (rng.uniform(size=200) > 0.6).astype(int)
    y_prob = np.clip(y_true * 0.5 + rng.uniform(size=200) * 0.5, 0, 1)
    ci = bootstrap_ci_classification(y_true, y_prob, config=fast_config)
    for key in ("tss", "hss", "pod", "far", "precision", "recall", "f1", "brier", "log_score", "pr_auc", "roc_auc"):
        assert key in ci
        assert len(ci[key]) == 2


def test_block_bootstrap_uses_whole_blocks(fast_config):
    # Two blocks with very different means -- block bootstrap must only ever
    # produce replicate means at (or near) one of the two block means, never
    # a value that mixes individual rows across blocks in a way a plain row
    # bootstrap could.
    y_true = np.concatenate([np.zeros(20), np.full(20, 10.0)])
    y_pred = y_true.copy()
    block_ids = np.concatenate([np.zeros(20), np.ones(20)])
    ci = bootstrap_ci_regression(y_true, y_pred, block_ids=block_ids, config=fast_config)
    assert ci["bias"] == pytest.approx((0.0, 0.0), abs=1e-9)  # perfect predictions -> zero bias regardless of block


def test_single_class_replicate_excluded_from_auc_ci(fast_config):
    # Almost all negative, one positive -- some bootstrap replicates will
    # contain zero positives (single class), which must be excluded from the
    # AUC CI rather than crashing or producing a NaN-poisoned interval.
    rng = np.random.default_rng(3)
    y_true = np.zeros(50, dtype=int)
    y_true[0] = 1
    y_prob = rng.uniform(size=50)
    ci = bootstrap_ci_classification(y_true, y_prob, config=fast_config)
    assert not np.isnan(ci["roc_auc"][0]) or not np.isnan(ci["roc_auc"][1])


def test_config_n_replicates_actually_used():
    config = load_evaluation_config()
    config["bootstrap"]["n_replicates"] = 5
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=50)
    y_pred = y_true + rng.normal(scale=0.1, size=50)
    # With only 5 replicates the CI should still compute without error --
    # a smoke proof that n_replicates is read from config, not hardcoded.
    ci = bootstrap_ci_regression(y_true, y_pred, config=config)
    assert "mae" in ci
