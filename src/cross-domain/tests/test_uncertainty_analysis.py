"""Tests for uncertainty_analysis.py (Section 5.6.1's three-component
protocol: seed variation, block bootstrap, paired comparison) against
synthetic data with planted structure -- this script does no training of
its own, so correctness here is entirely about the statistics, not about
any model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import uncertainty_analysis as ua


# ---------------------------------------------------------------- metrics


def test_true_skill_statistic_hand_computed():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert ua.true_skill_statistic(y_true, y_pred) == 1.0


def test_true_skill_statistic_undefined_when_one_class_absent():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 0, 1])
    assert np.isnan(ua.true_skill_statistic(y_true, y_pred))


# ---------------------------------------------------------------- block bootstrap mechanics


def test_block_indices_length_and_contiguity():
    rng = np.random.default_rng(0)
    idx = ua.block_indices(n=50, block_length=5, rng=rng)
    assert len(idx) == 50
    # Every run of 5 consecutive output positions is itself a run of 5
    # CONSECUTIVE input indices (a block), possibly wrapping only at the
    # trailing truncation -- verify the first 9 full blocks (45 of 50).
    for b in range(9):
        block = idx[b * 5:(b + 1) * 5]
        assert np.array_equal(block, np.arange(block[0], block[0] + 5))


def test_block_indices_length_one_reduces_to_ordinary_iid_bootstrap():
    rng = np.random.default_rng(0)
    idx = ua.block_indices(n=30, block_length=1, rng=rng)
    assert len(idx) == 30
    assert idx.min() >= 0 and idx.max() < 30


def test_bootstrap_tss_recovers_planted_skill_on_a_near_perfect_classifier():
    rng = np.random.default_rng(1)
    n = 2000
    y_true = (rng.random(n) > 0.8).astype(int)  # 20% prevalence, planted
    # A near-perfect classifier: correct 95% of the time, independent noise.
    flip = rng.random(n) < 0.05
    y_pred_prob = np.where(flip, 1 - y_true, y_true).astype(float)
    point = ua.tss_from_probs(y_true, y_pred_prob, threshold=0.5)
    assert point > 0.85  # sanity: the planted classifier really is good

    dist = ua.bootstrap_tss(y_true, y_pred_prob, threshold=0.5, block_length=1,
                             resamples=300, rng=np.random.default_rng(2))
    lo, hi = ua.percentile_interval(dist)
    assert lo <= point <= hi  # the point estimate falls inside its own bootstrap CI
    assert hi - lo < 0.15     # 2000 near-iid rows -> a tight interval, not a vacuous one


def test_paired_bootstrap_difference_is_near_zero_for_identical_predictions():
    rng = np.random.default_rng(3)
    n = 500
    y_true = (rng.random(n) > 0.5).astype(int)
    y_prob = np.clip(y_true * 0.7 + rng.random(n) * 0.3, 0, 1)
    a = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "threshold": 0.5})
    b = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "threshold": 0.5})
    dist = ua.paired_bootstrap_difference(a, b, block_length=1, resamples=300)
    lo, hi = ua.percentile_interval(dist)
    assert not ua.established((lo, hi))  # identical models -> never a real difference


def test_paired_bootstrap_difference_establishes_a_real_planted_gap():
    rng = np.random.default_rng(4)
    n = 1000
    y_true = (rng.random(n) > 0.5).astype(int)
    # Config a: strong signal. Config b: near-random.
    y_prob_a = np.clip(y_true * 0.8 + rng.random(n) * 0.15, 0, 1)
    y_prob_b = np.clip(0.5 + (rng.random(n) - 0.5) * 0.1, 0, 1)
    a = pd.DataFrame({"y_true": y_true, "y_prob": y_prob_a, "threshold": 0.5})
    b = pd.DataFrame({"y_true": y_true, "y_prob": y_prob_b, "threshold": 0.5})
    dist = ua.paired_bootstrap_difference(a, b, block_length=1, resamples=300)
    lo, hi = ua.percentile_interval(dist)
    assert ua.established((lo, hi))
    assert lo > 0  # a is planted strictly better than b


def test_paired_bootstrap_difference_rejects_mismatched_partitions():
    a = pd.DataFrame({"y_true": [1, 0, 1], "y_prob": [0.9, 0.1, 0.8], "threshold": 0.5})
    b = pd.DataFrame({"y_true": [1, 0], "y_prob": [0.9, 0.1], "threshold": 0.5})
    with pytest.raises(ValueError):
        ua.paired_bootstrap_difference(a, b, block_length=1, resamples=10)


def test_paired_bootstrap_difference_rejects_differing_labels():
    a = pd.DataFrame({"y_true": [1, 0, 1], "y_prob": [0.9, 0.1, 0.8], "threshold": 0.5})
    b = pd.DataFrame({"y_true": [1, 1, 0], "y_prob": [0.9, 0.1, 0.8], "threshold": 0.5})
    with pytest.raises(ValueError):
        ua.paired_bootstrap_difference(a, b, block_length=1, resamples=10)


# ---------------------------------------------------------------- loading


def test_load_parses_filenames_and_skips_unrecognised(tmp_path):
    df = pd.DataFrame({"y_true": [1, 0], "y_prob": [0.9, 0.1], "threshold": 0.5, "regime": ["2020", "2020"]})
    df.to_csv(tmp_path / "predictions_A7_solar_seed44.csv", index=False)
    df.to_csv(tmp_path / "predictions_A1_martian_seed03.csv", index=False)
    (tmp_path / "not_a_predictions_file.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    frames = ua.load(tmp_path)
    assert set(frames.keys()) == {ua.Key("A7", "solar", 44), ua.Key("A1", "martian", 3)}


def test_load_defaults_missing_regime_column_to_all(tmp_path):
    pd.DataFrame({"y_true": [1, 0], "y_prob": [0.9, 0.1], "threshold": 0.5}).to_csv(
        tmp_path / "predictions_A2_martian_seed44.csv", index=False
    )
    frames = ua.load(tmp_path)
    key = ua.Key("A2", "martian", 44)
    assert (frames[key]["regime"] == "all").all()


def test_load_raises_on_missing_required_columns(tmp_path):
    pd.DataFrame({"y_true": [1, 0], "threshold": 0.5}).to_csv(  # y_prob missing
        tmp_path / "predictions_A1_solar_seed44.csv", index=False
    )
    with pytest.raises(ValueError):
        ua.load(tmp_path)


# ---------------------------------------------------------------- report structure


def _write_predictions(tmp_path, *, config, domain, seed, n=200, tss_level=0.8, block_length_hint=None, regime="R"):
    # y_true depends only on (domain, seed) -- every config sharing a
    # (domain, seed) test partition must see IDENTICAL labels, exactly as
    # paired_bootstrap_difference requires (real test instances are the
    # same across configs; only the model differs).
    label_rng = np.random.default_rng(seed * 1000 + (hash(domain) % 997))
    y_true = (label_rng.random(n) > 0.5).astype(int)
    prob_rng = np.random.default_rng(seed * 1000 + (hash((config, domain)) % 997) + 500_000)
    noise = 1.0 - tss_level
    y_prob = np.clip(y_true * tss_level + prob_rng.random(n) * noise, 0, 1)
    pd.DataFrame({
        "y_true": y_true, "y_prob": y_prob, "threshold": 0.5,
        "regime": [regime] * n,
    }).to_csv(tmp_path / f"predictions_{config}_{domain}_seed{seed:02d}.csv", index=False)


def test_build_report_flags_fewer_than_five_seeds(tmp_path):
    for seed in (44, 45):  # only 2 seeds -- below the Section 5.6.1 minimum of 5
        _write_predictions(tmp_path, config="A7", domain="solar", seed=seed)
        _write_predictions(tmp_path, config="A1", domain="solar", seed=seed)
    frames = ua.load(tmp_path)
    report = ua.build_report(frames, {"solar": 1}, resamples=50, comparisons=[("A7", "A1")])
    assert "Fewer than five seeds are present" in report


def test_build_report_does_not_flag_when_five_seeds_present(tmp_path):
    for seed in (44, 45, 46, 47, 48):
        _write_predictions(tmp_path, config="A7", domain="solar", seed=seed)
        _write_predictions(tmp_path, config="A1", domain="solar", seed=seed)
    frames = ua.load(tmp_path)
    report = ua.build_report(frames, {"solar": 1}, resamples=50, comparisons=[("A7", "A1")])
    assert "Fewer than five seeds are present" not in report


def test_build_report_uses_the_block_length_supplied_for_each_domain(tmp_path):
    for domain in ("solar", "martian"):
        for seed in (44, 45, 46, 47, 48):
            _write_predictions(tmp_path, config="A7", domain=domain, seed=seed)
            _write_predictions(tmp_path, config="A1", domain=domain, seed=seed)
    frames = ua.load(tmp_path)
    report = ua.build_report(frames, {"solar": 120, "martian": 1}, resamples=20, comparisons=[("A7", "A1")])
    assert "**solar** (block length = 120 instances)" in report
    assert "**martian** (block length = 1 instances)" in report


def test_build_report_raises_a_clear_error_for_a_domain_with_no_block_length(tmp_path):
    _write_predictions(tmp_path, config="A7", domain="solar", seed=44)
    _write_predictions(tmp_path, config="A1", domain="solar", seed=44)
    frames = ua.load(tmp_path)
    with pytest.raises(KeyError):
        ua.build_report(frames, {}, resamples=20, comparisons=[("A7", "A1")])


def test_default_block_lengths_match_the_real_solar_and_martian_forecast_windows():
    # Solar: 12-minute cadence (src/solar/configs/goes_preprocessing.yaml),
    # 24h horizon (src/solar/configs/labeling.yaml) -> 120 rows.
    # Martian: 1 row = 1 sol, 1-sol horizon -> 1 row.
    assert ua.DEFAULT_BLOCK_LENGTHS == {"solar": 120, "martian": 1}
