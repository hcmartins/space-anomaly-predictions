"""Tests for comparing a learned model against its task's appropriate
baseline, including correct handling of lower-is-better metrics."""
import numpy as np

from martian.evaluation.baseline_comparison import compare_to_baseline


def test_higher_is_better_metric_skill_sign():
    model = {"r2": 0.8}
    baseline = {"r2": 0.4}
    df = compare_to_baseline(model, baseline)
    row = df.set_index("metric").loc["r2"]
    assert row["model_beats_baseline"]
    assert row["relative_skill"] > 0


def test_lower_is_better_metric_skill_sign():
    model = {"mae": 0.01}
    baseline = {"mae": 0.02}
    df = compare_to_baseline(model, baseline)
    row = df.set_index("metric").loc["mae"]
    assert row["model_beats_baseline"]
    assert row["relative_skill"] > 0


def test_model_worse_than_baseline_flagged_false():
    model = {"mae": 0.05}
    baseline = {"mae": 0.02}
    df = compare_to_baseline(model, baseline)
    row = df.set_index("metric").loc["mae"]
    assert not row["model_beats_baseline"]
    assert row["relative_skill"] < 0


def test_far_is_treated_as_lower_is_better():
    model = {"far": 0.1}
    baseline = {"far": 0.3}
    df = compare_to_baseline(model, baseline)
    row = df.set_index("metric").loc["far"]
    assert row["model_beats_baseline"]


def test_non_numeric_and_missing_keys_skipped():
    model = {"mae": 0.01, "n_classes_present_in_y_true": 2, "extra_only_in_model": 5}
    baseline = {"mae": 0.02, "n_classes_present_in_y_true": 2}
    df = compare_to_baseline(model, baseline)
    assert "extra_only_in_model" not in set(df["metric"])
    assert "mae" in set(df["metric"])


def test_nan_metric_produces_nan_skill_not_a_crash():
    model = {"correlation": float("nan")}
    baseline = {"correlation": 0.5}
    df = compare_to_baseline(model, baseline)
    row = df.set_index("metric").loc["correlation"]
    assert np.isnan(row["relative_skill"])
    assert row["model_beats_baseline"] is None
