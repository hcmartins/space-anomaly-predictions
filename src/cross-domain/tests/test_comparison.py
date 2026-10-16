"""evaluate_predictions's predictions-CSV dump (Section 5.6.1's uncertainty
protocol input format) -- the "three lines" uncertainty_analysis.py's own
docstring specifies, exercised here rather than only by inspection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.comparison import evaluate_predictions


def _synthetic_arrays():
    rng = np.random.default_rng(0)
    n_train, n_val, n_test = 40, 20, 20
    y_train = (rng.random(n_train) > 0.5).astype(int)
    y_val = (rng.random(n_val) > 0.5).astype(int)
    y_test = (rng.random(n_test) > 0.5).astype(int)
    # Fused/unimodal probabilities correlated with the true label but not
    # identical, so threshold selection/TSS are non-degenerate.
    y_prob_fused_val = np.clip(y_val * 0.6 + rng.random(n_val) * 0.3, 0, 1)
    y_prob_fused_test = np.clip(y_test * 0.6 + rng.random(n_test) * 0.3, 0, 1)
    y_prob_unimodal_val = np.clip(y_val * 0.5 + rng.random(n_val) * 0.4, 0, 1)
    y_prob_unimodal_test = np.clip(y_test * 0.5 + rng.random(n_test) * 0.4, 0, 1)
    regime_test = np.array(["2020", "2021"] * (n_test // 2))
    return dict(
        y_train=y_train, y_val=y_val, y_test=y_test,
        y_prob_fused_val=y_prob_fused_val, y_prob_fused_test=y_prob_fused_test,
        y_prob_unimodal_val=y_prob_unimodal_val, y_prob_unimodal_test=y_prob_unimodal_test,
        regime_test=regime_test,
    )


def test_evaluate_predictions_does_not_write_a_csv_when_dump_args_are_omitted(tmp_path):
    a = _synthetic_arrays()
    evaluate_predictions(
        a["y_train"], a["y_val"], a["y_test"],
        a["y_prob_fused_val"], a["y_prob_fused_test"],
        a["y_prob_unimodal_val"], a["y_prob_unimodal_test"],
        a["regime_test"],
    )
    assert list(tmp_path.iterdir()) == []  # nothing written anywhere by default


def test_evaluate_predictions_writes_predictions_csv_in_uncertainty_analysis_format(tmp_path):
    a = _synthetic_arrays()
    result = evaluate_predictions(
        a["y_train"], a["y_val"], a["y_test"],
        a["y_prob_fused_val"], a["y_prob_fused_test"],
        a["y_prob_unimodal_val"], a["y_prob_unimodal_test"],
        a["regime_test"],
        predictions_dir=tmp_path, config_id="A7", domain="solar", seed=44,
    )
    out_path = tmp_path / "predictions_A7_solar_seed44.csv"
    assert out_path.exists()

    df = pd.read_csv(out_path)
    assert list(df.columns) == ["y_true", "y_prob", "threshold", "regime"]
    assert len(df) == len(a["y_test"])
    np.testing.assert_array_equal(df["y_true"].to_numpy(), a["y_test"])
    np.testing.assert_allclose(df["y_prob"].to_numpy(), a["y_prob_fused_test"])
    # threshold is a single scalar (the fused model's own selected threshold),
    # broadcast identically to every row, not per-row.
    assert df["threshold"].nunique() == 1
    assert float(df["threshold"].iloc[0]) == result["threshold_fused"]
    np.testing.assert_array_equal(df["regime"].astype(str).to_numpy(), a["regime_test"].astype(str))


def test_evaluate_predictions_creates_predictions_dir_if_missing(tmp_path):
    a = _synthetic_arrays()
    nested = tmp_path / "does" / "not" / "exist" / "yet"
    evaluate_predictions(
        a["y_train"], a["y_val"], a["y_test"],
        a["y_prob_fused_val"], a["y_prob_fused_test"],
        a["y_prob_unimodal_val"], a["y_prob_unimodal_test"],
        a["regime_test"],
        predictions_dir=nested, config_id="A1", domain="martian", seed=45,
    )
    assert (nested / "predictions_A1_martian_seed45.csv").exists()


def test_evaluate_predictions_partial_dump_args_do_not_write(tmp_path):
    """All four of predictions_dir/config_id/domain/seed are required --
    a partially-specified call must not silently write a malformed or
    wrongly-named file."""
    a = _synthetic_arrays()
    evaluate_predictions(
        a["y_train"], a["y_val"], a["y_test"],
        a["y_prob_fused_val"], a["y_prob_fused_test"],
        a["y_prob_unimodal_val"], a["y_prob_unimodal_test"],
        a["regime_test"],
        predictions_dir=tmp_path, config_id="A7",  # domain, seed omitted
    )
    assert list(tmp_path.iterdir()) == []
