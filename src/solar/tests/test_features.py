"""Tests for strictly causal lag/trailing/diff/slope/persistence features."""
import numpy as np
import pandas as pd

from preprocessing.features import build_features, feature_column_names, load_features_config


def _config(**overrides):
    config = load_features_config()
    config["variables"] = ["USFLUX"]
    config["lags_steps"] = [1, 2]
    config["trailing_window_steps"] = 3
    config["enable_persistence_counter"] = False
    config["enable_qc_indicators"] = False
    config.update(overrides)
    return config


def _regular_series(values):
    return pd.DataFrame(
        [
            {"time_utc": pd.Timestamp("2018-01-01T00:00:00Z") + i * pd.Timedelta(minutes=12), "USFLUX": v}
            for i, v in enumerate(values)
        ]
    )


def test_lag_features_use_exact_prior_values():
    df = _regular_series([10.0, 20.0, 30.0, 40.0])
    out = build_features(df, config=_config())
    row = out.iloc[3]  # value 40.0 at step 3
    assert row["feat_USFLUX_lag1"] == 30.0
    assert row["feat_USFLUX_lag2"] == 20.0


def test_first_row_has_no_lag_history():
    df = _regular_series([10.0, 20.0])
    out = build_features(df, config=_config())
    assert np.isnan(out.iloc[0]["feat_USFLUX_lag1"])


def test_trailing_stats_over_window():
    df = _regular_series([10.0, 20.0, 30.0, 40.0])
    out = build_features(df, config=_config())
    row = out.iloc[3]  # trailing window (steps 1..3 back): 30, 20, 10
    assert row["feat_USFLUX_trailing3_mean"] == 20.0
    assert row["feat_USFLUX_trailing3_max"] == 30.0
    assert row["feat_USFLUX_trailing3_min"] == 10.0


def test_diff_and_slope():
    df = _regular_series([10.0, 20.0, 30.0])
    out = build_features(df, config=_config())
    row = out.iloc[2]  # current 30, lag1 20
    assert row["feat_USFLUX_diff1"] == 10.0
    assert row["feat_USFLUX_slope3"] > 0  # monotonically increasing series


def test_lag_across_dropped_gap_is_nan_not_fabricated():
    # Missing the 00:12 step entirely (e.g. dropped by missing_data.py for
    # exceeding its max fillable gap) -- the lag1 lookup for 00:24 must not
    # silently fall back to the previous ROW (00:00), which would fabricate
    # a false 12-minute lag.
    df = pd.DataFrame(
        [
            {"time_utc": pd.Timestamp("2018-01-01T00:00:00Z"), "USFLUX": 10.0},
            {"time_utc": pd.Timestamp("2018-01-01T00:24:00Z"), "USFLUX": 30.0},
        ]
    )
    out = build_features(df, config=_config())
    row = out.iloc[1]
    assert np.isnan(row["feat_USFLUX_lag1"])


def test_persistence_counter_counts_consecutive_elevated_steps():
    df = pd.DataFrame(
        [
            {"time_utc": pd.Timestamp("2018-01-01T00:00:00Z") + i * pd.Timedelta(minutes=12),
             "goes_xrsb_flux_trailing_max_w_m2": v}
            for i, v in enumerate([1e-7, 2e-6, 3e-6, 4e-6, 1e-7])
        ]
    )
    config = load_features_config()
    config["variables"] = []
    config["enable_lags"] = False
    config["enable_trailing_stats"] = False
    config["enable_first_difference_and_slope"] = False
    config["enable_qc_indicators"] = False
    config["persistence_flux_threshold_w_m2"] = 1.0e-6
    out = build_features(df, config=config)
    # Steps 1,2,3 (values 2e-6,3e-6,4e-6) are >= threshold consecutively;
    # step 3's run counts steps 1 and 2 as PRIOR elevated steps (run=2).
    assert out.iloc[3]["feat_persistence_elevated_flux_run_steps"] == 2
    assert out.iloc[1]["feat_persistence_elevated_flux_run_steps"] == 0  # prior step (0) was below threshold
    assert out.iloc[4]["feat_persistence_elevated_flux_run_steps"] == 3  # steps 1,2,3 all elevated


def test_qc_indicator_passthrough():
    df = _regular_series([10.0, 20.0])
    df["is_observed"] = [True, False]
    df["interpolation_flag"] = [False, True]
    df["gap_length_steps"] = [0, 1]
    config = _config(enable_qc_indicators=True)
    out = build_features(df, config=config)
    assert list(out["feat_is_observed"]) == [1, 0]
    assert list(out["feat_interpolation_flag"]) == [0, 1]


def test_feature_column_names_only_returns_prefixed_columns():
    df = _regular_series([10.0, 20.0])
    out = build_features(df, config=_config())
    names = feature_column_names(out)
    assert all(n.startswith("feat_") for n in names)
    assert "USFLUX" not in names
