"""Tests for SHARP Winsorize + robust-scale preprocessing (Chapter 3.4.2,
Algorithm 3.2)."""
import numpy as np
import pandas as pd

from preprocessing.scaling import apply_robust_scaling, fit_robust_scaling, sharp_feature_columns


def _features_config(**overrides):
    config = {
        "trailing_window_steps": 3,
        "lags_steps": [1, 2],
        "enable_lags": True,
        "enable_trailing_stats": True,
        "enable_first_difference_and_slope": True,
    }
    config.update(overrides)
    return config


def _scaling_config(**overrides):
    config = {"winsorize_lower_percentile": 1.0, "winsorize_upper_percentile": 99.0}
    config.update(overrides)
    return config


def test_sharp_feature_columns_selects_only_sharp_derived_names():
    manifest = [
        "feat_USFLUX_lag1",
        "feat_USFLUX_trailing3_mean",
        "feat_R_VALUE_diff1",
        "feat_R_VALUE_slope3",
        # not SHARP-derived: GOES-trailing source var, active-region count,
        # QC indicators -- must NOT be selected.
        "feat_goes_xrsb_flux_trailing_max_w_m2_lag1",
        "feat_n_active_regions_tracked_lag1",
        "feat_is_observed",
        "feat_persistence_elevated_flux_run_steps",
    ]
    selected = sharp_feature_columns(manifest, _features_config())
    assert selected == ["feat_USFLUX_lag1", "feat_USFLUX_trailing3_mean", "feat_R_VALUE_diff1", "feat_R_VALUE_slope3"]


def test_sharp_feature_columns_also_selects_bare_current_state_names():
    # modeling/datasets.py::build_flare_forecast_dataset deliberately puts
    # each variable's raw CURRENT value in feature_manifest too
    # (raw_current_state_cols), not just its feat_-prefixed derivatives --
    # a real full-range run found one of these (bare "SAVNCPP") carrying an
    # unscaled ~2.18e38 value straight into logistic regression even after
    # the derived columns were fixed. Bare SHARP names must be recognized;
    # bare non-SHARP names (n_active_regions_tracked, GOES trailing vars)
    # must not.
    manifest = ["USFLUX", "R_VALUE", "n_active_regions_tracked", "goes_xrsb_flux_trailing_max_w_m2"]
    assert sharp_feature_columns(manifest, _features_config()) == ["USFLUX", "R_VALUE"]


def test_sharp_feature_columns_only_returns_columns_actually_present():
    # USFLUX has lag/trailing columns configured but only feat_USFLUX_lag1
    # is actually present in this manifest -- must not invent the rest.
    manifest = ["feat_USFLUX_lag1"]
    assert sharp_feature_columns(manifest, _features_config()) == ["feat_USFLUX_lag1"]


def test_fit_robust_scaling_hand_computed():
    # Train values 1..10 (median 5.5, Q1 3.25, Q3 7.75 by linear
    # interpolation, IQR 4.5); 1st/99th percentile winsorize bounds barely
    # clip the extremes at this small n.
    values = np.arange(1, 11, dtype=float)
    X_train = pd.DataFrame({"feat_USFLUX_lag1": values})
    params = fit_robust_scaling(X_train, ["feat_USFLUX_lag1"], _scaling_config())

    lower, upper = np.nanpercentile(values, [1.0, 99.0])
    clipped = np.clip(values, lower, upper)
    expected_median = float(np.median(clipped))
    expected_iqr = float(np.percentile(clipped, 75) - np.percentile(clipped, 25))

    p = params["feat_USFLUX_lag1"]
    assert p["lower"] == lower and p["upper"] == upper
    assert p["median"] == expected_median
    assert p["scale"] == expected_iqr


def test_apply_robust_scaling_clips_then_centers_and_scales():
    X_train = pd.DataFrame({"feat_USFLUX_lag1": np.arange(1, 11, dtype=float)})
    params = fit_robust_scaling(X_train, ["feat_USFLUX_lag1"], _scaling_config())

    # A held-out (val/test) value far above the train winsorize upper bound
    # must be clipped to that bound before scaling -- never extrapolated.
    X_other = pd.DataFrame({"feat_USFLUX_lag1": [1000.0]})
    scaled = apply_robust_scaling(X_other, params)
    expected = (params["feat_USFLUX_lag1"]["upper"] - params["feat_USFLUX_lag1"]["median"]) / params[
        "feat_USFLUX_lag1"
    ]["scale"]
    assert scaled["feat_USFLUX_lag1"].iloc[0] == expected


def test_apply_robust_scaling_leaves_unrelated_columns_untouched():
    X_train = pd.DataFrame({"feat_USFLUX_lag1": np.arange(1, 11, dtype=float)})
    params = fit_robust_scaling(X_train, ["feat_USFLUX_lag1"], _scaling_config())

    X_other = pd.DataFrame({"feat_USFLUX_lag1": [5.0], "feat_n_active_regions_tracked_lag1": [3.0]})
    scaled = apply_robust_scaling(X_other, params)
    assert scaled["feat_n_active_regions_tracked_lag1"].iloc[0] == 3.0


def test_apply_robust_scaling_never_mutates_the_input_frame():
    X_train = pd.DataFrame({"feat_USFLUX_lag1": np.arange(1, 11, dtype=float)})
    params = fit_robust_scaling(X_train, ["feat_USFLUX_lag1"], _scaling_config())
    X_other = pd.DataFrame({"feat_USFLUX_lag1": [5.0]})
    apply_robust_scaling(X_other, params)
    assert X_other["feat_USFLUX_lag1"].iloc[0] == 5.0


def test_fit_robust_scaling_all_nan_column_is_not_fabricated():
    X_train = pd.DataFrame({"feat_USFLUX_lag1": [np.nan, np.nan, np.nan]})
    params = fit_robust_scaling(X_train, ["feat_USFLUX_lag1"], _scaling_config())
    assert "feat_USFLUX_lag1" not in params

    # apply_robust_scaling then leaves the column untouched (still NaN, not
    # a fabricated 0) since it never received fitted params.
    X_other = pd.DataFrame({"feat_USFLUX_lag1": [np.nan]})
    scaled = apply_robust_scaling(X_other, params)
    assert np.isnan(scaled["feat_USFLUX_lag1"].iloc[0])


def test_fit_robust_scaling_constant_column_gets_scale_one_not_divide_by_zero():
    X_train = pd.DataFrame({"feat_R_VALUE_diff1": [3.0] * 10})
    params = fit_robust_scaling(X_train, ["feat_R_VALUE_diff1"], _scaling_config())
    p = params["feat_R_VALUE_diff1"]
    assert p["scale"] == 1.0

    X_other = pd.DataFrame({"feat_R_VALUE_diff1": [3.0]})
    scaled = apply_robust_scaling(X_other, params)
    assert scaled["feat_R_VALUE_diff1"].iloc[0] == 0.0


def test_apply_robust_scaling_copy_false_mutates_in_place_and_returns_same_object():
    # Real fix for a real ArrayMemoryError at the per-HARP pipeline's
    # ~1.97M-row train split (2026-07-28): modeling/train.py uses
    # copy=False on frames it owns and never reads again, to avoid an
    # extra full-frame copy. Must actually mutate in place (not silently
    # behave like copy=True).
    X_train = pd.DataFrame({"feat_USFLUX_lag1": np.arange(1, 11, dtype=float)})
    params = fit_robust_scaling(X_train, ["feat_USFLUX_lag1"], _scaling_config())
    X_other = pd.DataFrame({"feat_USFLUX_lag1": [1000.0]})
    result = apply_robust_scaling(X_other, params, copy=False)
    assert result is X_other
    expected = (params["feat_USFLUX_lag1"]["upper"] - params["feat_USFLUX_lag1"]["median"]) / params[
        "feat_USFLUX_lag1"
    ]["scale"]
    assert X_other["feat_USFLUX_lag1"].iloc[0] == expected


def test_fit_robust_scaling_ignores_nans_when_computing_statistics():
    values = [np.nan] + list(np.arange(1, 11, dtype=float))
    X_train = pd.DataFrame({"feat_USFLUX_lag1": values})
    params = fit_robust_scaling(X_train, ["feat_USFLUX_lag1"], _scaling_config())
    dense = np.arange(1, 11, dtype=float)
    expected_median = float(np.median(np.clip(dense, *np.nanpercentile(dense, [1.0, 99.0]))))
    assert params["feat_USFLUX_lag1"]["median"] == expected_median
