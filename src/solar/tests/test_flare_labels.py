"""Tests for forward-looking flare-occurrence labeling."""
import numpy as np
import pandas as pd

from labeling.flare_labels import (
    build_labels,
    build_primary_label,
    classify_flux,
    load_labeling_config,
    run_horizon_sensitivity,
)

_THRESHOLDS = {"X": 1.0e-4, "M": 1.0e-5, "C": 1.0e-6, "B": 1.0e-7}


def test_classify_flux_maps_to_correct_class():
    assert classify_flux(2.0e-4, _THRESHOLDS) == "X"
    assert classify_flux(2.0e-5, _THRESHOLDS) == "M"
    assert classify_flux(2.0e-6, _THRESHOLDS) == "C"
    assert classify_flux(2.0e-7, _THRESHOLDS) == "B"
    assert classify_flux(2.0e-8, _THRESHOLDS) is None
    assert classify_flux(np.nan, _THRESHOLDS) is None


def _full_hour_goes(spike_time=None, spike_value=None, base_value=1e-8):
    times = pd.date_range("2018-01-01T00:00:00Z", "2018-01-01T01:00:00Z", freq="1min", tz="UTC")
    values = np.full(len(times), base_value)
    if spike_time is not None:
        values[list(times).index(spike_time)] = spike_value
    return pd.DataFrame({"time_utc": times, "xrsb_flux_w_m2": values})


def _config_1h():
    config = load_labeling_config()
    config["forecast_horizon_hours"] = 1
    return config


def test_build_labels_detects_flare_within_horizon():
    aligned = pd.DataFrame([{"time_utc": pd.Timestamp("2018-01-01T00:00:00Z")}])
    goes = _full_hour_goes(spike_time=pd.Timestamp("2018-01-01T00:30:00Z"), spike_value=2.0e-5)  # M class
    out = build_labels(aligned, goes, config=_config_1h())
    row = out.iloc[0]
    assert row["label_max_flare_class_in_horizon"] == "M"
    assert row["label_flare_b_or_above"] == 1.0
    assert row["label_flare_m_or_above"] == 1.0
    assert row["label_flare_x_or_above"] == 0.0
    assert row["label_coverage_fraction"] == 1.0


def test_build_labels_excludes_observation_exactly_at_t0():
    # A huge spike exactly AT t0 must be excluded -- the window is strictly
    # AFTER the current timestamp, never inclusive of it.
    aligned = pd.DataFrame([{"time_utc": pd.Timestamp("2018-01-01T00:00:00Z")}])
    goes = _full_hour_goes(spike_time=pd.Timestamp("2018-01-01T00:00:00Z"), spike_value=9.0e-4)
    out = build_labels(aligned, goes, config=_config_1h())
    assert out.iloc[0]["label_max_flare_class_in_horizon"] is None
    assert out.iloc[0]["label_flare_x_or_above"] == 0.0


def test_build_labels_excludes_observation_after_horizon_end():
    aligned = pd.DataFrame([{"time_utc": pd.Timestamp("2018-01-01T00:00:00Z")}])
    goes = _full_hour_goes()
    # Extra row strictly after the 1h horizon -- must never leak in.
    goes = pd.concat(
        [goes, pd.DataFrame([{"time_utc": pd.Timestamp("2018-01-01T01:01:00Z"), "xrsb_flux_w_m2": 9.0e-4}])],
        ignore_index=True,
    )
    out = build_labels(aligned, goes, config=_config_1h())
    assert out.iloc[0]["label_max_flare_class_in_horizon"] is None


def test_build_labels_undefined_when_coverage_too_low():
    aligned = pd.DataFrame([{"time_utc": pd.Timestamp("2018-01-01T00:00:00Z")}])
    goes = pd.DataFrame(
        {"time_utc": [pd.Timestamp("2018-01-01T00:30:00Z")], "xrsb_flux_w_m2": [9.0e-4]}
    )  # only 1 of 60 expected minutes observed
    out = build_labels(aligned, goes, config=_config_1h())
    row = out.iloc[0]
    assert row["label_max_flare_class_in_horizon"] is None
    assert pd.isna(row["label_flare_x_or_above"])  # UNDEFINED, not fabricated as 0


def test_build_primary_label_surfaces_configured_class():
    aligned = pd.DataFrame([{"time_utc": pd.Timestamp("2018-01-01T00:00:00Z")}])
    goes = _full_hour_goes(spike_time=pd.Timestamp("2018-01-01T00:30:00Z"), spike_value=2.0e-5)
    config = _config_1h()
    config["primary_target_class"] = "M"
    out = build_primary_label(aligned, goes, config=config)
    assert out.iloc[0]["label_primary"] == out.iloc[0]["label_flare_m_or_above"]


def test_run_horizon_sensitivity_covers_all_horizons_and_classes():
    aligned = pd.DataFrame([{"time_utc": pd.Timestamp("2018-01-01T00:00:00Z")}])
    goes = _full_hour_goes(spike_time=pd.Timestamp("2018-01-01T00:30:00Z"), spike_value=2.0e-5)
    config = load_labeling_config()
    config["sensitivity_horizons_hours"] = [1]
    result = run_horizon_sensitivity(aligned, goes, config=config)
    assert set(result["class"]) == {"B", "C", "M", "X"}
    assert len(result) == 4
