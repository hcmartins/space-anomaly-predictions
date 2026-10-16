"""Tests for disk-level alignment: multi-satellite/multi-active-region
aggregation, and the causal (never-future) GOES trailing-window join."""
import pandas as pd

from preprocessing.alignment import (
    aggregate_goes_disk,
    aggregate_sharp_disk,
    attach_goes_trailing_state,
    build_aligned_timeline,
    load_alignment_config,
)


def test_aggregate_goes_disk_takes_max_across_satellites_by_default():
    df = pd.DataFrame([
        {"time_utc": "2018-01-01T00:00:00Z", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 1e-7},
        {"time_utc": "2018-01-01T00:00:00Z", "satellite": 17, "xrsa_flux_w_m2": 2e-8, "xrsb_flux_w_m2": 5e-7},
        {"time_utc": "2018-01-01T00:01:00Z", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 1e-7},
    ])
    out = aggregate_goes_disk(df, load_alignment_config())
    row0 = out.iloc[0]
    assert row0["xrsb_flux_w_m2"] == 5e-7  # max across the two satellites
    assert row0["n_satellites_reporting"] == 2
    assert len(out) == 2


def test_aggregate_sharp_disk_sums_extensive_and_maxes_intensive_keywords():
    df = pd.DataFrame([
        {"HARPNUM": 401, "T_REC": "2018.01.01_00:00:00_TAI", "USFLUX": 1.0e21, "R_VALUE": 2.0},
        {"HARPNUM": 402, "T_REC": "2018.01.01_00:00:00_TAI", "USFLUX": 3.0e21, "R_VALUE": 5.0},
    ])
    out = aggregate_sharp_disk(df, load_alignment_config())
    row0 = out.iloc[0]
    assert row0["USFLUX"] == 4.0e21   # sum (extensive)
    assert row0["R_VALUE"] == 5.0     # max (intensive/complexity)
    assert row0["n_active_regions_tracked"] == 2


def test_attach_goes_trailing_state_is_causal_never_uses_future_goes():
    goes_disk = pd.DataFrame([
        {"time_utc": pd.Timestamp("2018-01-01T00:00:00Z"), "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 1e-7},
        {"time_utc": pd.Timestamp("2018-01-01T00:06:00Z"), "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 3e-7},
        # This one is AFTER the SHARP timestamp below -- must never be included.
        {"time_utc": pd.Timestamp("2018-01-01T00:20:00Z"), "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 9e-4},
    ])
    sharp_disk = pd.DataFrame([{"time_utc": pd.Timestamp("2018-01-01T00:12:00Z"), "USFLUX": 1.0e21}])

    out = attach_goes_trailing_state(sharp_disk, goes_disk, load_alignment_config())
    assert out.iloc[0]["goes_xrsb_flux_trailing_max_w_m2"] == 3e-7  # not 9e-4
    assert out.iloc[0]["goes_n_obs_in_trailing_window"] == 2


def test_attach_goes_trailing_state_no_observations_in_window_is_nan():
    goes_disk = pd.DataFrame([{"time_utc": pd.Timestamp("2018-01-01T00:00:00Z"), "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 1e-7}])
    sharp_disk = pd.DataFrame([{"time_utc": pd.Timestamp("2018-01-02T00:00:00Z"), "USFLUX": 1.0e21}])
    out = attach_goes_trailing_state(sharp_disk, goes_disk, load_alignment_config())
    assert pd.isna(out.iloc[0]["goes_xrsb_flux_trailing_max_w_m2"])
    assert out.iloc[0]["goes_n_obs_in_trailing_window"] == 0


def test_build_aligned_timeline_end_to_end():
    goes_df = pd.DataFrame([
        {"time_utc": "2018-01-01T00:00:00Z", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 1e-7},
        {"time_utc": "2018-01-01T00:11:00Z", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 2e-7},
    ])
    sharp_df = pd.DataFrame([{"HARPNUM": 401, "T_REC": "2018.01.01_00:12:00_TAI", "USFLUX": 1.0e21, "R_VALUE": 2.0}])

    aligned, goes_disk = build_aligned_timeline(goes_df, sharp_df)
    assert len(aligned) == 1
    assert aligned.iloc[0]["goes_xrsb_flux_trailing_max_w_m2"] == 2e-7
    assert len(goes_disk) == 2
