"""Tests for per-HARP alignment, hand-computed."""
import pandas as pd

from preprocessing.harp_alignment import build_per_harp_timeline, prepare_sharp_per_harp


def _config(**overrides):
    from preprocessing.alignment import load_alignment_config

    config = load_alignment_config()
    config.update(overrides)
    return config


def _sharp_df():
    return pd.DataFrame([
        {"HARPNUM": 7115, "T_REC": "2017.09.04_00:00:00_TAI", "USFLUX": 1.0, "LAT_FWT": 10.0, "LON_FWT": 20.0},
        {"HARPNUM": 7115, "T_REC": "2017.09.04_00:12:00_TAI", "USFLUX": 1.1, "LAT_FWT": 10.0, "LON_FWT": 22.0},
        {"HARPNUM": 9001, "T_REC": "2017.09.04_00:00:00_TAI", "USFLUX": 2.0, "LAT_FWT": -5.0, "LON_FWT": 75.0},  # beyond limb
    ])


def test_prepare_sharp_per_harp_keeps_one_row_per_harp_timestamp():
    out = prepare_sharp_per_harp(_sharp_df())
    # HARP 9001's row is beyond the default 68-degree limb threshold and dropped.
    assert set(out["HARPNUM"]) == {7115}
    assert len(out) == 2
    assert list(out["time_utc"]) == sorted(out["time_utc"])


def test_prepare_sharp_per_harp_respects_configured_limb_threshold():
    config = _config()
    config["per_harp"]["limb_longitude_threshold_deg"] = 90
    out = prepare_sharp_per_harp(_sharp_df(), config=config)
    assert set(out["HARPNUM"]) == {7115, 9001}


def _goes_df():
    return pd.DataFrame([
        {"time_utc": pd.Timestamp("2017-09-04T00:00:00Z"), "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 1e-7},
        {"time_utc": pd.Timestamp("2017-09-04T00:12:00Z"), "satellite": 16, "xrsa_flux_w_m2": 2e-8, "xrsb_flux_w_m2": 2e-7},
    ])


def test_build_per_harp_timeline_broadcasts_same_goes_state_across_harps():
    sharp = pd.DataFrame([
        {"HARPNUM": 7115, "T_REC": "2017.09.04_00:00:00_TAI", "USFLUX": 1.0, "LAT_FWT": 0.0, "LON_FWT": 0.0},
        {"HARPNUM": 9001, "T_REC": "2017.09.04_00:00:00_TAI", "USFLUX": 2.0, "LAT_FWT": 0.0, "LON_FWT": 0.0},
    ])
    aligned, goes_disk = build_per_harp_timeline(_goes_df(), sharp)
    assert len(aligned) == 2  # both HARPs kept, same T_REC
    # Both HARPs at the same timestamp see the SAME disk-wide GOES trailing state.
    row_7115 = aligned.loc[aligned["HARPNUM"] == 7115].iloc[0]
    row_9001 = aligned.loc[aligned["HARPNUM"] == 9001].iloc[0]
    assert row_7115["goes_xrsb_flux_trailing_max_w_m2"] == row_9001["goes_xrsb_flux_trailing_max_w_m2"]
    assert len(goes_disk) == 2
