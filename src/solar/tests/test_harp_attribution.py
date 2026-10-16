"""Tests for NOAA AR -> HARPNUM attribution, hand-computed."""
import pandas as pd

from preprocessing.harp_attribution import (
    attribute_flares_to_harps,
    attribution_quality_report,
    build_harp_ar_index,
    parse_noaa_ars,
)


def test_parse_noaa_ars_splits_comma_separated_list():
    assert parse_noaa_ars("12673,12674") == {"12673", "2673", "12674", "2674"}


def test_parse_noaa_ars_handles_empty_and_nan():
    assert parse_noaa_ars("") == set()
    assert parse_noaa_ars(None) == set()
    assert parse_noaa_ars(float("nan")) == set()


def test_parse_noaa_ars_registers_both_4_and_5_digit_forms():
    assert parse_noaa_ars("12673") == {"12673", "2673"}
    assert parse_noaa_ars("2673") == {"2673"}  # already short -- no further stripping


def test_parse_noaa_ars_filters_missing_sentinel():
    # A real, confirmed-live JSOC value for a HARP with no assigned NOAA
    # AR -- the literal string "MISSING", not NaN or "0".
    assert parse_noaa_ars("MISSING") == set()


def _sharp_df():
    return pd.DataFrame([
        {"HARPNUM": 7115, "T_REC": pd.Timestamp("2017-09-04T06:00:00Z"), "NOAA_ARS": "12673"},
        {"HARPNUM": 7115, "T_REC": pd.Timestamp("2017-09-06T06:00:00Z"), "NOAA_ARS": "12673"},
        {"HARPNUM": 9001, "T_REC": pd.Timestamp("2017-09-04T06:00:00Z"), "NOAA_ARS": "12680,12681"},
        {"HARPNUM": 9002, "T_REC": pd.Timestamp("2017-09-04T06:00:00Z"), "NOAA_ARS": float("nan")},
    ])


def test_build_harp_ar_index_explodes_multi_ar_harps():
    index = build_harp_ar_index(_sharp_df())
    ars_for_9001 = set(index.loc[index["HARPNUM"] == 9001, "active_region"])
    assert {"12680", "2680", "12681", "2681"} <= ars_for_9001
    assert 9002 not in set(index["HARPNUM"])  # NaN NOAA_ARS contributes nothing


def test_build_harp_ar_index_raises_on_missing_column():
    import pytest

    with pytest.raises(KeyError):
        build_harp_ar_index(pd.DataFrame({"HARPNUM": [1]}))


def _flare_catalogue_df():
    return pd.DataFrame([
        {"flare_id": "f1", "time_utc_peak": pd.Timestamp("2017-09-04T05:49:00Z"), "active_region": "2673"},
        {"flare_id": "f2", "time_utc_peak": pd.Timestamp("2017-09-06T12:02:00Z"), "active_region": ""},
        {"flare_id": "f3", "time_utc_peak": pd.Timestamp("2017-09-04T06:00:00Z"), "active_region": "99999"},
    ])


def test_attribute_flares_to_harps_matches_within_tolerance():
    index = build_harp_ar_index(_sharp_df())
    out = attribute_flares_to_harps(_flare_catalogue_df(), index, time_tolerance_hours=12.0)

    f1 = out.loc[out["flare_id"] == "f1"].iloc[0]
    assert f1["harpnums"] == [7115]  # AR 2673 real short form matches HARP 7115's exploded "2673"

    f2 = out.loc[out["flare_id"] == "f2"].iloc[0]
    assert f2["harpnums"] == []  # unassigned flare (no active_region) -- never attributed

    f3 = out.loc[out["flare_id"] == "f3"].iloc[0]
    assert f3["harpnums"] == []  # AR with no matching HARP anywhere


def test_attribute_flares_to_harps_respects_time_tolerance():
    index = build_harp_ar_index(_sharp_df())
    far_flare = pd.DataFrame([
        {"flare_id": "far1", "time_utc_peak": pd.Timestamp("2017-09-10T06:00:00Z"), "active_region": "2673"}
    ])
    out = attribute_flares_to_harps(far_flare, index, time_tolerance_hours=1.0)
    assert out.iloc[0]["harpnums"] == []  # nearest HARP 7115 record is days away, outside a 1h tolerance


def test_attribution_quality_report_hand_computed():
    index = build_harp_ar_index(_sharp_df())
    attributed = attribute_flares_to_harps(_flare_catalogue_df(), index, time_tolerance_hours=12.0)
    report = attribution_quality_report(attributed)
    assert report["n_flares_total"] == 3
    assert report["n_flares_with_active_region"] == 2  # f1, f3 (f2 unassigned)
    assert report["n_flares_attributed_to_at_least_one_harp"] == 1  # only f1
    assert report["pct_flares_with_active_region"] == (2 / 3 * 100.0)
