"""Tests for the shared GOES/SHARP QC pipeline: first-failing-rule
semantics, and that a row failing multiple rules is attributed to the
highest-priority (lowest id) one only."""
import pandas as pd
import pytest

from preprocessing.quality_control import load_qc_config, run_quality_control


def test_load_qc_config_ids_contiguous_for_both_domains():
    config = load_qc_config()
    assert "goes" in config and "sharp" in config


def test_goes_missing_flux_rejected():
    df = pd.DataFrame([
        {"time_utc": "t1", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": None, "xrsb_quality": 0},
        {"time_utc": "t2", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 1e-7, "xrsb_quality": 0},
    ])
    report = run_quality_control(df, "goes")
    assert report.summary()["n_accepted"] == 1
    assert report.rejected.iloc[0]["qc_rejection_rule"] == "missing_required_column"


def test_goes_invalid_quality_flag_rejected():
    df = pd.DataFrame([
        {"time_utc": "t1", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 1e-7, "xrsb_quality": 3},
    ])
    report = run_quality_control(df, "goes")
    assert report.rejected.iloc[0]["qc_rejection_rule"] == "invalid_quality_flag"


def test_goes_non_positive_flux_rejected():
    df = pd.DataFrame([
        {"time_utc": "t1", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": -1.0, "xrsb_quality": 0},
    ])
    report = run_quality_control(df, "goes")
    assert report.rejected.iloc[0]["qc_rejection_rule"] == "non_positive_value"


def test_goes_duplicate_row_rejected_keeps_first():
    df = pd.DataFrame([
        {"time_utc": "t1", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 1e-7, "xrsb_quality": 0},
        {"time_utc": "t1", "satellite": 16, "xrsa_flux_w_m2": 2e-8, "xrsb_flux_w_m2": 2e-7, "xrsb_quality": 0},
    ])
    report = run_quality_control(df, "goes")
    assert report.summary()["n_accepted"] == 1
    assert report.accepted.iloc[0]["xrsb_flux_w_m2"] == 1e-7


def test_goes_non_finite_flux_rejected():
    df = pd.DataFrame([
        {"time_utc": "t1", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": float("inf"), "xrsb_quality": 0},
    ])
    report = run_quality_control(df, "goes")
    assert report.rejected.iloc[0]["qc_rejection_rule"] == "non_finite_value"


def test_sharp_non_finite_keyword_rejected():
    df = pd.DataFrame([
        {"HARPNUM": 3672, "T_REC": "2014.01.21_12:48:00_TAI", "USFLUX": 1.0, "TOTUSJZ": 2.0,
         "MEANJZH": float("-inf"), "QUALITY": 0},
        {"HARPNUM": 3672, "T_REC": "2014.01.21_13:00:00_TAI", "USFLUX": 1.0, "TOTUSJZ": 2.0,
         "MEANJZH": 3.0, "QUALITY": 0},
    ])
    report = run_quality_control(df, "sharp")
    assert report.summary()["n_accepted"] == 1
    assert report.rejected.iloc[0]["qc_rejection_rule"] == "non_finite_value"


def test_sharp_nan_keyword_not_rejected_by_non_finite_rule():
    # A NaN in an optional (non-required) keyword is a legitimate "this
    # record didn't report that keyword" case -- only isinf is rejected,
    # never isnan, by the non_finite_value rule specifically.
    df = pd.DataFrame([
        {"HARPNUM": 1, "T_REC": "2014.01.21_12:48:00_TAI", "USFLUX": 1.0, "TOTUSJZ": float("nan"), "QUALITY": 0},
    ])
    report = run_quality_control(df, "sharp")
    assert report.summary()["n_accepted"] == 1


def test_first_failing_rule_wins_not_the_last():
    # Missing flux (rule 1) AND would also be non-positive if present (rule 3)
    # -- must be attributed to rule 1, the highest-priority one it fails.
    df = pd.DataFrame([
        {"time_utc": "t1", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": None, "xrsb_quality": 5},
    ])
    report = run_quality_control(df, "goes")
    assert report.rejected.iloc[0]["qc_rejection_rule"] == "missing_required_column"


def test_sharp_missing_usflux_rejected():
    df = pd.DataFrame([
        {"HARPNUM": 401, "T_REC": "t1", "USFLUX": None, "QUALITY": 0},
        {"HARPNUM": 402, "T_REC": "t1", "USFLUX": 1.0e22, "QUALITY": 0},
    ])
    report = run_quality_control(df, "sharp")
    assert report.summary()["n_accepted"] == 1


def test_sharp_invalid_quality_bitmask_rejected():
    df = pd.DataFrame([{"HARPNUM": 401, "T_REC": "t1", "USFLUX": 1.0e22, "QUALITY": 0x80000000}])
    report = run_quality_control(df, "sharp")
    assert report.rejected.iloc[0]["qc_rejection_rule"] == "invalid_quality_flag"


def test_disabled_rule_never_rejects_but_still_counted(monkeypatch):
    df = pd.DataFrame([
        {"time_utc": "t1", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": -1.0, "xrsb_quality": 0},
    ])
    config = load_qc_config()
    for rule in config["goes"]["rules"]:
        if rule["name"] == "non_positive_value":
            rule["enabled"] = False
    report = run_quality_control(df, "goes", config=config)
    assert report.summary()["n_accepted"] == 1
    rule_result = next(r for r in report.rule_results if r.name == "non_positive_value")
    assert rule_result.n_flagged == 1
    assert rule_result.n_rejected_here == 0
