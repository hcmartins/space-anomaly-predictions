"""Tests for the 8-rule first-failing-rule QC pipeline: one deliberately-dirty
row per rule, priority-order attribution, disabled-rule behaviour, and
malformed-config validation."""
import copy

import pandas as pd
import pytest

from martian.preprocessing.quality_control import (
    QC_STATUS_ACCEPTED,
    QC_STATUS_REJECTED,
    load_qc_config,
    run_quality_control,
)


def _clean_row(**overrides) -> dict:
    row = {
        "latitude": 10.0,
        "longitude": 100.0,
        "utc_timestamp": pd.Timestamp("2021-02-09", tz="UTC"),
        "mars_year": 1,
        "sol": 5,
        "solar_longitude_ls": 45.0,
        "local_true_solar_time": 14.0,
        "dust_optical_depth": 0.2,
        "dust_optical_depth_uncertainty": 0.02,
        "water_ice_optical_depth": 0.1,
        "water_ice_optical_depth_uncertainty": 0.01,
        "surface_temperature_k": 220.0,
        "surface_temperature_k_uncertainty": 5.0,
        "emission_angle_deg": 10.0,
        "quality_flag": 0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def qc_config():
    return load_qc_config()


def test_all_clean_fixture_fully_accepted(qc_config):
    df = pd.DataFrame([_clean_row(), _clean_row(longitude=101.0), _clean_row(longitude=102.0)])
    report = run_quality_control(df, qc_config)
    assert report.summary()["n_accepted"] == 3
    assert report.summary()["n_rejected"] == 0


def test_rule1_missing_required_variable(qc_config):
    df = pd.DataFrame([_clean_row(), _clean_row(surface_temperature_k=None)])
    report = run_quality_control(df, qc_config)
    assert report.df.loc[1, "qc_status"] == QC_STATUS_REJECTED
    assert report.df.loc[1, "qc_rejection_rule"] == "missing_required_variable"


def test_rule2_invalid_quality_flag(qc_config):
    df = pd.DataFrame([_clean_row(), _clean_row(quality_flag=1)])
    report = run_quality_control(df, qc_config)
    assert report.df.loc[1, "qc_rejection_rule"] == "invalid_quality_flag"


def test_rule3_impossible_coordinate(qc_config):
    df = pd.DataFrame([_clean_row(), _clean_row(latitude=200.0)])
    report = run_quality_control(df, qc_config)
    assert report.df.loc[1, "qc_rejection_rule"] == "impossible_coordinate_or_timestamp"


def test_rule4_negative_optical_depth(qc_config):
    df = pd.DataFrame([_clean_row(), _clean_row(dust_optical_depth=-0.1)])
    report = run_quality_control(df, qc_config)
    assert report.df.loc[1, "qc_rejection_rule"] == "negative_optical_depth"


def test_rule5_physically_impossible_temperature(qc_config):
    df = pd.DataFrame([_clean_row(), _clean_row(surface_temperature_k=999.0)])
    report = run_quality_control(df, qc_config)
    assert report.df.loc[1, "qc_rejection_rule"] == "physically_impossible_temperature"


def test_rule6_excessive_retrieval_uncertainty(qc_config):
    df = pd.DataFrame([_clean_row(), _clean_row(dust_optical_depth_uncertainty=5.0)])
    report = run_quality_control(df, qc_config)
    assert report.df.loc[1, "qc_rejection_rule"] == "excessive_retrieval_uncertainty"


def test_rule7_unacceptable_viewing_geometry(qc_config):
    # Rule 7 is disabled by default for real data (no viewing-geometry field
    # in the real EMIRS schema -- see configs/quality_control.yaml) -- force
    # it enabled here to test the rule's own mechanics independent of that.
    config = copy.deepcopy(qc_config)
    for rule in config["rules"]:
        if rule["name"] == "unacceptable_viewing_geometry":
            rule["enabled"] = True
    df = pd.DataFrame([_clean_row(), _clean_row(emission_angle_deg=85.0)])
    report = run_quality_control(df, config)
    assert report.df.loc[1, "qc_rejection_rule"] == "unacceptable_viewing_geometry"


def test_rule7_inert_when_emission_angle_column_absent(qc_config):
    """Real data has no emission_angle_deg field at all -- the check must not
    crash, and must not flag anything (not evaluable != a failure)."""
    df = pd.DataFrame([_clean_row()]).drop(columns=["emission_angle_deg"])
    report = run_quality_control(df, qc_config)
    assert report.df.loc[0, "qc_status"] == QC_STATUS_ACCEPTED
    rule_result = next(r for r in report.rule_results if r.name == "unacceptable_viewing_geometry")
    assert rule_result.n_flagged == 0


def test_rule8_duplicate_retrieval(qc_config):
    df = pd.DataFrame([_clean_row(), _clean_row()])
    report = run_quality_control(df, qc_config)
    assert report.df.loc[0, "qc_status"] == QC_STATUS_ACCEPTED
    assert report.df.loc[1, "qc_rejection_rule"] == "duplicate_retrieval"


def test_priority_order_attributes_first_failing_rule(qc_config):
    # Fails BOTH rule 1 (missing surface_temperature_k) and rule 4 (negative
    # dust_optical_depth) -- must be attributed to rule 1, the lower id.
    # A clean row is included alongside so surface_temperature_k stays a
    # float64 column (with NaN) rather than collapsing to object dtype, which
    # is how the field actually arrives in real multi-row batches.
    df = pd.DataFrame([_clean_row(), _clean_row(surface_temperature_k=None, dust_optical_depth=-0.1)])
    report = run_quality_control(df, qc_config)
    assert report.df.loc[1, "qc_rejection_rule"] == "missing_required_variable"


def test_disabled_rule_never_rejects(qc_config):
    config = copy.deepcopy(qc_config)
    for rule in config["rules"]:
        if rule["name"] == "unacceptable_viewing_geometry":
            rule["enabled"] = False
    df = pd.DataFrame([_clean_row(emission_angle_deg=85.0)])
    report = run_quality_control(df, config)
    assert report.df.loc[0, "qc_status"] == QC_STATUS_ACCEPTED
    rule_result = next(r for r in report.rule_results if r.name == "unacceptable_viewing_geometry")
    assert rule_result.n_flagged == 1
    assert rule_result.n_rejected_here == 0


def test_malformed_config_rule_ids_must_be_contiguous(tmp_path):
    config_path = tmp_path / "bad_qc.yaml"
    config_path.write_text(
        "rules:\n"
        "  - id: 1\n"
        "    name: missing_required_variable\n"
        "    enabled: true\n"
        "    source: VERIFIED_PRODUCT_DOCUMENTATION\n"
        "    citation: test\n"
        "    parameters: {required_variables: []}\n"
        "  - id: 3\n"
        "    name: invalid_quality_flag\n"
        "    enabled: true\n"
        "    source: VERIFIED_PRODUCT_DOCUMENTATION\n"
        "    citation: test\n"
        "    parameters: {pass_value: 0}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_qc_config(config_path)


def test_accepted_property_drops_qc_internal_columns(qc_config):
    df = pd.DataFrame([_clean_row()])
    report = run_quality_control(df, qc_config)
    assert "qc_status" not in report.accepted.columns
    assert "qc_rejection_rule" not in report.accepted.columns
