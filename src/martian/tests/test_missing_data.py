"""Tests for causal (trailing-only) missing-data handling: chronological
gap-filling, max-gap enforcement, gap-length/indicator retention, and the
never-interpolate-labels guard."""
import copy

import numpy as np
import pandas as pd
import pytest

from martian.preprocessing.missing_data import (
    apply_trailing_fill,
    load_missing_data_config,
    run_missing_data_sensitivity,
)


def _series_df(sols, dust_values, *, cell_id=0, mars_year=1, extra_cols=None) -> pd.DataFrame:
    n = len(sols)
    row = {
        "cell_id": [cell_id] * n,
        "mars_year": [mars_year] * n,
        "sol": sols,
        "dust_optical_depth": dust_values,
        "dust_optical_depth_uncertainty": [0.1] * n,
        "water_ice_optical_depth": [0.1] * n,
        "water_ice_optical_depth_uncertainty": [0.01] * n,
        "surface_temperature_k": [220.0] * n,
        "surface_temperature_k_uncertainty": [2.0] * n,
        "surface_pressure_mbar": [6.1] * n,
        "surface_emissivity": [0.9] * n,
        "emission_angle_deg": [10.0] * n,
        "solar_longitude_ls": [45.0] * n,
        "local_true_solar_time": [14.0] * n,
        "pressure_level_coverage_frac": [1.0] * n,
        "uncertainty_summary": [0.05] * n,
        "observation_count": [3] * n,
        "data_source": ["SIMULATED-AGGREGATED-CELL-SOL"] * n,
    }
    if extra_cols:
        row.update(extra_cols)
    return pd.DataFrame(row)


@pytest.fixture
def config():
    return load_missing_data_config()


def test_gap_within_threshold_is_filled_forward(config):
    df = _series_df([0, 1, 4, 5], [1.0, 2.0, 5.0, 6.0])
    out = apply_trailing_fill(df, max_gap_sols=2, config=config)
    filled = out[out["sol"].isin([2, 3])]
    assert (filled["dust_optical_depth"] == 2.0).all()
    assert filled["interpolation_flag"].all()
    assert filled["gap_length_sols"].tolist() == [1.0, 2.0]


def test_gap_exceeding_threshold_is_dropped_not_fabricated(config):
    # Gap from sol 1 to sol 6 spans sols 2,3,4,5 (gap_length 1,2,3,4 resp.).
    # Only the leading portion within the 2-sol threshold (sols 2,3) is
    # fillable; sols 4,5 exceed it and must be ABSENT -- never present with a
    # fabricated or NaN placeholder value.
    df = _series_df([0, 1, 6, 7], [1.0, 2.0, 8.0, 9.0])
    out = apply_trailing_fill(df, max_gap_sols=2, config=config)
    assert set(out["sol"]) == {0, 1, 2, 3, 6, 7}
    assert 4 not in set(out["sol"])
    assert 5 not in set(out["sol"])


def test_gap_before_first_observation_never_filled(config):
    df = _series_df([5, 6], [5.0, 6.0])
    out = apply_trailing_fill(df, max_gap_sols=2, config=config)
    # sols 3,4 precede the first real observation -- no prior value exists to
    # carry forward, so they must never appear regardless of max_gap_sols.
    assert set(out["sol"]) == {5, 6}


def test_max_gap_zero_means_no_fill_at_all(config):
    df = _series_df([0, 1, 3, 4], [1.0, 2.0, 3.0, 4.0])
    out = apply_trailing_fill(df, max_gap_sols=0, config=config)
    assert len(out) == 4
    assert out["is_observed"].all()
    assert not out["interpolation_flag"].any()


def test_sensitivity_sweep_covers_configured_thresholds(config):
    df = _series_df([0, 1, 4, 5, 9, 10], [1.0, 2.0, 5.0, 6.0, 10.0, 11.0])
    table = run_missing_data_sensitivity(df, config=config)
    assert table["max_trailing_fill_gap_sols"].tolist() == [0, 1, 2, 3]
    by_gap = table.set_index("max_trailing_fill_gap_sols")
    # stricter (smaller) max-gap thresholds can only fill fewer or equal rows
    assert by_gap.loc[0, "n_filled"] == 0
    assert by_gap.loc[0, "n_filled"] <= by_gap.loc[1, "n_filled"]
    assert by_gap.loc[1, "n_filled"] <= by_gap.loc[2, "n_filled"]
    assert by_gap.loc[2, "n_filled"] <= by_gap.loc[3, "n_filled"]


def test_reserved_label_pattern_in_fill_variables_raises(tmp_path):
    config = load_missing_data_config()
    bad_config = copy.deepcopy(config)
    bad_config["fill_variables"].append("label")
    bad_path = tmp_path / "bad_missing_data.yaml"
    import yaml

    bad_path.write_text(yaml.safe_dump(bad_config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_missing_data_config(bad_path)


def test_label_column_is_never_interpolated(config):
    df = _series_df(
        [0, 1, 4, 5],
        [1.0, 2.0, 5.0, 6.0],
        extra_cols={"label": [0, 1, 0, 1]},
    )
    out = apply_trailing_fill(df, max_gap_sols=2, config=config)
    filled_rows = out[out["interpolation_flag"]]
    assert filled_rows["label"].isna().all()
    observed_rows = out[out["is_observed"]]
    assert observed_rows["label"].tolist() == [0, 1, 0, 1]


def test_chronological_processing_two_cells_independent(config):
    df_a = _series_df([0, 1, 4, 5], [1.0, 2.0, 5.0, 6.0], cell_id=0)
    df_b = _series_df([0, 3], [100.0, 103.0], cell_id=1)
    df = pd.concat([df_a, df_b], ignore_index=True)
    out = apply_trailing_fill(df, max_gap_sols=2, config=config)
    cell0 = out[out["cell_id"] == 0]
    cell1 = out[out["cell_id"] == 1]
    # cell 1's values must never leak into cell 0's filled rows or vice versa
    assert set(cell0["dust_optical_depth"].round(1)) <= {1.0, 2.0, 5.0, 6.0}
    assert set(cell1["dust_optical_depth"].round(1)) <= {100.0, 103.0}


def test_low_support_flag_recomputed_true_for_filled_rows(config):
    df = _series_df([0, 1, 4, 5], [1.0, 2.0, 5.0, 6.0])
    out = apply_trailing_fill(df, max_gap_sols=2, config=config)
    filled = out[out["interpolation_flag"]]
    assert filled["low_support_flag"].all()  # observation_count forced to 0
