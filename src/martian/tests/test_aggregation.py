"""Tests for cell-sol aggregation: hand-computed inverse-variance weighting,
observation counts, low-support-flag recomputation, and error handling."""
import math

import numpy as np
import pandas as pd
import pytest

from martian.preprocessing.aggregation import (
    aggregate_by_cell_sol,
    compute_low_support_flag,
    load_aggregation_config,
)


def _base_row(**overrides) -> dict:
    row = {
        "cell_id": 0,
        "mars_year": 1,
        "sol": 1,
        "dust_optical_depth": 10.0,
        "dust_optical_depth_uncertainty": 1.0,
        "water_ice_optical_depth": 0.1,
        "water_ice_optical_depth_uncertainty": 0.01,
        "surface_temperature_k": 220.0,
        "surface_temperature_k_uncertainty": 2.0,
        "surface_pressure_mbar": 6.1,
        "surface_emissivity": 0.9,
        "latitude": 10.0,
        "longitude": 100.0,
        "solar_longitude_ls": 45.0,
        "local_true_solar_time": 14.0,
        "emission_angle_deg": 10.0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def agg_config():
    return load_aggregation_config()


def test_weighted_mean_matches_hand_computed_value(agg_config):
    df = pd.DataFrame(
        [
            _base_row(dust_optical_depth=10.0, dust_optical_depth_uncertainty=1.0),
            _base_row(dust_optical_depth=20.0, dust_optical_depth_uncertainty=2.0),
        ]
    )
    out = aggregate_by_cell_sol(
        df, min_obs_per_cell_sol=1, min_pressure_level_coverage_frac=0.5, config=agg_config
    )
    assert len(out) == 1
    # weights = 1/1^2, 1/2^2 = 1, 0.25 ; mean = (10*1 + 20*0.25) / 1.25 = 12.0
    assert out.loc[0, "dust_optical_depth"] == pytest.approx(12.0)
    expected_uncertainty = math.sqrt(1.0 / 1.25)
    assert out.loc[0, "dust_optical_depth_uncertainty"] == pytest.approx(expected_uncertainty)


def test_observation_count_correctness(agg_config):
    df = pd.DataFrame([_base_row(), _base_row(), _base_row()])
    out = aggregate_by_cell_sol(
        df, min_obs_per_cell_sol=1, min_pressure_level_coverage_frac=0.5, config=agg_config
    )
    assert out.loc[0, "observation_count"] == 3


def test_multiple_cell_sol_groups_kept_separate(agg_config):
    df = pd.DataFrame([_base_row(), _base_row(cell_id=1), _base_row(sol=2)])
    out = aggregate_by_cell_sol(
        df, min_obs_per_cell_sol=1, min_pressure_level_coverage_frac=0.5, config=agg_config
    )
    assert len(out) == 3


def test_interpolation_flag_always_false(agg_config):
    df = pd.DataFrame([_base_row(), _base_row()])
    out = aggregate_by_cell_sol(
        df, min_obs_per_cell_sol=1, min_pressure_level_coverage_frac=0.5, config=agg_config
    )
    assert (out["interpolation_flag"] == False).all()  # noqa: E712


def test_uncertainty_summary_non_nan_when_uncertainties_present(agg_config):
    df = pd.DataFrame([_base_row(), _base_row()])
    out = aggregate_by_cell_sol(
        df, min_obs_per_cell_sol=1, min_pressure_level_coverage_frac=0.5, config=agg_config
    )
    assert not np.isnan(out.loc[0, "uncertainty_summary"])


def test_missing_cell_id_raises_keyerror(agg_config):
    df = pd.DataFrame([_base_row()]).drop(columns=["cell_id"])
    with pytest.raises(KeyError):
        aggregate_by_cell_sol(df, min_obs_per_cell_sol=1, min_pressure_level_coverage_frac=0.5, config=agg_config)


def test_compute_low_support_flag_reacts_to_min_obs_threshold():
    df = pd.DataFrame(
        {
            "observation_count": [1, 3, 5],
            "pressure_level_coverage_frac": [1.0, 1.0, 1.0],
        }
    )
    flag_lenient = compute_low_support_flag(df, min_obs_per_cell_sol=1, min_pressure_level_coverage_frac=0.5)
    flag_strict = compute_low_support_flag(df, min_obs_per_cell_sol=5, min_pressure_level_coverage_frac=0.5)
    assert not flag_lenient.any()
    assert flag_strict.tolist() == [True, True, False]


def test_compute_low_support_flag_reacts_to_coverage_threshold():
    df = pd.DataFrame(
        {
            "observation_count": [10, 10, 10],
            "pressure_level_coverage_frac": [0.4, 0.6, 0.95],
        }
    )
    flag = compute_low_support_flag(df, min_obs_per_cell_sol=1, min_pressure_level_coverage_frac=0.5)
    assert flag.tolist() == [True, False, False]


def test_compute_low_support_flag_treats_nan_coverage_as_not_applicable():
    df = pd.DataFrame({"observation_count": [10], "pressure_level_coverage_frac": [np.nan]})
    flag = compute_low_support_flag(df, min_obs_per_cell_sol=1, min_pressure_level_coverage_frac=0.5)
    assert flag.tolist() == [False]
