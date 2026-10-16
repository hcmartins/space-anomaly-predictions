"""Tests for grid-resolution, minimum-observation, and pressure-coverage
sensitivity analyses."""
import pandas as pd
import pytest

from martian.preprocessing.aggregation import load_aggregation_config
from martian.preprocessing.sensitivity import (
    grid_resolution_sensitivity,
    min_obs_sensitivity,
    pressure_coverage_sensitivity,
)


def _accepted_row(**overrides) -> dict:
    row = {
        "latitude": 10.0,
        "longitude": 100.0,
        "mars_year": 1,
        "sol": 1,
        "dust_optical_depth": 0.2,
        "dust_optical_depth_uncertainty": 0.02,
        "water_ice_optical_depth": 0.1,
        "water_ice_optical_depth_uncertainty": 0.01,
        "surface_temperature_k": 220.0,
        "surface_temperature_k_uncertainty": 5.0,
        "surface_pressure_mbar": 6.1,
        "surface_emissivity": 0.9,
        "solar_longitude_ls": 45.0,
        "local_true_solar_time": 14.0,
        "emission_angle_deg": 10.0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def agg_config():
    return load_aggregation_config()


def test_grid_resolution_sensitivity_covers_all_configured_resolutions(agg_config):
    df = pd.DataFrame(
        [_accepted_row(latitude=lat, longitude=lon) for lat, lon in [(10, 10), (12, 12), (-40, 200)]]
    )
    table = grid_resolution_sensitivity(
        df, min_obs_per_cell_sol=1, min_pressure_level_coverage_frac=0.5, aggregation_config=agg_config
    )
    assert sorted(table["resolution_deg"].tolist()) == [2.5, 5.0, 10.0]
    assert table["is_primary"].sum() == 1
    assert table.loc[table["is_primary"], "resolution_deg"].iloc[0] == 5.0


def test_grid_resolution_sensitivity_nominal_cell_counts(agg_config):
    df = pd.DataFrame([_accepted_row()])
    table = grid_resolution_sensitivity(
        df, min_obs_per_cell_sol=1, min_pressure_level_coverage_frac=0.5, aggregation_config=agg_config
    )
    by_res = table.set_index("resolution_deg")
    assert by_res.loc[5.0, "n_cells_nominal"] == 2592
    assert by_res.loc[2.5, "n_cells_nominal"] == 72 * 144
    assert by_res.loc[10.0, "n_cells_nominal"] == 18 * 36


def test_min_obs_sensitivity_retained_fraction():
    aggregated = pd.DataFrame(
        {
            "observation_count": [1, 3, 5],
            "pressure_level_coverage_frac": [1.0, 1.0, 1.0],
        }
    )
    table = min_obs_sensitivity(
        aggregated, min_obs_values=[1, 3, 5], min_pressure_level_coverage_frac=0.5
    )
    by_threshold = table.set_index("min_obs_per_cell_sol")
    assert by_threshold.loc[1, "retained_fraction"] == pytest.approx(1.0)
    assert by_threshold.loc[3, "retained_fraction"] == pytest.approx(2 / 3)
    assert by_threshold.loc[5, "retained_fraction"] == pytest.approx(1 / 3)


def test_pressure_coverage_sensitivity_retained_fraction():
    aggregated = pd.DataFrame(
        {
            "observation_count": [10, 10, 10],
            "pressure_level_coverage_frac": [0.4, 0.6, 0.95],
        }
    )
    table = pressure_coverage_sensitivity(
        aggregated, coverage_values=[0.5, 0.7, 0.9], min_obs_per_cell_sol=1
    )
    by_threshold = table.set_index("min_pressure_level_coverage_frac")
    assert by_threshold.loc[0.5, "retained_fraction"] == pytest.approx(2 / 3)
    assert by_threshold.loc[0.7, "retained_fraction"] == pytest.approx(1 / 3)
    assert by_threshold.loc[0.9, "retained_fraction"] == pytest.approx(1 / 3)
