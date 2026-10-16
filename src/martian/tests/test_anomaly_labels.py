"""Tests for z-scores, dust exceedance sidedness, the storm-episode
persistence+area gate, horizon-target construction, and runtime config
loading. Dust-only -- no real climatological source exists for
surface_temperature_k."""
import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from martian.preprocessing.grid import GridDefinition, assign_cells, load_primary_grid
from martian.labeling.anomaly_labels import (
    compute_exceedance,
    compute_horizon_labels,
    compute_storm_episodes,
    compute_z_scores,
    load_anomaly_definition_config,
    run_anomaly_sensitivity,
    run_full_labeling_pipeline,
)


class _FixedMCDProvider:
    """Deterministic mu/sigma stub -- constant regardless of input -- so
    z-scores are exactly predictable by hand."""

    def __init__(self, mu: dict, sigma: dict):
        self._mu = mu
        self._sigma = sigma

    def query(self, *, latitude, longitude, solar_longitude_ls, local_true_solar_time, variable, pressure_level_pa=None):
        n = len(np.atleast_1d(latitude))
        return pd.DataFrame({"mu": [self._mu[variable]] * n, "sigma": [self._sigma[variable]] * n})


def _row(cell_id, mars_year, sol, dust, lat=10.0, lon=100.0, ls=45.0, ltst=14.0) -> dict:
    return {
        "cell_id": cell_id,
        "mars_year": mars_year,
        "sol": sol,
        "latitude": lat,
        "longitude": lon,
        "solar_longitude_ls": ls,
        "local_true_solar_time": ltst,
        "dust_optical_depth": dust,
    }


@pytest.fixture
def grid() -> GridDefinition:
    return load_primary_grid()


@pytest.fixture
def config():
    return load_anomaly_definition_config()


@pytest.fixture
def fixed_mcd():
    return _FixedMCDProvider(mu={"dust_optical_depth": 0.2}, sigma={"dust_optical_depth": 0.05})


def test_z_score_formula_matches_hand_computation(fixed_mcd, config):
    df = pd.DataFrame([_row(0, 1, 0, dust=0.4)])
    z = compute_z_scores(df, fixed_mcd, epsilon=config["epsilon"])
    # (0.4 - 0.2) / 0.05 = 4.0
    assert z.loc[0, "dust_optical_depth_z"] == pytest.approx(4.0)


def test_dust_exceedance_is_one_sided_high_only(fixed_mcd, config):
    # z_dust = (0.0 - 0.2)/0.05 = -4.0 -- large NEGATIVE deviation must NOT
    # be flagged (one-sided high only).
    df = pd.DataFrame([_row(0, 1, 0, dust=0.0), _row(0, 1, 1, dust=0.5)])
    z = compute_z_scores(df, fixed_mcd, epsilon=config["epsilon"])
    exc = compute_exceedance(z, tau=2.0, config=config)
    assert exc.loc[0, "dust_optical_depth_z"] < -2.0
    assert not exc.loc[0, "dust_anomaly"]
    assert exc.loc[1, "dust_optical_depth_z"] > 2.0
    assert exc.loc[1, "dust_anomaly"]


def test_atmospheric_anomaly_equals_dust_anomaly(fixed_mcd, config):
    df = pd.DataFrame(
        [
            _row(0, 1, 0, dust=0.5),   # exceeds
            _row(0, 1, 1, dust=0.2),   # does not
        ]
    )
    z = compute_z_scores(df, fixed_mcd, epsilon=config["epsilon"])
    exc = compute_exceedance(z, tau=2.0, config=config)
    assert exc["atmospheric_anomaly"].tolist() == exc["dust_anomaly"].tolist() == [True, False]


def test_storm_episode_persistence_gate(grid, fixed_mcd, config):
    # Single-sol exceedance (duration_sols=1) at a single cell.
    df = pd.DataFrame([_row(0, 1, 5, dust=0.5)])
    df = assign_cells(df, grid)
    z = compute_z_scores(df, fixed_mcd, epsilon=config["epsilon"])
    exc = compute_exceedance(z, tau=2.0, config=config)

    out_p1, _ = compute_storm_episodes(exc, grid, persistence_sols=1, min_area_cells=1)
    assert out_p1["storm_episode"].all()

    out_p2, cat_p2 = compute_storm_episodes(exc, grid, persistence_sols=2, min_area_cells=1)
    assert not out_p2["storm_episode"].any()
    assert not cat_p2["qualifies"].any()


def test_storm_episode_area_gate(grid, fixed_mcd, config):
    # Two DIFFERENT, spatially far-apart cells anomalous on the same sol --
    # each forms its own single-cell episode (peak_area_cells=1).
    df = pd.DataFrame(
        [
            _row(0, 1, 5, dust=0.5, lat=-80.0, lon=10.0),
            _row(1, 1, 5, dust=0.5, lat=80.0, lon=200.0),
        ]
    )
    df = assign_cells(df, grid)
    z = compute_z_scores(df, fixed_mcd, epsilon=config["epsilon"])
    exc = compute_exceedance(z, tau=2.0, config=config)

    out_a1, _ = compute_storm_episodes(exc, grid, persistence_sols=1, min_area_cells=1)
    assert out_a1["storm_episode"].all()

    out_a3, cat_a3 = compute_storm_episodes(exc, grid, persistence_sols=1, min_area_cells=3)
    assert not out_a3["storm_episode"].any()
    assert not cat_a3["qualifies"].any()


def test_horizon_label_true_within_window_false_beyond(grid, fixed_mcd, config):
    # Storm only at sol=6. From sol=0 (5 sols before), horizon_5sol should
    # see it (0 < 6 <= 0+5? no, 6 > 5, so NOT within horizon 5 either) --
    # use sol=2 as the anchor: storm at sol 6 is 4 sols ahead.
    rows = [_row(0, 1, s, dust=0.2) for s in range(10) if s != 6]
    rows.append(_row(0, 1, 6, dust=0.5))
    df = pd.DataFrame(rows)
    df = assign_cells(df, grid)
    z = compute_z_scores(df, fixed_mcd, epsilon=config["epsilon"])
    exc = compute_exceedance(z, tau=2.0, config=config)
    out, _ = compute_storm_episodes(exc, grid, persistence_sols=1, min_area_cells=1)
    out = compute_horizon_labels(out, horizons_sols=(1, 3, 5))

    by_sol = out.set_index("sol")
    # From sol=2: storm at sol=6 is 4 sols ahead -> within horizon 5, not 1 or 3.
    assert not by_sol.loc[2, "horizon_1sol"]
    assert not by_sol.loc[2, "horizon_3sol"]
    assert by_sol.loc[2, "horizon_5sol"]
    # From sol=5: storm at sol=6 is 1 sol ahead -> within horizon 1, 3, and 5.
    assert by_sol.loc[5, "horizon_1sol"]
    assert by_sol.loc[5, "horizon_3sol"]
    assert by_sol.loc[5, "horizon_5sol"]
    # From sol=7 (AFTER the storm): no future storm exists -> all horizons False.
    assert not by_sol.loc[7, "horizon_1sol"]
    assert not by_sol.loc[7, "horizon_3sol"]
    assert not by_sol.loc[7, "horizon_5sol"]


def test_horizon_label_never_looks_at_the_past(grid, fixed_mcd, config):
    # A storm at sol=1 (in the PAST relative to sol=5) must never make
    # horizon_5sol True at sol=5.
    rows = [_row(0, 1, s, dust=0.2) for s in range(10) if s != 1]
    rows.append(_row(0, 1, 1, dust=0.5))
    df = pd.DataFrame(rows)
    df = assign_cells(df, grid)
    z = compute_z_scores(df, fixed_mcd, epsilon=config["epsilon"])
    exc = compute_exceedance(z, tau=2.0, config=config)
    out, _ = compute_storm_episodes(exc, grid, persistence_sols=1, min_area_cells=1)
    out = compute_horizon_labels(out, horizons_sols=(1, 3, 5))
    by_sol = out.set_index("sol")
    assert not by_sol.loc[5, "horizon_5sol"]


def test_exclude_interpolated_rows_never_labeled(grid, fixed_mcd, config):
    df = pd.DataFrame(
        [
            _row(0, 1, 0, dust=0.5),
            _row(0, 1, 1, dust=0.5),
        ]
    )
    df = assign_cells(df, grid)
    df["interpolation_flag"] = [False, True]
    labeled_df, _ = run_full_labeling_pipeline(df, fixed_mcd, grid, config=config)
    assert set(labeled_df["sol"]) == {0}


def test_config_thresholds_loaded_at_runtime(tmp_path, fixed_mcd, config, grid):
    custom_config = copy.deepcopy(config)
    custom_config["tau_primary"] = 100.0  # impossibly high -- nothing should exceed
    custom_path = tmp_path / "custom_anomaly_definition.yaml"
    custom_path.write_text(yaml.safe_dump(custom_config), encoding="utf-8")
    loaded = load_anomaly_definition_config(custom_path)
    assert loaded["tau_primary"] == 100.0

    df = pd.DataFrame([_row(0, 1, 0, dust=0.5)])
    df = assign_cells(df, grid)
    labeled_default, _ = run_full_labeling_pipeline(df, fixed_mcd, grid, config=config)
    labeled_custom, _ = run_full_labeling_pipeline(df, fixed_mcd, grid, config=loaded)
    assert labeled_default["dust_anomaly"].any()
    assert not labeled_custom["dust_anomaly"].any()


def test_run_anomaly_sensitivity_covers_configured_values(grid, fixed_mcd, config):
    rows = [_row(0, 1, s, dust=0.5 if s == 5 else 0.2) for s in range(10)]
    df = pd.DataFrame(rows)
    df = assign_cells(df, grid)
    tables = run_anomaly_sensitivity(df, fixed_mcd, grid, config=config)
    assert set(tables.keys()) == {"tau_sensitivity", "persistence_sensitivity", "min_area_sensitivity"}
    assert tables["tau_sensitivity"]["tau"].tolist() == config["tau_sensitivity"]
    assert tables["persistence_sensitivity"]["persistence_sols"].tolist() == config["sensitivity"]["persistence_sols"]
    assert tables["min_area_sensitivity"]["min_area_cells"].tolist() == config["sensitivity"]["min_area_cells"]
