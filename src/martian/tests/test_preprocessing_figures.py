"""Tests for evaluation/preprocessing_figures.py's data-computation helpers
(Figures 4.5-4.8, CHAPTER_4_PARAMETER_VALUES.md open item 3). Plotting
functions themselves are exercised only via the smoke tests at the bottom
(figure objects are produced without error); the numerical logic behind
each panel is tested directly."""
import numpy as np
import pandas as pd
import pytest

from martian.preprocessing.grid import load_primary_grid
from martian.evaluation.preprocessing_figures import (
    build_sol_to_ls_lookup,
    cell_centroid_lat_lon,
    compute_episode_centred_composite,
    compute_gapfill_state_matrix,
    compute_missingness_matrix,
    compute_pooled_lag_autocorrelation,
    compute_semivariogram,
    durbin_levinson_pacf,
    figure_4_5_raw_qc_comparison,
    figure_4_6_missingness_gapfill_distribution,
    figure_4_7_spatiotemporal_structure,
    figure_4_8_seasonal_series,
    haversine_distance_km,
    select_representative_cell,
    _ls_crossing_sols,
)


@pytest.fixture
def grid():
    return load_primary_grid()


# ---------------------------------------------------------------------------
# select_representative_cell
# ---------------------------------------------------------------------------

def test_select_representative_cell_picks_most_observed_train_cell():
    aggregated = pd.DataFrame(
        {"cell_id": [1, 1, 1, 2, 2, 3], "sol": [1, 2, 3, 1, 2, 1]}
    )
    split = pd.Series(["train", "train", "train", "train", "train", "test"])
    assert select_representative_cell(aggregated, split) == 1


def test_select_representative_cell_respects_exclusion():
    aggregated = pd.DataFrame(
        {"cell_id": [99, 99, 99, 99, 2, 2], "sol": [1, 2, 3, 4, 1, 2]}
    )
    split = pd.Series(["train"] * 6)
    assert select_representative_cell(aggregated, split, excluded_cell_ids=(99,)) == 2


def test_select_representative_cell_raises_if_all_excluded():
    aggregated = pd.DataFrame({"cell_id": [1, 1], "sol": [1, 2]})
    split = pd.Series(["train", "train"])
    with pytest.raises(ValueError):
        select_representative_cell(aggregated, split, excluded_cell_ids=(1,))


def test_select_representative_cell_ignores_non_train_rows():
    aggregated = pd.DataFrame({"cell_id": [1, 1, 1, 2], "sol": [1, 2, 3, 1]})
    split = pd.Series(["test", "test", "test", "train"])
    # cell 1 has more rows overall, but none in train -> cell 2 wins
    assert select_representative_cell(aggregated, split) == 2


# ---------------------------------------------------------------------------
# cell_centroid_lat_lon / haversine_distance_km
# ---------------------------------------------------------------------------

def test_cell_centroid_matches_grid_geometry(grid):
    # cell_id 0 = lat_band 0, lon_band 0 -> centre of the southernmost,
    # westernmost 5x5 cell: lat=-90+2.5=-87.5, lon=2.5
    lat, lon = cell_centroid_lat_lon(0, grid)
    assert lat == pytest.approx(-90.0 + grid.resolution_deg / 2.0)
    assert lon == pytest.approx(grid.resolution_deg / 2.0)


def test_haversine_distance_zero_for_same_point():
    assert haversine_distance_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0)


def test_haversine_distance_equator_quarter_circumference():
    # Two equatorial points 90 degrees of longitude apart span a quarter of
    # Mars's great-circle circumference.
    d = haversine_distance_km(0.0, 0.0, 0.0, 90.0)
    expected = 2 * np.pi * 3389.5 / 4.0
    assert d == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# compute_missingness_matrix / compute_gapfill_state_matrix
# ---------------------------------------------------------------------------

def test_missingness_matrix_marks_only_observed_cell_sols():
    aggregated = pd.DataFrame({"cell_id": [1, 1, 2], "sol": [10, 12, 11]})
    matrix = compute_missingness_matrix(aggregated, cell_ids=[1, 2], sol_min=10, sol_max=12)
    assert matrix.loc[1, 10] == 1
    assert matrix.loc[1, 11] == 0
    assert matrix.loc[1, 12] == 1
    assert matrix.loc[2, 11] == 1
    assert matrix.loc[2, 10] == 0


def test_gapfill_state_matrix_distinguishes_observed_filled_absent():
    filled = pd.DataFrame(
        {"cell_id": [1, 1, 1], "sol": [10, 11, 12], "is_observed": [True, False, True]}
    )
    matrix = compute_gapfill_state_matrix(filled, cell_ids=[1], sol_min=10, sol_max=13)
    assert matrix.loc[1, 10] == 2  # genuine
    assert matrix.loc[1, 11] == 1  # filled
    assert matrix.loc[1, 12] == 2  # genuine
    assert matrix.loc[1, 13] == 0  # absent (never in filled_df at all)


# ---------------------------------------------------------------------------
# compute_episode_centred_composite
# ---------------------------------------------------------------------------

def test_episode_composite_only_uses_genuine_observations_at_each_offset():
    labeled = pd.DataFrame(
        [
            {"cell_id": 1, "mars_year": 36, "sol": 9, "episode_id": 0, "dust_optical_depth_z": 1.0},
            {"cell_id": 1, "mars_year": 36, "sol": 10, "episode_id": 0, "dust_optical_depth_z": 3.0},
            # sol 11 (offset +1) missing entirely -> must not appear at offset +1
        ]
    )
    episodes = pd.DataFrame(
        [{"episode_id": 0, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "qualifies": True}]
    )
    result = compute_episode_centred_composite(labeled, episodes, window_before=1, window_after=1)
    offsets = result.set_index("offset_sols")
    assert offsets.loc[-1, "mean"] == pytest.approx(1.0)
    assert offsets.loc[0, "mean"] == pytest.approx(3.0)
    assert 1 not in offsets.index  # no genuine observation at sol 11


def test_episode_composite_ignores_non_qualifying_episodes():
    labeled = pd.DataFrame(
        [{"cell_id": 1, "mars_year": 36, "sol": 10, "episode_id": 0, "dust_optical_depth_z": 5.0}]
    )
    episodes = pd.DataFrame(
        [{"episode_id": 0, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "qualifies": False}]
    )
    result = compute_episode_centred_composite(labeled, episodes, window_before=1, window_after=1)
    assert result.empty


# ---------------------------------------------------------------------------
# compute_pooled_lag_autocorrelation / durbin_levinson_pacf
# ---------------------------------------------------------------------------

def test_pooled_acf_lag_zero_is_one():
    df = pd.DataFrame({"cell_id": [1, 1, 1], "mars_year": [36, 36, 36], "sol": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    acf = compute_pooled_lag_autocorrelation(df, value_col="v", max_lag=2)
    assert acf[0] == pytest.approx(1.0)


def test_pooled_acf_perfect_linear_trend_is_positive_and_declining():
    sols = np.arange(1, 21)
    df = pd.DataFrame({"cell_id": 1, "mars_year": 36, "sol": sols, "v": sols.astype(float)})
    acf = compute_pooled_lag_autocorrelation(df, value_col="v", max_lag=5)
    assert acf[1] > acf[2] > acf[3]
    assert acf[1] > 0


def test_pooled_acf_skips_non_adjacent_gap_pairs():
    # Cell A has sols 1,2 (lag-1 pair exists); cell B has sols 1,5 only (no
    # lag-1 pair -- must not be silently bridged).
    df = pd.DataFrame(
        {
            "cell_id": [1, 1, 2, 2],
            "mars_year": [36, 36, 36, 36],
            "sol": [1, 2, 1, 5],
            "v": [1.0, 2.0, 10.0, -10.0],
        }
    )
    acf = compute_pooled_lag_autocorrelation(df, value_col="v", max_lag=4)
    # lag=4 should be driven ONLY by the (1,5) pair from cell 2, not bridge
    # cell 1's sols.
    assert not np.isnan(acf[4])
    assert acf[4] < 0  # (10 - mean)(-10 - mean) with mean ~0.75 is negative


def test_pooled_acf_one_nan_value_does_not_poison_other_pairs_at_same_lag():
    # Real bug caught during Figure 4.7's own 2026-08-12 generation: with a
    # naive np.mean(products), a SINGLE NaN value_col entry anywhere in the
    # panel silently turned an entire lag's result to NaN even when
    # thousands of other valid pairs existed at that lag.
    df = pd.DataFrame(
        {
            "cell_id": [1, 1, 2, 2],
            "mars_year": [36, 36, 36, 36],
            "sol": [1, 2, 1, 2],
            "v": [1.0, 2.0, np.nan, 5.0],
        }
    )
    acf = compute_pooled_lag_autocorrelation(df, value_col="v", max_lag=1)
    assert not np.isnan(acf[1])  # the (1,2)-cell-1 pair must still contribute


def test_durbin_levinson_pacf_matches_acf_at_lag_one():
    acf = np.array([1.0, 0.5, 0.3, 0.1])
    pacf = durbin_levinson_pacf(acf)
    assert pacf[0] == pytest.approx(1.0)
    assert pacf[1] == pytest.approx(acf[1])


def test_durbin_levinson_pacf_zero_for_white_noise_acf():
    acf = np.array([1.0, 0.0, 0.0, 0.0])
    pacf = durbin_levinson_pacf(acf)
    assert pacf[1] == pytest.approx(0.0)
    assert pacf[2] == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# compute_semivariogram
# ---------------------------------------------------------------------------

def test_semivariogram_zero_for_identical_values_at_same_sol(grid):
    lat, lon = cell_centroid_lat_lon([0, 1], grid)
    df = pd.DataFrame(
        {
            "cell_id": [0, 1],
            "mars_year": [36, 36],
            "sol": [10, 10],
            "solar_longitude_ls": [50.0, 50.0],
            "dust_optical_depth_z": [2.0, 2.0],
        }
    )
    result = compute_semivariogram(
        df, grid,
        distance_bins_km=[0, 500, 1000],
        latitude_bands={"low": [-90, 90]},
    )
    assert (result["semivariance"] == 0.0).all()


def test_semivariogram_ignores_different_sol_pairs(grid):
    # Two cells observed on DIFFERENT sols must never be paired -- a
    # semivariogram is a same-sol spatial-structure statistic.
    df = pd.DataFrame(
        {
            "cell_id": [0, 1],
            "mars_year": [36, 36],
            "sol": [10, 11],
            "solar_longitude_ls": [50.0, 50.0],
            "dust_optical_depth_z": [2.0, 100.0],
        }
    )
    result = compute_semivariogram(
        df, grid,
        distance_bins_km=[0, 20000],
        latitude_bands={"low": [-90, 90]},
    )
    assert result.empty


# ---------------------------------------------------------------------------
# build_sol_to_ls_lookup / _ls_crossing_sols
# ---------------------------------------------------------------------------

def test_sol_to_ls_lookup_is_median_across_cells():
    aggregated = pd.DataFrame(
        {
            "mars_year": [36, 36, 36],
            "sol": [1, 1, 2],
            "solar_longitude_ls": [10.0, 12.0, 20.0],
            "cell_id": [1, 2, 1],
        }
    )
    lookup = build_sol_to_ls_lookup(aggregated, mars_year=36)
    assert lookup.loc[1] == pytest.approx(11.0)
    assert lookup.loc[2] == pytest.approx(20.0)


def test_ls_crossing_sols_interpolates_linearly():
    sol_to_ls = pd.Series([0.0, 10.0, 20.0], index=[100, 110, 120])
    crossings = _ls_crossing_sols(sol_to_ls, [5.0])
    assert crossings[5.0] == pytest.approx(105.0)


def test_ls_crossing_sols_skips_targets_outside_range():
    sol_to_ls = pd.Series([10.0, 20.0], index=[100, 110])
    crossings = _ls_crossing_sols(sol_to_ls, [5.0, 25.0])
    assert crossings == {}


# ---------------------------------------------------------------------------
# Smoke tests: each figure_4_* function runs end-to-end on small synthetic
# data without raising, and returns a real matplotlib Figure.
# ---------------------------------------------------------------------------

def test_figure_4_5_smoke():
    raw_qc = pd.DataFrame(
        [
            {"sol": 10, "mars_year": 36, "dust_optical_depth": 0.2, "qc_status": "accepted"},
            {"sol": 11, "mars_year": 36, "dust_optical_depth": 5.0, "qc_status": "rejected"},
            {"sol": 10, "mars_year": 37, "dust_optical_depth": 9.0, "qc_status": "accepted"},
        ]
    )
    aggregated_cell = pd.DataFrame([{"sol": 10, "dust_optical_depth": 0.2}])
    filled_cell = pd.DataFrame([{"sol": 10, "dust_optical_depth": 0.2, "interpolation_flag": False}])
    fig, source_df = figure_4_5_raw_qc_comparison(raw_qc, aggregated_cell, filled_cell, cell_id=1393, mars_year=36)
    assert fig is not None
    assert len(source_df) == 2  # the MY37 row must be excluded
    assert set(source_df["mars_year"]) == {36}


def test_figure_4_6_smoke(grid):
    lat, lon = cell_centroid_lat_lon(1000, grid)
    aggregated = pd.DataFrame(
        {
            "cell_id": [1000, 1000], "mars_year": [36, 36], "sol": [10, 11],
            "latitude": [lat, lat], "longitude": [lon, lon],
            "solar_longitude_ls": [50.0, 51.0], "local_true_solar_time": [12.0, 13.0],
            "dust_optical_depth": [0.2, 0.3],
        }
    )
    filled = aggregated.assign(is_observed=True)

    class _StubMCD:
        def query(self, **kwargs):
            n = len(kwargs["latitude"])
            return pd.DataFrame({"mu": [0.25] * n, "sigma": [0.1] * n})

    fig, source_df = figure_4_6_missingness_gapfill_distribution(
        aggregated, filled, _StubMCD(),
        representative_cell_id=1000, grid=grid, mars_year=36, sol_min=10, sol_max=11,
    )
    assert fig is not None
    assert source_df["climatology_mean_sample_averaged"].iloc[0] == pytest.approx(0.25)


def test_figure_4_7_smoke(grid):
    lat0, lon0 = cell_centroid_lat_lon(0, grid)
    lat1, lon1 = cell_centroid_lat_lon(1, grid)
    labeled = pd.DataFrame(
        [
            {"cell_id": 0, "mars_year": 36, "sol": 10, "episode_id": 0, "dust_optical_depth_z": 2.5, "solar_longitude_ls": 50.0},
            {"cell_id": 1, "mars_year": 36, "sol": 10, "episode_id": 0, "dust_optical_depth_z": 2.2, "solar_longitude_ls": 50.0},
            {"cell_id": 0, "mars_year": 36, "sol": 11, "episode_id": pd.NA, "dust_optical_depth_z": 1.0, "solar_longitude_ls": 51.0},
        ]
    )
    episodes = pd.DataFrame([{"episode_id": 0, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "qualifies": True}])
    config = {
        "episode_composite": {"window_sols_before": 1, "window_sols_after": 1},
        "acf": {"max_lag_sols": 2},
        "semivariogram": {"distance_bins_km": [0, 500, 10000], "latitude_bands": {"low": [-90, 90]}},
    }
    fig, source = figure_4_7_spatiotemporal_structure(labeled, episodes, grid, config=config)
    assert fig is not None
    assert set(source.keys()) == {"episode_composite", "acf_pacf", "semivariogram"}


def test_figure_4_8_smoke():
    aggregated = pd.DataFrame(
        {
            "cell_id": [5, 5, 5], "mars_year": [36, 36, 36], "sol": [1, 2, 3],
            "dust_optical_depth": [0.2, 0.25, 0.22],
            "solar_longitude_ls": [0.0, 45.0, 90.0],
        }
    )
    fig, source_df = figure_4_8_seasonal_series(aggregated, cell_id=5, mars_year=36)
    assert fig is not None
    assert len(source_df) == 3
