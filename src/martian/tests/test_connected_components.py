"""Tests for spatiotemporal episode construction: eight-neighbour spatial
connectivity, longitude wraparound, temporal linking across sols, and the
mars_year boundary (sol indices restart at 0 each Mars year)."""
import pandas as pd
import pytest

from martian.preprocessing.grid import GridDefinition, cell_id
from martian.labeling.connected_components import assign_episode_ids, summarize_episodes


@pytest.fixture
def grid() -> GridDefinition:
    return GridDefinition.from_resolution(5.0)


def _exceedance_row(lat_band, lon_band, sol, grid, mars_year=1, lat=10.0, lon=100.0) -> dict:
    return {
        "cell_id": int(cell_id(lat_band, lon_band, grid)),
        "mars_year": mars_year,
        "sol": sol,
        "latitude": lat,
        "longitude": lon,
    }


def test_eight_neighbour_spatial_connectivity_merges_adjacent_cells(grid):
    # Two spatially-adjacent cells (same sol) must merge into ONE episode.
    df = pd.DataFrame(
        [
            _exceedance_row(10, 10, 0, grid),
            _exceedance_row(10, 11, 0, grid),  # east neighbour
        ]
    )
    episode_ids = assign_episode_ids(df, grid)
    assert episode_ids.nunique() == 1


def test_non_adjacent_cells_remain_separate_episodes(grid):
    df = pd.DataFrame(
        [
            _exceedance_row(10, 10, 0, grid),
            _exceedance_row(10, 20, 0, grid),  # far away, not a neighbour
        ]
    )
    episode_ids = assign_episode_ids(df, grid)
    assert episode_ids.nunique() == 2


def test_longitude_wraparound_merges_components(grid):
    # lon_band 0 and lon_band (n_lon_bands - 1) are spatial neighbours under
    # wraparound -- must merge into one episode.
    df = pd.DataFrame(
        [
            _exceedance_row(10, 0, 0, grid),
            _exceedance_row(10, grid.n_lon_bands - 1, 0, grid),
        ]
    )
    episode_ids = assign_episode_ids(df, grid)
    assert episode_ids.nunique() == 1


def test_temporal_linking_same_cell_across_consecutive_sols(grid):
    # Same cell, three consecutive sols -- must be ONE episode (persistence).
    df = pd.DataFrame(
        [
            _exceedance_row(10, 10, 0, grid),
            _exceedance_row(10, 10, 1, grid),
            _exceedance_row(10, 10, 2, grid),
        ]
    )
    episode_ids = assign_episode_ids(df, grid)
    assert episode_ids.nunique() == 1


def test_non_consecutive_sols_same_cell_do_not_merge(grid):
    # Same cell but sols 0 and 5 (gap of 5, no bridging sols present) --
    # must be TWO separate episodes (no temporal edge beyond |delta_sol|=1).
    df = pd.DataFrame(
        [
            _exceedance_row(10, 10, 0, grid),
            _exceedance_row(10, 10, 5, grid),
        ]
    )
    episode_ids = assign_episode_ids(df, grid)
    assert episode_ids.nunique() == 2


def test_spatiotemporal_drift_merges_moving_storm(grid):
    # A storm that drifts to an adjacent cell on the NEXT sol is still one
    # episode (spatiotemporal, not merely spatial-same-sol or temporal-same-cell).
    df = pd.DataFrame(
        [
            _exceedance_row(10, 10, 0, grid),
            _exceedance_row(10, 11, 1, grid),  # adjacent cell, next sol
        ]
    )
    episode_ids = assign_episode_ids(df, grid)
    assert episode_ids.nunique() == 1


def test_mars_year_boundary_never_merges_same_cell(grid):
    # Same cell_id, sol=667 of MY1 and sol=0 of MY2: numerically |667-0|=667,
    # so this would already fail the |delta_sol|<=1 rule -- but the
    # regression risk is a bug that keys only on (cell_id, sol) and ignores
    # mars_year. Assert explicitly they never merge.
    df = pd.DataFrame(
        [
            _exceedance_row(10, 10, 667, grid, mars_year=1),
            _exceedance_row(10, 10, 0, grid, mars_year=2),
        ]
    )
    episode_ids = assign_episode_ids(df, grid)
    assert episode_ids.nunique() == 2


def test_mars_year_same_sol_number_different_year_never_merges(grid):
    # Same cell_id, SAME sol number (5), but different mars_year -- must
    # never merge even though (cell_id, sol) alone would collide.
    df = pd.DataFrame(
        [
            _exceedance_row(10, 10, 5, grid, mars_year=1),
            _exceedance_row(10, 10, 5, grid, mars_year=2),
        ]
    )
    episode_ids = assign_episode_ids(df, grid)
    assert episode_ids.nunique() == 2


def test_pole_row_neighbour_count_still_respected_in_episode_construction(grid):
    # A cell in the topmost latitude band only has 5 spatial neighbours (no
    # row beyond the pole) -- confirm a cell OUTSIDE that reduced neighbour
    # set does not spuriously merge.
    top_band = grid.n_lat_bands - 1
    df = pd.DataFrame(
        [
            _exceedance_row(top_band, 10, 0, grid),
            _exceedance_row(top_band - 2, 10, 0, grid),  # two rows away -- not a neighbour
        ]
    )
    episode_ids = assign_episode_ids(df, grid)
    assert episode_ids.nunique() == 2


def test_empty_exceedance_dataframe_returns_empty_series(grid):
    df = pd.DataFrame(columns=["cell_id", "mars_year", "sol", "latitude", "longitude"])
    episode_ids = assign_episode_ids(df, grid)
    assert len(episode_ids) == 0


def test_summarize_episodes_duration_and_peak_area_hand_computed(grid):
    df = pd.DataFrame(
        [
            _exceedance_row(10, 10, 0, grid),
            _exceedance_row(10, 11, 0, grid),  # same sol, 2 cells -> peak_area=2
            _exceedance_row(10, 10, 1, grid),  # next sol, 1 cell
        ]
    )
    episode_ids = assign_episode_ids(df, grid)
    df = df.copy()
    df["episode_id"] = episode_ids
    catalogue = summarize_episodes(df)
    assert len(catalogue) == 1
    row = catalogue.iloc[0]
    assert row["onset_sol"] == 0
    assert row["end_sol"] == 1
    assert row["duration_sols"] == 2
    assert row["peak_area_cells"] == 2
    assert row["n_distinct_cells"] == 2
    assert row["n_member_cell_sols"] == 3
