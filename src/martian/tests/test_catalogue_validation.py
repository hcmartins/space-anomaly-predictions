"""Tests for labeling/catalogue_validation.py: matching detected
storm_episodes against the real, independently-sourced documented
dust-storm catalogue (Table 4.6)."""
import pandas as pd
import pytest

from martian.preprocessing.grid import GridDefinition, cell_id, lat_to_band_index, lon_to_band_index, load_primary_grid
from martian.labeling.catalogue_validation import (
    build_table_4_6,
    find_matching_pairs,
    run_catalogue_validation,
)

_DEFAULT_CONFIG = {
    "temporal_tolerance_sols": 2,
    "spatial_region_radius_cells": 1,
    "unmatched_episode_label": "false_positive_candidate",
}


@pytest.fixture
def grid() -> GridDefinition:
    return load_primary_grid()


def _cell_at(lat: float, lon: float, grid: GridDefinition) -> int:
    lat_band = int(lat_to_band_index([lat], grid)[0])
    lon_band = int(lon_to_band_index([lon], grid)[0])
    return int(cell_id(lat_band, lon_band, grid))


def _episode_catalogue_row(episode_id, mars_year, onset_sol, end_sol, *, qualifies=True, taxonomy_class="local"):
    return {
        "episode_id": episode_id, "mars_year": mars_year, "onset_sol": onset_sol, "end_sol": end_sol,
        "qualifies": qualifies, "taxonomy_class": taxonomy_class,
    }


def _documented_events(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.attrs["covers_mars_year"] = 36
    return df


def test_hit_when_temporal_and_spatial_criteria_both_satisfied(grid):
    cid = _cell_at(2.0, 2.0, grid)
    labeled_df = pd.DataFrame([{"episode_id": 1, "cell_id": cid, "mars_year": 36, "sol": 11}])
    episode_catalogue = pd.DataFrame([_episode_catalogue_row(1, 36, onset_sol=11, end_sol=11)])
    documented_events = _documented_events([
        {"event_id": 100, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 50000.0}
    ])

    result = run_catalogue_validation(labeled_df, episode_catalogue, documented_events, grid, config=_DEFAULT_CONFIG)

    assert result["summary"]["n_hits"] == 1
    assert result["summary"]["n_misses"] == 0
    assert result["summary"]["pod"] == pytest.approx(1.0)
    assert result["summary"]["n_false_positive_candidates"] == 0
    assert result["summary"]["episode_precision"] == pytest.approx(1.0)


def test_miss_when_no_spatial_overlap(grid):
    episode_cell = _cell_at(2.0, 2.0, grid)
    labeled_df = pd.DataFrame([{"episode_id": 1, "cell_id": episode_cell, "mars_year": 36, "sol": 10}])
    episode_catalogue = pd.DataFrame([_episode_catalogue_row(1, 36, onset_sol=10, end_sol=10)])
    documented_events = _documented_events([
        {"event_id": 100, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "centroid_lat": -80.0, "centroid_lon": 200.0, "peak_size_km2": 50000.0}
    ])

    result = run_catalogue_validation(labeled_df, episode_catalogue, documented_events, grid, config=_DEFAULT_CONFIG)

    assert result["summary"]["n_hits"] == 0
    assert result["summary"]["n_misses"] == 1
    assert result["summary"]["pod"] == pytest.approx(0.0)
    # the episode also gets no match -> false positive candidate
    assert result["summary"]["n_false_positive_candidates"] == 1


def test_temporal_tolerance_boundary_exact_vs_exceeded(grid):
    cid = _cell_at(2.0, 2.0, grid)

    def _run(episode_onset_sol):
        labeled_df = pd.DataFrame([{"episode_id": 1, "cell_id": cid, "mars_year": 36, "sol": episode_onset_sol}])
        episode_catalogue = pd.DataFrame([_episode_catalogue_row(1, 36, onset_sol=episode_onset_sol, end_sol=episode_onset_sol)])
        documented_events = _documented_events([
            {"event_id": 100, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 50000.0}
        ])
        return run_catalogue_validation(labeled_df, episode_catalogue, documented_events, grid, config=_DEFAULT_CONFIG)

    # tolerance=2: documented window expands to [8, 12] -- episode at sol=12 is exactly at the boundary (hit),
    # episode at sol=13 is one sol beyond it (miss).
    assert _run(12)["summary"]["n_hits"] == 1
    assert _run(13)["summary"]["n_hits"] == 0


def test_false_positive_candidate_when_episode_has_no_documented_match(grid):
    cid = _cell_at(2.0, 2.0, grid)
    labeled_df = pd.DataFrame([{"episode_id": 1, "cell_id": cid, "mars_year": 36, "sol": 10}])
    episode_catalogue = pd.DataFrame([_episode_catalogue_row(1, 36, onset_sol=10, end_sol=10)])
    documented_events = _documented_events(
        [{"event_id": 100, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "centroid_lat": -80.0, "centroid_lon": 200.0, "peak_size_km2": 50000.0}]
    )

    result = run_catalogue_validation(labeled_df, episode_catalogue, documented_events, grid, config=_DEFAULT_CONFIG)

    assert result["summary"]["n_false_positive_candidates"] == 1
    per_episode = result["per_episode"]
    assert per_episode.loc[per_episode["episode_id"] == 1, "status"].iloc[0] == "false_positive_candidate"


def test_non_qualifying_episode_never_matched(grid):
    cid = _cell_at(2.0, 2.0, grid)
    labeled_df = pd.DataFrame([{"episode_id": 1, "cell_id": cid, "mars_year": 36, "sol": 10}])
    episode_catalogue = pd.DataFrame([_episode_catalogue_row(1, 36, onset_sol=10, end_sol=10, qualifies=False)])
    documented_events = _documented_events(
        [{"event_id": 100, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 50000.0}]
    )

    result = run_catalogue_validation(labeled_df, episode_catalogue, documented_events, grid, config=_DEFAULT_CONFIG)

    assert result["summary"]["n_hits"] == 0
    assert result["summary"]["n_qualifying_episodes"] == 0


def test_restricted_to_documented_catalogue_mars_year(grid):
    cid = _cell_at(2.0, 2.0, grid)
    # episode in MY37 -- documented catalogue only covers MY36
    labeled_df = pd.DataFrame([{"episode_id": 1, "cell_id": cid, "mars_year": 37, "sol": 10}])
    episode_catalogue = pd.DataFrame([_episode_catalogue_row(1, 37, onset_sol=10, end_sol=10)])
    documented_events = _documented_events(
        [{"event_id": 100, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 50000.0}]
    )

    result = run_catalogue_validation(labeled_df, episode_catalogue, documented_events, grid, config=_DEFAULT_CONFIG)

    assert result["summary"]["n_qualifying_episodes"] == 0
    assert result["summary"]["n_misses"] == 1


def test_find_matching_pairs_returns_empty_dataframe_with_correct_columns_when_no_matches(grid):
    labeled_df = pd.DataFrame(columns=["episode_id", "cell_id", "mars_year", "sol"])
    episode_catalogue = pd.DataFrame(columns=["episode_id", "mars_year", "onset_sol", "end_sol", "qualifies", "taxonomy_class"])
    documented_events = _documented_events([])
    pairs = find_matching_pairs(
        labeled_df, episode_catalogue, documented_events, grid,
        temporal_tolerance_sols=2, spatial_region_radius_cells=1,
    )
    assert list(pairs.columns) == ["event_id", "episode_id", "temporal_offset_sols"]
    assert len(pairs) == 0


def test_build_table_4_6_reads_directly_from_summary(grid):
    cid = _cell_at(2.0, 2.0, grid)
    labeled_df = pd.DataFrame([{"episode_id": 1, "cell_id": cid, "mars_year": 36, "sol": 10}])
    episode_catalogue = pd.DataFrame([_episode_catalogue_row(1, 36, onset_sol=10, end_sol=10)])
    documented_events = _documented_events(
        [{"event_id": 100, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 50000.0}]
    )
    result = run_catalogue_validation(labeled_df, episode_catalogue, documented_events, grid, config=_DEFAULT_CONFIG)
    table = build_table_4_6(result)
    assert table.loc[0, "n_hits"] == result["summary"]["n_hits"]
    assert table.loc[0, "pod"] == pytest.approx(result["summary"]["pod"])
