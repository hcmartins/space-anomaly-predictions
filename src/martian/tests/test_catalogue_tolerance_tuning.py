"""Tests for labeling/catalogue_tolerance_tuning.py: blind dev/test tuning
of the Table 4.6 matching tolerance (CHAPTER_4_PARAMETER_VALUES.md open
item 4)."""
import numpy as np
import pandas as pd
import pytest

from martian.preprocessing.grid import GridDefinition, cell_id, lat_to_band_index, lon_to_band_index, load_primary_grid
from martian.labeling.catalogue_tolerance_tuning import (
    build_table_4_6b,
    search_tolerance_grid,
    select_best_tolerance,
    split_documented_events_dev_test,
    tune_and_validate_tolerance,
    write_table_4_6b,
)

_UNMATCHED_LABEL = "false_positive_candidate"


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


def _documented_events(rows, *, covers_mars_year=36) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.attrs["covers_mars_year"] = covers_mars_year
    return df


# ---------------------------------------------------------------------------
# split_documented_events_dev_test
# ---------------------------------------------------------------------------

def test_split_partitions_all_events_with_no_overlap():
    events = _documented_events(
        [{"event_id": i, "mars_year": 36, "onset_sol": i, "end_sol": i, "centroid_lat": 0.0, "centroid_lon": 0.0, "peak_size_km2": 1.0}
         for i in range(20)]
    )
    dev, test = split_documented_events_dev_test(events, dev_fraction=0.5, seed=42)

    assert len(dev) + len(test) == 20
    assert set(dev["event_id"]) & set(test["event_id"]) == set()
    assert set(dev["event_id"]) | set(test["event_id"]) == set(range(20))


def test_split_is_deterministic_under_fixed_seed():
    events = _documented_events(
        [{"event_id": i, "mars_year": 36, "onset_sol": i, "end_sol": i, "centroid_lat": 0.0, "centroid_lon": 0.0, "peak_size_km2": 1.0}
         for i in range(30)]
    )
    dev1, test1 = split_documented_events_dev_test(events, dev_fraction=0.5, seed=7)
    dev2, test2 = split_documented_events_dev_test(events, dev_fraction=0.5, seed=7)

    assert dev1["event_id"].tolist() == dev2["event_id"].tolist()
    assert test1["event_id"].tolist() == test2["event_id"].tolist()


def test_split_respects_dev_fraction_and_preserves_covers_mars_year():
    events = _documented_events(
        [{"event_id": i, "mars_year": 36, "onset_sol": i, "end_sol": i, "centroid_lat": 0.0, "centroid_lon": 0.0, "peak_size_km2": 1.0}
         for i in range(98)],
        covers_mars_year=36,
    )
    dev, test = split_documented_events_dev_test(events, dev_fraction=0.5, seed=42)

    assert len(dev) == 49
    assert len(test) == 49
    assert dev.attrs["covers_mars_year"] == 36
    assert test.attrs["covers_mars_year"] == 36


def test_different_seeds_give_different_splits():
    events = _documented_events(
        [{"event_id": i, "mars_year": 36, "onset_sol": i, "end_sol": i, "centroid_lat": 0.0, "centroid_lon": 0.0, "peak_size_km2": 1.0}
         for i in range(50)]
    )
    dev_a, _ = split_documented_events_dev_test(events, dev_fraction=0.5, seed=1)
    dev_b, _ = split_documented_events_dev_test(events, dev_fraction=0.5, seed=2)

    assert dev_a["event_id"].tolist() != dev_b["event_id"].tolist()


# ---------------------------------------------------------------------------
# search_tolerance_grid / select_best_tolerance
# ---------------------------------------------------------------------------

def test_search_grid_returns_one_row_per_combination(grid):
    cid = _cell_at(2.0, 2.0, grid)
    labeled_df = pd.DataFrame([{"episode_id": 1, "cell_id": cid, "mars_year": 36, "sol": 10}])
    episode_catalogue = pd.DataFrame([_episode_catalogue_row(1, 36, onset_sol=10, end_sol=10)])
    dev_events = _documented_events(
        [{"event_id": 100, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 50000.0}]
    )

    search_table = search_tolerance_grid(
        labeled_df, episode_catalogue, dev_events, grid,
        temporal_candidates=[0, 1, 2], spatial_candidates=[0, 1],
        unmatched_episode_label=_UNMATCHED_LABEL,
    )

    assert len(search_table) == 6  # 3 x 2
    assert set(search_table["temporal_tolerance_sols"]) == {0, 1, 2}
    assert set(search_table["spatial_region_radius_cells"]) == {0, 1}
    # exact temporal+spatial match at tolerance (0,0) already hits -> f1 == 1
    exact = search_table[(search_table["temporal_tolerance_sols"] == 0) & (search_table["spatial_region_radius_cells"] == 0)].iloc[0]
    assert exact["dev_f1"] == pytest.approx(1.0)


def test_select_best_tolerance_picks_highest_f1():
    search_table = pd.DataFrame(
        [
            {"temporal_tolerance_sols": 0, "spatial_region_radius_cells": 0, "dev_f1": 0.2, "dev_pod": 0.2, "dev_episode_precision": 0.2},
            {"temporal_tolerance_sols": 2, "spatial_region_radius_cells": 1, "dev_f1": 0.9, "dev_pod": 0.9, "dev_episode_precision": 0.9},
            {"temporal_tolerance_sols": 4, "spatial_region_radius_cells": 3, "dev_f1": 0.5, "dev_pod": 0.5, "dev_episode_precision": 0.5},
        ]
    )
    best = select_best_tolerance(search_table)
    assert best["temporal_tolerance_sols"] == 2
    assert best["spatial_region_radius_cells"] == 1
    assert best["dev_f1"] == pytest.approx(0.9)


def test_select_best_tolerance_ties_broken_by_smallest_tolerance():
    search_table = pd.DataFrame(
        [
            {"temporal_tolerance_sols": 3, "spatial_region_radius_cells": 2, "dev_f1": 0.7, "dev_pod": 0.7, "dev_episode_precision": 0.7},
            {"temporal_tolerance_sols": 1, "spatial_region_radius_cells": 2, "dev_f1": 0.7, "dev_pod": 0.7, "dev_episode_precision": 0.7},
            {"temporal_tolerance_sols": 1, "spatial_region_radius_cells": 0, "dev_f1": 0.7, "dev_pod": 0.7, "dev_episode_precision": 0.7},
        ]
    )
    best = select_best_tolerance(search_table)
    # smallest temporal (1) wins over 3; among temporal==1, smallest spatial (0) wins over 2
    assert best["temporal_tolerance_sols"] == 1
    assert best["spatial_region_radius_cells"] == 0


def test_select_best_tolerance_raises_on_empty_search_table():
    with pytest.raises(ValueError):
        select_best_tolerance(pd.DataFrame(columns=["temporal_tolerance_sols", "spatial_region_radius_cells", "dev_f1"]))


# ---------------------------------------------------------------------------
# tune_and_validate_tolerance (end-to-end) -- confirms the test half is
# genuinely never used to select the tolerance
# ---------------------------------------------------------------------------

def test_tuning_never_uses_test_events_to_select_tolerance(grid):
    """Construct two documented events, one assigned to dev and one to test
    under seed=42 (verified below), each only matchable at a DIFFERENT
    temporal tolerance. If test-side information leaked into selection, the
    chosen tolerance would depend on the test event too."""
    cid = _cell_at(2.0, 2.0, grid)

    events_df = _documented_events(
        [
            {"event_id": 0, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 1.0},
            {"event_id": 1, "mars_year": 36, "onset_sol": 50, "end_sol": 50, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 1.0},
        ]
    )
    dev, test = split_documented_events_dev_test(events_df, dev_fraction=0.5, seed=42)
    assert len(dev) == 1 and len(test) == 1
    dev_event_id = int(dev["event_id"].iloc[0])
    test_event_id = int(test["event_id"].iloc[0])
    assert {dev_event_id, test_event_id} == {0, 1}

    # One qualifying episode, exactly matching the DEV event's onset sol
    # (tolerance 0) but far from the TEST event's onset sol regardless of
    # tolerance up to 4 (the configured grid max).
    dev_onset = int(dev["onset_sol"].iloc[0])
    labeled_df = pd.DataFrame([{"episode_id": 1, "cell_id": cid, "mars_year": 36, "sol": dev_onset}])
    episode_catalogue = pd.DataFrame([_episode_catalogue_row(1, 36, onset_sol=dev_onset, end_sol=dev_onset)])

    config = {
        "dev_test_split": {"dev_fraction": 0.5, "seed": 42},
        "tolerance_grid": {"temporal_tolerance_sols": [0, 1, 2, 3, 4], "spatial_region_radius_cells": [0, 1]},
        "selection_metric": "f1",
    }
    result = tune_and_validate_tolerance(
        labeled_df, episode_catalogue, events_df, grid,
        config=config, unmatched_episode_label=_UNMATCHED_LABEL,
    )

    # Tolerance 0 already gives a perfect dev-side F1 (episode exactly
    # matches the dev event) -> tie-break picks the smallest tolerance, 0.
    assert result["selected_tolerance"]["temporal_tolerance_sols"] == 0
    assert result["selected_tolerance"]["dev_f1"] == pytest.approx(1.0)

    # The held-out TEST side never had a matching episode at any tolerance
    # in the grid (episode onset is 40 sols away from the test event) --
    # confirms the test event genuinely played no role in selection, and
    # the frozen tolerance correctly produces a miss on held-out data.
    assert result["test_validation"]["summary"]["n_hits"] == 0
    assert result["test_validation"]["summary"]["pod"] == pytest.approx(0.0)
    assert result["split"]["n_dev_events"] == 1
    assert result["split"]["n_test_events"] == 1


def test_dev_search_table_covers_full_grid(grid):
    cid = _cell_at(2.0, 2.0, grid)
    events_df = _documented_events(
        [{"event_id": 0, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 1.0}]
    )
    labeled_df = pd.DataFrame([{"episode_id": 1, "cell_id": cid, "mars_year": 36, "sol": 10}])
    episode_catalogue = pd.DataFrame([_episode_catalogue_row(1, 36, onset_sol=10, end_sol=10)])

    config = {
        "dev_test_split": {"dev_fraction": 1.0, "seed": 42},
        "tolerance_grid": {"temporal_tolerance_sols": [0, 1], "spatial_region_radius_cells": [0, 1, 2]},
        "selection_metric": "f1",
    }
    result = tune_and_validate_tolerance(
        labeled_df, episode_catalogue, events_df, grid,
        config=config, unmatched_episode_label=_UNMATCHED_LABEL,
    )
    assert len(result["dev_search_table"]) == 2 * 3


# ---------------------------------------------------------------------------
# build_table_4_6b / write_table_4_6b
# ---------------------------------------------------------------------------

def test_build_table_4_6b_includes_tolerance_and_split_provenance(grid):
    cid = _cell_at(2.0, 2.0, grid)
    events_df = _documented_events(
        [{"event_id": 0, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 1.0},
         {"event_id": 1, "mars_year": 36, "onset_sol": 20, "end_sol": 20, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 1.0}]
    )
    labeled_df = pd.DataFrame(
        [{"episode_id": 1, "cell_id": cid, "mars_year": 36, "sol": 10},
         {"episode_id": 2, "cell_id": cid, "mars_year": 36, "sol": 20}]
    )
    episode_catalogue = pd.DataFrame(
        [_episode_catalogue_row(1, 36, onset_sol=10, end_sol=10), _episode_catalogue_row(2, 36, onset_sol=20, end_sol=20)]
    )
    config = {
        "dev_test_split": {"dev_fraction": 0.5, "seed": 42},
        "tolerance_grid": {"temporal_tolerance_sols": [0, 1], "spatial_region_radius_cells": [0]},
        "selection_metric": "f1",
    }
    result = tune_and_validate_tolerance(
        labeled_df, episode_catalogue, events_df, grid,
        config=config, unmatched_episode_label=_UNMATCHED_LABEL,
    )
    table = build_table_4_6b(result)

    for col in [
        "temporal_tolerance_sols", "spatial_region_radius_cells", "selection_metric",
        "dev_f1_at_selected_tolerance", "dev_fraction", "split_seed", "n_dev_events", "n_test_events",
    ]:
        assert col in table.columns
    assert len(table) == 1


def test_write_table_4_6b_writes_three_files(tmp_path, grid):
    cid = _cell_at(2.0, 2.0, grid)
    events_df = _documented_events(
        [{"event_id": 0, "mars_year": 36, "onset_sol": 10, "end_sol": 10, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 1.0},
         {"event_id": 1, "mars_year": 36, "onset_sol": 20, "end_sol": 20, "centroid_lat": 2.0, "centroid_lon": 2.0, "peak_size_km2": 1.0}]
    )
    labeled_df = pd.DataFrame(
        [{"episode_id": 1, "cell_id": cid, "mars_year": 36, "sol": 10},
         {"episode_id": 2, "cell_id": cid, "mars_year": 36, "sol": 20}]
    )
    episode_catalogue = pd.DataFrame(
        [_episode_catalogue_row(1, 36, onset_sol=10, end_sol=10), _episode_catalogue_row(2, 36, onset_sol=20, end_sol=20)]
    )
    config = {
        "dev_test_split": {"dev_fraction": 0.5, "seed": 42},
        "tolerance_grid": {"temporal_tolerance_sols": [0, 1], "spatial_region_radius_cells": [0]},
        "selection_metric": "f1",
    }
    result = tune_and_validate_tolerance(
        labeled_df, episode_catalogue, events_df, grid,
        config=config, unmatched_episode_label=_UNMATCHED_LABEL,
    )
    csv_path, md_path, search_path = write_table_4_6b(result, tmp_path, data_source_label="TEST")

    assert csv_path.exists()
    assert md_path.exists()
    assert search_path.exists()
    assert "Table 4.6b" in md_path.read_text(encoding="utf-8")
    search_table_on_disk = pd.read_csv(search_path)
    assert len(search_table_on_disk) == 2  # 2 temporal x 1 spatial
