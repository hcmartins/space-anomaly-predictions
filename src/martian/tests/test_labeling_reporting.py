"""Tests for labeling prevalence tables, split/season tagging, and the
episode map/timeline figures."""
import pandas as pd
import pytest

from split_fixtures import mechanics_splits

from martian.preprocessing.grid import load_primary_grid
from martian.labeling.reporting import (
    assign_mars_season,
    assign_split,
    build_prevalence_tables,
    episode_map,
    episode_timeline,
    load_splits_config,
    prevalence_by_horizon,
)


def test_assign_split_boundaries():
    splits_config = mechanics_splits()
    df = pd.DataFrame(
        {
            "mars_year": [36, 36, 36, 37],
            "sol": [300, 650, 601, 10],
        }
    )
    split = assign_split(df, splits_config)
    assert split.tolist() == ["train", "val", "val", "test"]


def test_assign_split_unassigned_outside_all_ranges():
    splits_config = mechanics_splits()
    df = pd.DataFrame({"mars_year": [3], "sol": [10]})
    split = assign_split(df, splits_config)
    assert split.tolist() == ["unassigned"]


def test_assign_mars_season_quadrants():
    ls = pd.Series([0.0, 45.0, 90.0, 179.9, 270.0, 359.9])
    seasons = assign_mars_season(ls)
    assert seasons[0].startswith("Northern Spring")
    assert seasons[1].startswith("Northern Spring")
    assert seasons[2].startswith("Northern Summer")
    assert seasons[3].startswith("Northern Summer")
    assert seasons[4].startswith("Northern Winter")
    assert seasons[5].startswith("Northern Winter")


def _labeled_df_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [0, 1, 2, 3],
            "mars_year": [1, 1, 1, 2],
            "sol": [10, 300, 650, 5],
            "solar_longitude_ls": [10.0, 100.0, 200.0, 300.0],
            "dust_anomaly": [True, False, False, True],
            "temperature_anomaly": [False, True, False, False],
            "atmospheric_anomaly": [True, True, False, True],
            "storm_episode": [True, False, False, True],
            "horizon_1sol": [False, False, False, True],
            "horizon_3sol": [True, False, False, True],
            "horizon_5sol": [True, True, False, True],
        }
    )


def test_prevalence_by_horizon_matches_direct_counts():
    df = _labeled_df_fixture()
    table = prevalence_by_horizon(df)
    by_horizon = table.set_index("horizon")
    assert by_horizon.loc["horizon_1sol", "n_positive"] == 1
    assert by_horizon.loc["horizon_3sol", "n_positive"] == 2
    assert by_horizon.loc["horizon_5sol", "n_positive"] == 3


def test_build_prevalence_tables_shape():
    df = _labeled_df_fixture()
    catalogue = pd.DataFrame(
        [{"episode_id": 0, "taxonomy_class": "local", "qualifies": True, "duration_sols": 1, "peak_area_cells": 1}]
    )
    tables = build_prevalence_tables(df, catalogue)
    assert set(tables.keys()) == {
        "prevalence_by_split", "prevalence_by_season", "prevalence_by_horizon", "prevalence_by_storm_category",
    }
    assert tables["prevalence_by_storm_category"].loc[0, "taxonomy_class"] == "local"
    assert tables["prevalence_by_storm_category"].loc[0, "n_episodes"] == 1


def test_episode_map_and_timeline_are_stamped():
    grid = load_primary_grid()
    df = pd.DataFrame(
        {
            "cell_id": [0, 1],
            "storm_episode": [True, False],
        }
    )
    fig1 = episode_map(df, grid, "SIMULATED (test)")
    assert any("SIMULATED (test)" in t.get_text() for t in fig1.texts)

    catalogue = pd.DataFrame(
        [
            {"episode_id": 0, "onset_sol": 0, "end_sol": 2, "taxonomy_class": "local", "qualifies": True},
            {"episode_id": 1, "onset_sol": 5, "end_sol": 5, "taxonomy_class": "regional", "qualifies": True},
        ]
    )
    fig2 = episode_timeline(catalogue, "SIMULATED (test)")
    assert any("SIMULATED (test)" in t.get_text() for t in fig2.texts)
