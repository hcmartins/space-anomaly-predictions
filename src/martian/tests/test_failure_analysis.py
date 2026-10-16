"""Tests for failure-analysis breakdowns across the required dimensions:
season, latitude band, local time, retrieval density, interpolation status,
uncertainty, event category, label persistence, and spatial extent."""
import numpy as np
import pandas as pd

from martian.evaluation.failure_analysis import (
    breakdown_by_event_category,
    breakdown_by_interpolation_status,
    breakdown_by_label_persistence,
    breakdown_by_latitude_band,
    breakdown_by_local_time,
    breakdown_by_retrieval_density,
    breakdown_by_season,
    breakdown_by_spatial_extent,
    breakdown_by_uncertainty,
    run_all_breakdowns,
)
from martian.modeling.metrics import regression_metrics


def _df(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "solar_longitude_ls": rng.uniform(0, 360, n),
            "latitude": rng.uniform(-90, 90, n),
            "local_true_solar_time": rng.uniform(0, 24, n),
            "observation_count": rng.integers(1, 8, n),
            "uncertainty_summary": rng.uniform(0.01, 0.1, n),
            "is_observed": [True] * n,
            "y_true": rng.normal(size=n),
        }
    )
    df["y_pred"] = df["y_true"] + rng.normal(scale=0.2, size=n)
    return df


def test_breakdown_by_season_covers_all_rows():
    df = _df()
    table = breakdown_by_season(df, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics)
    assert table["n"].sum() == len(df)
    assert set(table["group"]) <= {
        "Northern Spring (Ls 0-90)", "Northern Summer (Ls 90-180)",
        "Northern Autumn (Ls 180-270)", "Northern Winter (Ls 270-360)",
    }


def test_breakdown_by_season_missing_column_returns_empty():
    df = _df().drop(columns=["solar_longitude_ls"])
    table = breakdown_by_season(df, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics)
    assert table.empty


def test_breakdown_by_latitude_band_hand_computed():
    df = pd.DataFrame({"latitude": [-85.0, 5.0, 85.0], "y_true": [1.0, 2.0, 3.0], "y_pred": [1.0, 2.0, 3.0]})
    table = breakdown_by_latitude_band(df, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics, band_width_deg=30.0)
    assert table["n"].sum() == 3
    assert len(table) == 3  # three different 30-degree bands


def test_breakdown_by_local_time_bins():
    df = pd.DataFrame({"local_true_solar_time": [1.0, 7.0, 13.0, 19.0], "y_true": [1.0] * 4, "y_pred": [1.0] * 4})
    table = breakdown_by_local_time(df, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics, bin_width_hours=6.0)
    assert table["n"].sum() == 4
    assert len(table) == 4  # each falls in a distinct 6-hour bin


def test_breakdown_by_retrieval_density_bins():
    df = pd.DataFrame({"observation_count": [1, 2, 3, 4, 5, 10], "y_true": [1.0] * 6, "y_pred": [1.0] * 6})
    table = breakdown_by_retrieval_density(
        df, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics, bin_edges=[1, 2, 3, 5]
    )
    assert table["n"].sum() == 6
    by_group = dict(zip(table["group"], table["n"]))
    assert by_group["1-1"] == 1
    assert by_group["2-2"] == 1
    assert by_group["3-4"] == 2  # observation_count 3 and 4
    assert by_group["5+"] == 2  # observation_count 5 and 10


def test_breakdown_by_interpolation_status_degenerate_when_all_observed():
    df = _df()
    table = breakdown_by_interpolation_status(df, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics)
    assert len(table) == 1
    assert table["group"].iloc[0] == "observed"
    assert table["n"].iloc[0] == len(df)


def test_breakdown_by_uncertainty_quantile_bins():
    df = _df(n=100)
    table = breakdown_by_uncertainty(df, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics, n_bins=4)
    assert table["n"].sum() == 100
    assert len(table) <= 4


def test_breakdown_by_uncertainty_too_few_rows_returns_empty():
    df = _df(n=2)
    table = breakdown_by_uncertainty(df, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics, n_bins=4)
    assert table.empty


def test_event_category_breakdown_quiet_vs_episode():
    df = pd.DataFrame(
        {
            "episode_id": [0, 1, np.nan, np.nan],
            "y_true": [1, 1, 0, 0],
            "y_pred": [1, 1, 0, 0],
        }
    )
    catalogue = pd.DataFrame([{"episode_id": 0, "taxonomy_class": "local"}, {"episode_id": 1, "taxonomy_class": "regional"}])
    table = breakdown_by_event_category(df, catalogue, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics)
    by_group = dict(zip(table["group"], table["n"]))
    assert by_group["local"] == 1
    assert by_group["regional"] == 1
    assert by_group["quiet"] == 2


def test_event_category_breakdown_no_catalogue_returns_empty():
    df = pd.DataFrame({"episode_id": [0], "y_true": [1], "y_pred": [1]})
    table = breakdown_by_event_category(df, None, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics)
    assert table.empty


def test_label_persistence_breakdown():
    df = pd.DataFrame({"episode_id": [0, np.nan], "y_true": [1, 0], "y_pred": [1, 0]})
    catalogue = pd.DataFrame([{"episode_id": 0, "duration_sols": 3}])
    table = breakdown_by_label_persistence(df, catalogue, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics)
    assert set(table["group"]) == {"3sol", "quiet"}


def test_spatial_extent_breakdown():
    df = pd.DataFrame({"episode_id": [0, np.nan], "y_true": [1, 0], "y_pred": [1, 0]})
    catalogue = pd.DataFrame([{"episode_id": 0, "peak_area_cells": 5}])
    table = breakdown_by_spatial_extent(df, catalogue, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics)
    assert set(table["group"]) == {"5cell(s)", "quiet"}


def test_run_all_breakdowns_returns_expected_dimensions_without_catalogue():
    df = _df()
    out = run_all_breakdowns(df, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics)
    assert set(out.keys()) == {
        "season", "latitude_band", "local_time", "retrieval_density",
        "interpolation_status", "uncertainty",
    }


def test_run_all_breakdowns_adds_event_dimensions_with_catalogue():
    df = _df()
    df["episode_id"] = np.nan
    catalogue = pd.DataFrame(columns=["episode_id", "taxonomy_class", "duration_sols", "peak_area_cells"])
    out = run_all_breakdowns(df, y_true_col="y_true", y_pred_col="y_pred", metric_fn=regression_metrics, episode_catalogue=catalogue)
    assert {"event_category", "label_persistence", "spatial_extent"} <= set(out.keys())
