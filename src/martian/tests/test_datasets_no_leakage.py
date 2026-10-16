"""Tests proving cross-task target leakage is structurally prevented:
no task's feature manifest ever contains a `target_` column, no task's
feature manifest contains another task's specific target column, target
column names are mutually exclusive across tasks, and a task's target is
NEVER read from a trailing-filled (fabricated) future value."""
import numpy as np
import pandas as pd
import pytest

from martian.preprocessing.grid import GridDefinition, assign_cells
from martian.preprocessing.features import build_features
from martian.modeling.datasets import (
    TASK_TARGET_PREFIXES,
    build_dust_regression_dataset,
    build_storm_detection_dataset,
)


@pytest.fixture
def grid() -> GridDefinition:
    return GridDefinition.from_resolution(5.0)


def _make_labeled_and_features(grid, sols, dust_values, storm_flags):
    n = len(sols)
    base = pd.DataFrame(
        {
            "latitude": [10.0] * n,
            "longitude": [100.0] * n,
            "mars_year": [1] * n,
            "sol": sols,
            "solar_longitude_ls": [45.0] * n,
            "local_true_solar_time": [14.0] * n,
            "dust_optical_depth": dust_values,
            "dust_optical_depth_z": [0.0] * n,
            "dust_anomaly": [False] * n,
            "atmospheric_anomaly": storm_flags,
            "is_observed": [True] * n,
            "interpolation_flag": [False] * n,
            "low_support_flag": [False] * n,
            "observation_count": [1] * n,
        }
    )
    base = assign_cells(base, grid)

    labeled_df = base.copy()
    labeled_df["storm_episode"] = storm_flags
    labeled_df["episode_id"] = range(n)
    for h in (1, 3, 5):
        # any storm within (s, s+h] using EXACT sol matching for this test fixture
        labeled_df[f"horizon_{h}sol"] = [
            any(
                storm_flags[j]
                for j, s2 in enumerate(sols)
                if sols[i] < s2 <= sols[i] + h
            )
            for i in range(n)
        ]

    aggregated_df = base.copy()  # "real" observations -- same rows here, all genuinely observed
    features_df = build_features(base, grid)
    return aggregated_df, labeled_df, features_df


def test_no_task_feature_manifest_contains_any_target_column(grid):
    sols = list(range(10))
    dust = [float(s) for s in sols]
    storm = [s % 3 == 0 for s in sols]
    aggregated_df, labeled_df, features_df = _make_labeled_and_features(grid, sols, dust, storm)

    _, fm_a = build_dust_regression_dataset(aggregated_df, labeled_df, features_df)
    _, fm_c = build_storm_detection_dataset(labeled_df, features_df)

    for fm in (fm_a, fm_c):
        assert not any(c.startswith("target_") for c in fm)


def test_task_feature_manifests_never_contain_another_tasks_target_name(grid):
    sols = list(range(10))
    dust = [float(s) for s in sols]
    storm = [s % 3 == 0 for s in sols]
    aggregated_df, labeled_df, features_df = _make_labeled_and_features(grid, sols, dust, storm)

    ds_a, fm_a = build_dust_regression_dataset(aggregated_df, labeled_df, features_df)
    ds_c, fm_c = build_storm_detection_dataset(labeled_df, features_df)

    all_target_prefixes = list(TASK_TARGET_PREFIXES.values())
    for fm in (fm_a, fm_c):
        for col in fm:
            assert not any(col.startswith(prefix) for prefix in all_target_prefixes)

    # each dataset's OWN target columns must never appear in another
    # dataset's feature manifest either (belt and braces on column names,
    # not just prefixes).
    a_targets = [c for c in ds_a.columns if c.startswith("target_")]
    c_targets = [c for c in ds_c.columns if c.startswith("target_")]
    assert not set(a_targets) & set(fm_c)
    assert not set(c_targets) & set(fm_a)


def test_target_column_names_are_mutually_exclusive_across_tasks():
    prefixes = list(TASK_TARGET_PREFIXES.values())
    assert len(prefixes) == len(set(prefixes))
    # no prefix is a substring-prefix of another (would allow accidental
    # startswith() cross-matches in downstream filtering code)
    for p1 in prefixes:
        for p2 in prefixes:
            if p1 != p2:
                assert not p1.startswith(p2)
                assert not p2.startswith(p1)


def test_dust_target_never_uses_a_trailing_filled_future_value(grid):
    # sol=5 exists ONLY in the (dense, filled-style) features_df, never in
    # the REAL aggregated_df -- simulating a gap-filled cell-sol. The
    # horizon-1 target looked up from sol=4 (real) should be NaN, not the
    # fabricated value that would exist at sol=5 in a filled series.
    sols_real = [0, 1, 2, 3, 4, 6, 7]  # sol 5 missing from REAL data
    dust_real = [10.0, 11.0, 12.0, 13.0, 14.0, 16.0, 17.0]
    storm = [False] * len(sols_real)
    aggregated_df, labeled_df, features_df = _make_labeled_and_features(
        grid, sols_real, dust_real, storm
    )
    # Simulate a filled row at sol=5 injected into features_df ONLY (as
    # apply_trailing_fill would produce), with a fabricated dust value.
    fabricated_row = features_df.iloc[[4]].copy()  # copy of sol=4's row
    fabricated_row["sol"] = 5
    fabricated_row["dust_optical_depth"] = 999.0  # obviously fabricated/carried-forward
    features_with_fill = pd.concat([features_df, fabricated_row], ignore_index=True)

    ds_a, _ = build_dust_regression_dataset(aggregated_df, labeled_df, features_with_fill)
    row_sol4 = ds_a.loc[ds_a["sol"] == 4].iloc[0]
    # target at sol=4, horizon=1 would look up sol=5 -- which is NOT in the
    # REAL aggregated_df -- so it must be NaN, never 999.0.
    assert np.isnan(row_sol4["target_dust_optical_depth_h1"])


def test_storm_detection_target_reuses_horizon_column_exactly(grid):
    sols = list(range(10))
    dust = [float(s) for s in sols]
    storm = [s == 5 for s in sols]  # single storm sol
    aggregated_df, labeled_df, features_df = _make_labeled_and_features(grid, sols, dust, storm)

    ds_c, _ = build_storm_detection_dataset(labeled_df, features_df)
    merged = ds_c.merge(labeled_df[["sol", "horizon_1sol"]], on="sol")
    assert (merged["target_storm_horizon_h1"] == merged["horizon_1sol"].astype(int)).all()
