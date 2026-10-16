"""Tests for step 12: runtime leakage checks against THIS run's actual
data -- complementary to the static pytest leakage suite
(tests/test_datasets_no_leakage.py, tests/test_chronological_split.py)."""
import pandas as pd
import pytest

from split_fixtures import mechanics_splits

from martian.mars.leakage_checks import LeakageCheckFailed, run_leakage_checks
from martian.splitting.chronological_split import load_splits_config


def _clean_task_datasets() -> dict:
    dust_df = pd.DataFrame({"cell_id": [0], "target_dust_optical_depth_h1": [0.2]})
    storm_df = pd.DataFrame({"cell_id": [0], "target_storm_horizon_h1": [1]})
    return {
        "dust_regression": (dust_df, ["feat_x1"]),
        "storm_detection": (storm_df, ["feat_x3"]),
    }


def test_clean_run_passes_leakage_checks():
    splits_config = mechanics_splits()
    empty_catalogue = pd.DataFrame(columns=["episode_id", "mars_year", "onset_sol", "end_sol"])
    summary = run_leakage_checks(
        episode_catalogue=empty_catalogue, splits_config=splits_config, task_datasets=_clean_task_datasets()
    )
    assert summary["episode_boundary_check"] == "passed"
    assert summary["feature_manifest_target_leakage_check"] == "passed"
    assert set(summary["tasks_checked"]) == {"dust_regression", "storm_detection"}


def test_episode_crossing_boundary_fails_the_run():
    splits_config = mechanics_splits()
    crossing_catalogue = pd.DataFrame(
        [{"episode_id": 0, "mars_year": 36, "onset_sol": 430, "end_sol": 435}]
    )
    with pytest.raises(LeakageCheckFailed, match="boundary"):
        run_leakage_checks(
            episode_catalogue=crossing_catalogue, splits_config=splits_config, task_datasets=_clean_task_datasets()
        )


def test_target_leaked_into_own_feature_manifest_fails():
    splits_config = mechanics_splits()
    empty_catalogue = pd.DataFrame(columns=["episode_id", "mars_year", "onset_sol", "end_sol"])
    datasets = _clean_task_datasets()
    dust_df, dust_features = datasets["dust_regression"]
    datasets["dust_regression"] = (dust_df, dust_features + ["target_dust_optical_depth_h1"])
    with pytest.raises(LeakageCheckFailed, match="target column"):
        run_leakage_checks(episode_catalogue=empty_catalogue, splits_config=splits_config, task_datasets=datasets)


def test_cross_task_target_leakage_fails():
    # Task A's target column name accidentally ends up in Task B's feature manifest.
    splits_config = mechanics_splits()
    empty_catalogue = pd.DataFrame(columns=["episode_id", "mars_year", "onset_sol", "end_sol"])
    datasets = _clean_task_datasets()
    storm_df, storm_features = datasets["storm_detection"]
    datasets["storm_detection"] = (storm_df, storm_features + ["target_dust_optical_depth_h1"])
    with pytest.raises(LeakageCheckFailed, match="storm_detection"):
        run_leakage_checks(episode_catalogue=empty_catalogue, splits_config=splits_config, task_datasets=datasets)


def test_multiple_problems_all_reported_together():
    splits_config = mechanics_splits()
    crossing_catalogue = pd.DataFrame(
        [{"episode_id": 0, "mars_year": 36, "onset_sol": 430, "end_sol": 435}]
    )
    datasets = _clean_task_datasets()
    dust_df, dust_features = datasets["dust_regression"]
    datasets["dust_regression"] = (dust_df, dust_features + ["target_dust_optical_depth_h1"])
    with pytest.raises(LeakageCheckFailed) as exc_info:
        run_leakage_checks(episode_catalogue=crossing_catalogue, splits_config=splits_config, task_datasets=datasets)
    message = str(exc_info.value)
    assert "boundary" in message
    assert "target column" in message
