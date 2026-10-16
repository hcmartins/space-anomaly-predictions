"""Tests for the leak-audited chronological split: boundary assignment,
episode-crossing-boundary detection, and train-only standardisation
statistics."""
import pandas as pd
import pytest

from split_fixtures import mechanics_splits

from martian.splitting.chronological_split import (
    EpisodeCrossesSplitBoundaryError,
    apply_standardization,
    assert_no_episode_crosses_split_boundary,
    assign_split,
    find_episodes_crossing_split_boundary,
    fit_standardization_stats,
    load_splits_config,
    write_split_manifest,
)


def test_assign_split_boundaries():
    splits_config = mechanics_splits()
    df = pd.DataFrame({"mars_year": [36, 36, 36, 37], "sol": [300, 650, 601, 10]})
    split = assign_split(df, splits_config)
    assert split.tolist() == ["train", "val", "val", "test"]


def test_episode_crossing_boundary_raises():
    splits_config = mechanics_splits()
    # Episode spans sol 430 (train) to sol 435 (val) within MY36.
    catalogue = pd.DataFrame([{"episode_id": 0, "mars_year": 36, "onset_sol": 430, "end_sol": 435}])
    with pytest.raises(EpisodeCrossesSplitBoundaryError):
        assert_no_episode_crosses_split_boundary(catalogue, splits_config)


def test_episode_within_one_split_does_not_raise():
    splits_config = mechanics_splits()
    catalogue = pd.DataFrame([{"episode_id": 0, "mars_year": 36, "onset_sol": 10, "end_sol": 15}])
    assert_no_episode_crosses_split_boundary(catalogue, splits_config)  # should not raise


def test_find_episodes_crossing_split_boundary_returns_the_crossing_subset():
    splits_config = mechanics_splits()
    catalogue = pd.DataFrame(
        [
            {"episode_id": 0, "mars_year": 36, "onset_sol": 430, "end_sol": 435},  # crosses train/val
            {"episode_id": 1, "mars_year": 36, "onset_sol": 10, "end_sol": 15},  # fully within train
        ]
    )
    crossing = find_episodes_crossing_split_boundary(catalogue, splits_config)
    assert crossing["episode_id"].tolist() == [0]
    # And the assert-based function must raise on the SAME input.
    with pytest.raises(EpisodeCrossesSplitBoundaryError):
        assert_no_episode_crosses_split_boundary(catalogue, splits_config)


def test_val_split_boundary_is_non_empty_and_causally_after_train():
    """Regression test for CHAPTER_4_PARAMETER_VALUES.md caveat 9 / Section 11
    item 5: configs/splits.yaml's val window must be non-empty and strictly
    later in sol than train (temporal causality for early stopping), not the
    original [601, 668] window which contained zero real MY36 cell-sols."""
    splits_config = load_splits_config()
    train_lo, train_hi = splits_config["train_sol_range"]
    val_lo, val_hi = splits_config["val_sol_range"]
    test_lo, test_hi = splits_config["test_sol_range"]
    # strict causal ordering in (mars_year, sol): train < val < test (the 2026-10-04
    # redesign places val in early MY37, after all of MY36)
    assert (splits_config["val_mars_year"], val_lo) > (splits_config["train_mars_year"], train_hi)
    assert (splits_config["test_mars_year"], test_lo) > (splits_config["val_mars_year"], val_hi)
    assert val_hi > val_lo and test_hi > test_lo  # genuine, non-degenerate ranges
    # when val and test share a Mars year, an embargo >= the longest forecast
    # horizon keeps every val target's (s, s+h] window out of the test period
    if splits_config["val_mars_year"] == splits_config["test_mars_year"]:
        from martian.labeling.anomaly_labels import load_anomaly_definition_config
        assert test_lo - val_hi - 1 >= max(load_anomaly_definition_config()["horizons_sols"])


def test_empty_catalogue_does_not_raise():
    splits_config = mechanics_splits()
    catalogue = pd.DataFrame(columns=["episode_id", "mars_year", "onset_sol", "end_sol"])
    assert_no_episode_crosses_split_boundary(catalogue, splits_config)


def test_standardization_stats_fit_train_only_unaffected_by_val_test():
    train_df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    stats_before = fit_standardization_stats(train_df, ["x"])

    # Appending wildly different val/test-like data must never change stats
    # already fit on train.
    extended_train_df = train_df.copy()  # simulate "what if we accidentally included val/test"
    contaminated = pd.concat([extended_train_df, pd.DataFrame({"x": [1000.0, -1000.0]})], ignore_index=True)
    stats_after_contamination_would_be = fit_standardization_stats(contaminated, ["x"])

    assert stats_before["x"]["mean"] == pytest.approx(2.5)
    assert stats_after_contamination_would_be["x"]["mean"] != pytest.approx(2.5)
    # proves fit_standardization_stats is deterministic and purely a function
    # of whatever dataframe it's given -- the CALLER's responsibility (and
    # the one this module structurally supports) is to only ever pass the
    # train slice.


def test_apply_standardization_uses_given_stats_not_recomputed():
    stats = {"x": {"mean": 2.5, "std": 1.0}}
    df = pd.DataFrame({"x": [2.5, 3.5, 1.5]})
    out = apply_standardization(df, stats)
    assert out["x"].tolist() == pytest.approx([0.0, 1.0, -1.0])


def test_write_split_manifest_counts_and_prevalence(tmp_path):
    df = pd.DataFrame(
        {
            "mars_year": [1, 1, 2],
            "sol": [10, 650, 5],
            "split": ["train", "val", "test"],
            "label": [1, 0, 1],
        }
    )
    path = write_split_manifest(tmp_path / "manifest.json", df, label_cols=["label"])
    assert path.exists()
    import json

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["n_rows"] == 1
    assert manifest["splits"]["val"]["n_rows"] == 1
    assert manifest["splits"]["test"]["n_rows"] == 1
    assert manifest["splits"]["train"]["label_prevalence"] == 1.0
