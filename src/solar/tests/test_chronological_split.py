"""Tests for the leak-audited chronological split: boundary assignment,
label-horizon-crossing-boundary detection/embargo, and train-only
standardization statistics."""
import pandas as pd
import pytest
import yaml

from splitting.chronological_split import (
    LabelHorizonCrossesSplitBoundaryError,
    apply_embargo,
    apply_standardization,
    assert_no_label_horizon_crosses_split_boundary,
    assign_split,
    find_rows_with_label_horizon_crossing_split_boundary,
    fit_standardization_stats,
    load_splits_config,
    write_split_manifest,
)

# A self-contained synthetic config, deliberately NOT read from the real
# configs/splits.yaml -- that file's real date boundaries are a live
# research choice (see planning/SPLITTING_DOCUMENTATION.md) and have
# already changed once as real data coverage was validated; tests must not
# be coupled to whatever its current values happen to be.
_TEST_SPLITS_CONFIG = {
    "train_start": "2010-05-01", "train_end": "2015-12-31",
    "val_start": "2016-01-01", "val_end": "2017-12-31",
    "test_start": "2018-01-01", "test_end": "2019-12-31",
    "embargo_hours": 24,
}


def test_assign_split_boundaries():
    config = _TEST_SPLITS_CONFIG
    df = pd.DataFrame(
        {
            "time_utc": [
                "2012-06-15T00:00:00Z",   # well within train
                "2015-12-31T23:59:00Z",   # last minute of train (inclusive)
                "2016-01-01T00:00:00Z",   # first minute of val
                "2018-06-01T00:00:00Z",   # well within test
                "2009-01-01T00:00:00Z",   # before train_start
                "2025-01-01T00:00:00Z",   # after test_end
            ]
        }
    )
    split = assign_split(df, config)
    assert split.tolist() == ["train", "train", "val", "test", "unassigned", "unassigned"]


def test_label_horizon_crossing_boundary_is_found():
    config = _TEST_SPLITS_CONFIG
    df = pd.DataFrame({"time_utc": ["2015-12-31T12:00:00Z"]})  # train, 12h before val starts
    df["split"] = assign_split(df, config)
    crossing = find_rows_with_label_horizon_crossing_split_boundary(df, config, horizon_hours=24)
    assert len(crossing) == 1

    with pytest.raises(LabelHorizonCrossesSplitBoundaryError):
        assert_no_label_horizon_crosses_split_boundary(df, config, horizon_hours=24)


def test_row_with_horizon_inside_its_own_split_does_not_cross():
    config = _TEST_SPLITS_CONFIG
    df = pd.DataFrame({"time_utc": ["2012-06-01T00:00:00Z"]})  # far from any boundary
    df["split"] = assign_split(df, config)
    crossing = find_rows_with_label_horizon_crossing_split_boundary(df, config, horizon_hours=24)
    assert len(crossing) == 0
    assert_no_label_horizon_crosses_split_boundary(df, config, horizon_hours=24)  # must not raise


def test_apply_embargo_reassigns_crossing_rows_and_clears_the_error():
    config = _TEST_SPLITS_CONFIG
    df = pd.DataFrame(
        {"time_utc": ["2015-12-31T12:00:00Z", "2012-06-01T00:00:00Z"]}
    )  # row 0 crosses into val, row 1 does not
    df["split"] = assign_split(df, config)
    df["split"] = apply_embargo(df, config, horizon_hours=24)

    assert df.iloc[0]["split"] == "embargo"
    assert df.iloc[1]["split"] == "train"
    assert_no_label_horizon_crosses_split_boundary(df, config, horizon_hours=24)  # must not raise now


def test_invalid_splits_config_raises_on_overlapping_ranges(tmp_path):
    bad_config = {
        "train_start": "2010-01-01", "train_end": "2016-06-01",
        "val_start": "2016-01-01", "val_end": "2017-12-31",  # overlaps train
        "test_start": "2018-01-01", "test_end": "2019-12-31",
        "embargo_hours": 24,
    }
    path = tmp_path / "bad_splits.yaml"
    path.write_text(yaml.safe_dump(bad_config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_splits_config(path)


def test_standardization_stats_fit_train_only_unaffected_by_val_test():
    train_df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    stats_before = fit_standardization_stats(train_df, ["x"])

    contaminated = pd.concat([train_df, pd.DataFrame({"x": [1000.0, -1000.0]})], ignore_index=True)
    stats_after_contamination_would_be = fit_standardization_stats(contaminated, ["x"])

    assert stats_before["x"]["mean"] == pytest.approx(2.5)
    assert stats_after_contamination_would_be["x"]["mean"] != pytest.approx(2.5)


def test_apply_standardization_uses_given_stats_not_recomputed():
    stats = {"x": {"mean": 2.5, "std": 1.0}}
    df = pd.DataFrame({"x": [2.5, 3.5, 1.5]})
    out = apply_standardization(df, stats)
    assert out["x"].tolist() == pytest.approx([0.0, 1.0, -1.0])


def test_write_split_manifest_counts_and_prevalence(tmp_path):
    df = pd.DataFrame(
        {
            "time_utc": ["2012-01-01T00:00:00Z", "2016-06-01T00:00:00Z", "2018-06-01T00:00:00Z"],
            "split": ["train", "val", "test"],
            "label_flare_m_or_above": [1.0, 0.0, float("nan")],
        }
    )
    path = write_split_manifest(tmp_path / "manifest.json", df, label_cols=["label_flare_m_or_above"])
    assert path.exists()
    import json

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["n_rows"] == 1
    assert manifest["splits"]["val"]["n_rows"] == 1
    assert manifest["splits"]["test"]["n_rows"] == 1
    assert manifest["splits"]["train"]["label_flare_m_or_above_prevalence"] == 1.0
    assert manifest["splits"]["test"]["label_flare_m_or_above_prevalence"] is None  # only NaN in this split
