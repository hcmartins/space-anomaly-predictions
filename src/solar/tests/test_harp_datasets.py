"""Tests for the per-HARP features+labels join."""
import pandas as pd
import pytest

from modeling.datasets import assign_split_with_embargo
from modeling.harp_datasets import build_harp_flare_forecast_dataset


def _labels_with_full_metadata(rows):
    # Includes BOTH harp_catalogue_labels.py metadata columns
    # (label_horizon_hours, label_max_flare_class_in_harp_horizon) --
    # real bug regression: these both match the label_ prefix AND are
    # separately listed in HARP_LABEL_METADATA_COLUMNS, which previously
    # caused them to be selected twice into the merge's right side,
    # producing duplicate-named columns in the output (dataset[col]
    # silently returns a DataFrame instead of a Series) -- caught for
    # real by assign_split_with_embargo crashing with "'DataFrame' object
    # has no attribute 'unique'" on a genuine full-scale run, not by any
    # exception in build_harp_flare_forecast_dataset itself.
    for row in rows:
        row.setdefault("label_max_flare_class_in_harp_horizon", "M")
        row.setdefault("label_horizon_hours", 24.0)
    return pd.DataFrame(rows)


def test_join_keys_on_harpnum_and_time_not_time_alone():
    # Same time_utc, different HARPNUM -- must not cross-join or collide.
    features = pd.DataFrame([
        {"HARPNUM": 7115, "time_utc": pd.Timestamp("2017-09-04T00:00:00Z"), "feat_x": 1.0},
        {"HARPNUM": 9001, "time_utc": pd.Timestamp("2017-09-04T00:00:00Z"), "feat_x": 2.0},
    ])
    labels = _labels_with_full_metadata([
        {"HARPNUM": 7115, "time_utc": pd.Timestamp("2017-09-04T00:00:00Z"), "label_primary": 1.0,
         "label_flare_m_or_above": 1.0},
        {"HARPNUM": 9001, "time_utc": pd.Timestamp("2017-09-04T00:00:00Z"), "label_primary": 0.0,
         "label_flare_m_or_above": 0.0},
    ])
    dataset, feature_manifest, target_cols = build_harp_flare_forecast_dataset(features, labels)

    assert len(dataset) == 2
    assert feature_manifest == ["feat_x"]
    assert not dataset.columns.duplicated().any(), f"duplicate columns: {dataset.columns[dataset.columns.duplicated()]}"
    row_7115 = dataset.loc[dataset["HARPNUM"] == 7115].iloc[0]
    row_9001 = dataset.loc[dataset["HARPNUM"] == 9001].iloc[0]
    assert row_7115["label_primary"] == 1.0
    assert row_9001["label_primary"] == 0.0

    # The real integration point that caught this bug: assign_split_with_embargo
    # reads dataset["label_horizon_hours"] expecting a Series, not a DataFrame.
    out = assign_split_with_embargo(dataset, {
        "train_start": "2017-01-01", "train_end": "2017-12-31",
        "val_start": "2018-01-01", "val_end": "2018-12-31",
        "test_start": "2019-01-01", "test_end": "2019-12-31", "embargo_hours": 24,
    })
    assert "split" in out.columns


def test_raises_on_missing_required_column():
    features = pd.DataFrame([{"time_utc": pd.Timestamp("2017-09-04T00:00:00Z"), "feat_x": 1.0}])
    labels = pd.DataFrame([{"HARPNUM": 1, "time_utc": pd.Timestamp("2017-09-04T00:00:00Z")}])
    with pytest.raises(ValueError):
        build_harp_flare_forecast_dataset(features, labels)


def test_raises_on_feature_label_column_collision():
    features = pd.DataFrame([
        {"HARPNUM": 1, "time_utc": pd.Timestamp("2017-09-04T00:00:00Z"), "label_primary": 999.0}
    ])
    labels = pd.DataFrame([
        {"HARPNUM": 1, "time_utc": pd.Timestamp("2017-09-04T00:00:00Z"), "label_primary": 1.0}
    ])
    with pytest.raises(ValueError):
        build_harp_flare_forecast_dataset(features, labels)
