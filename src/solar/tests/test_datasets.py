"""Tests for the features/labels join (the "missing link" between
preprocessing/features.py and labeling/flare_labels.py)."""
import numpy as np
import pandas as pd
import pytest

from labeling.flare_labels import build_primary_label, load_labeling_config
from modeling.datasets import (
    LABEL_METADATA_COLUMNS,
    TARGET_COLUMNS,
    assign_split_with_embargo,
    build_flare_forecast_dataset,
    drop_rows_with_undefined_label,
    summarize_targets,
)
from preprocessing.alignment import build_aligned_timeline
from preprocessing.features import build_features
from preprocessing.missing_data import apply_trailing_fill

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


def _build_pipeline_dataset():
    # SHARP reports at 00:00, 00:12, and 00:36 -- missing 00:24 (a 1-step,
    # fillable gap) so missing_data.py introduces one interpolated row that
    # has no corresponding label (labeling always runs on the pre-fill
    # aligned series).
    sharp_df = pd.DataFrame(
        [
            {"HARPNUM": 401, "T_REC": "2018.01.01_00:00:00_TAI", "USFLUX": 1.0e21, "R_VALUE": 2.0},
            {"HARPNUM": 401, "T_REC": "2018.01.01_00:12:00_TAI", "USFLUX": 1.1e21, "R_VALUE": 2.1},
            {"HARPNUM": 401, "T_REC": "2018.01.01_00:36:00_TAI", "USFLUX": 1.2e21, "R_VALUE": 2.2},
        ]
    )
    goes_df = pd.DataFrame(
        [
            {"time_utc": "2018-01-01T00:00:00Z", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 1e-7},
            {"time_utc": "2018-01-01T00:30:00Z", "satellite": 16, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 1e-7},
        ]
    )
    aligned, _sparse_goes_disk = build_aligned_timeline(goes_df, sharp_df)

    filled = apply_trailing_fill(aligned)
    assert len(filled) == 4  # 3 real + 1 interpolated (00:24)
    features_df = build_features(filled)

    # A dense 1-min GOES series, standing in for what a real ingestion run
    # would provide, covering a 1h horizon after every SHARP timestamp with
    # an M-class spike at 01:00.
    fine_times = pd.date_range("2018-01-01T00:00:00Z", "2018-01-01T02:00:00Z", freq="1min", tz="UTC")
    fine_values = np.full(len(fine_times), 1e-8)
    fine_values[list(fine_times).index(pd.Timestamp("2018-01-01T01:00:00Z"))] = 2.0e-5  # M class
    goes_fine = pd.DataFrame({"time_utc": fine_times, "xrsb_flux_w_m2": fine_values})

    label_config = load_labeling_config()
    label_config["forecast_horizon_hours"] = 1
    labeled_df = build_primary_label(aligned, goes_fine, config=label_config)

    return features_df, labeled_df


def test_join_restricts_to_genuinely_observed_forecast_origins():
    features_df, labeled_df = _build_pipeline_dataset()
    dataset, feature_manifest, target_cols = build_flare_forecast_dataset(features_df, labeled_df)

    assert len(dataset) == 3  # the interpolated 00:24 row has no label and is dropped
    assert dataset["is_observed"].all()
    assert pd.Timestamp("2018-01-01T00:24:00Z") not in set(dataset["time_utc"])


def test_feature_manifest_and_target_columns():
    features_df, labeled_df = _build_pipeline_dataset()
    dataset, feature_manifest, target_cols = build_flare_forecast_dataset(features_df, labeled_df)

    assert any(c.startswith("feat_") for c in feature_manifest)
    assert "USFLUX" in feature_manifest  # raw current-state column, causal, legitimate feature
    assert set(target_cols) == set(TARGET_COLUMNS)
    for col in LABEL_METADATA_COLUMNS:
        assert col in dataset.columns
    # M-class spike is within every row's 1h horizon.
    assert (dataset["label_flare_m_or_above"] == 1.0).all()
    assert (dataset["label_flare_x_or_above"] == 0.0).all()


def test_overlapping_label_column_raises():
    features_df, labeled_df = _build_pipeline_dataset()
    features_df = features_df.copy()
    features_df["label_flare_m_or_above"] = 0.0  # simulates a caller bug
    with pytest.raises(ValueError):
        build_flare_forecast_dataset(features_df, labeled_df)


def test_drop_rows_with_undefined_label():
    dataset = pd.DataFrame({"time_utc": pd.date_range("2018-01-01", periods=3, freq="D", tz="UTC"),
                             "label_primary": [1.0, np.nan, 0.0]})
    out = drop_rows_with_undefined_label(dataset, "label_primary")
    assert len(out) == 2
    assert out["label_primary"].notna().all()


def test_drop_rows_missing_target_column_raises_keyerror():
    dataset = pd.DataFrame({"time_utc": ["2018-01-01"]})
    with pytest.raises(KeyError):
        drop_rows_with_undefined_label(dataset, "label_primary")


def test_summarize_targets_reports_counts_and_positive_rate():
    dataset = pd.DataFrame(
        {
            "label_flare_m_or_above": [1.0, 0.0, np.nan],
            "label_flare_x_or_above": [np.nan, np.nan, np.nan],  # entirely undefined
        }
    )
    summary = summarize_targets(dataset, ["label_flare_m_or_above", "label_flare_x_or_above"])
    m_row = summary[summary["target"] == "label_flare_m_or_above"].iloc[0]
    assert m_row["n_defined"] == 2
    assert m_row["n_positive"] == 1
    assert m_row["positive_rate"] == pytest.approx(0.5)

    x_row = summary[summary["target"] == "label_flare_x_or_above"].iloc[0]
    assert x_row["n_defined"] == 0
    assert np.isnan(x_row["positive_rate"])


def test_assign_split_with_embargo_infers_horizon_from_dataset():
    config = _TEST_SPLITS_CONFIG
    # horizon=48h (NOT configs/splits.yaml's default embargo_hours=24) --
    # only crosses the train/val boundary if the function actually reads
    # this dataset's own label_horizon_hours rather than the config default.
    dataset = pd.DataFrame(
        {
            "time_utc": ["2015-12-30T00:00:00Z", "2012-06-01T00:00:00Z"],
            "label_horizon_hours": [48.0, 48.0],
        }
    )
    out = assign_split_with_embargo(dataset, config)
    assert out.iloc[0]["split"] == "embargo"  # (t, t+48h] reaches into val
    assert out.iloc[1]["split"] == "train"    # far from any boundary


def test_assign_split_with_embargo_missing_horizon_column_raises():
    dataset = pd.DataFrame({"time_utc": ["2012-06-01T00:00:00Z"]})
    with pytest.raises(ValueError):
        assign_split_with_embargo(dataset)


def test_assign_split_with_embargo_inconsistent_horizon_raises():
    dataset = pd.DataFrame(
        {
            "time_utc": ["2012-06-01T00:00:00Z", "2012-06-02T00:00:00Z"],
            "label_horizon_hours": [24.0, 48.0],
        }
    )
    with pytest.raises(ValueError):
        assign_split_with_embargo(dataset)


def test_assign_split_with_embargo_explicit_horizon_overrides_column():
    config = _TEST_SPLITS_CONFIG
    dataset = pd.DataFrame(
        {"time_utc": ["2015-12-30T00:00:00Z"], "label_horizon_hours": [1.0]}  # would NOT cross on its own
    )
    out = assign_split_with_embargo(dataset, config, horizon_hours=48)
    assert out.iloc[0]["split"] == "embargo"


def test_assign_split_with_embargo_end_to_end_from_built_dataset():
    features_df, labeled_df = _build_pipeline_dataset()
    dataset, _feature_manifest, _target_cols = build_flare_forecast_dataset(features_df, labeled_df)
    out = assign_split_with_embargo(dataset, _TEST_SPLITS_CONFIG)
    assert "split" in out.columns
    # All rows are 2018-01-01 (inside _TEST_SPLITS_CONFIG's "test" range)
    # with only a 1h horizon -- nowhere near test_end, so none embargoed.
    assert set(out["split"]) == {"test"}
