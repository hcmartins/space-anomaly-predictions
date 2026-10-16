"""Tests for the baseline hierarchy: climatology, persistence, seasonal
mean -- all dataframe-aware, config-driven, and fit-on-train-only."""
import numpy as np
import pandas as pd
import pytest

from martian.modeling.baselines import (
    BASELINE_FACTORIES,
    ClimatologyBaseline,
    PersistenceBaseline,
    SeasonalMeanBaseline,
    build_baseline,
)


def test_persistence_baseline_predicts_current_value():
    model = PersistenceBaseline(source_column="dust_optical_depth")
    train_df = pd.DataFrame({"dust_optical_depth": [1.0, 2.0]})
    model.fit(train_df, [1.0, 2.0])
    apply_df = pd.DataFrame({"dust_optical_depth": [5.0, 6.0]})
    assert model.predict(apply_df).tolist() == [5.0, 6.0]


def test_persistence_baseline_predict_proba_clips_to_valid_range():
    model = PersistenceBaseline(source_column="atmospheric_anomaly")
    apply_df = pd.DataFrame({"atmospheric_anomaly": [0, 1]})
    proba = model.predict_proba(apply_df)
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_seasonal_mean_baseline_fit_train_only():
    train_df = pd.DataFrame({"solar_longitude_ls": [10.0, 20.0, 190.0, 200.0]})
    y_train = np.array([1.0, 3.0, 100.0, 200.0])  # spring mean=2, autumn mean=150
    model = SeasonalMeanBaseline()
    model.fit(train_df, y_train)

    apply_df = pd.DataFrame({"solar_longitude_ls": [15.0, 195.0]})
    preds = model.predict(apply_df)
    assert preds[0] == pytest.approx(2.0)
    assert preds[1] == pytest.approx(150.0)


def test_seasonal_mean_baseline_unseen_season_falls_back_to_global_mean():
    train_df = pd.DataFrame({"solar_longitude_ls": [10.0, 20.0]})  # only spring seen
    y_train = np.array([4.0, 6.0])
    model = SeasonalMeanBaseline()
    model.fit(train_df, y_train)
    apply_df = pd.DataFrame({"solar_longitude_ls": [280.0]})  # winter, unseen in train
    preds = model.predict(apply_df)
    assert preds[0] == pytest.approx(5.0)  # global train mean


def test_climatology_baseline_column_mode():
    model = ClimatologyBaseline(mode="column", column="dust_optical_depth_mu")
    model.fit(pd.DataFrame(), [])
    apply_df = pd.DataFrame({"dust_optical_depth_mu": [0.15, 0.22]})
    assert model.predict(apply_df).tolist() == pytest.approx([0.15, 0.22])


def test_climatology_baseline_zero_mode():
    model = ClimatologyBaseline(mode="zero")
    model.fit(pd.DataFrame(), [1.0, 2.0])
    apply_df = pd.DataFrame({"anything": [1, 2, 3]})
    assert model.predict(apply_df).tolist() == [0.0, 0.0, 0.0]


def test_climatology_baseline_train_mean_mode():
    model = ClimatologyBaseline(mode="train_mean")
    model.fit(pd.DataFrame(), np.array([0, 1, 1, 0, 1]))  # rate = 0.6
    apply_df = pd.DataFrame({"anything": [1, 2]})
    preds = model.predict(apply_df)
    assert preds.tolist() == pytest.approx([0.6, 0.6])


def test_climatology_baseline_column_mode_requires_column_name():
    with pytest.raises(ValueError):
        ClimatologyBaseline(mode="column")


def test_climatology_baseline_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ClimatologyBaseline(mode="not_a_real_mode")


def test_build_baseline_factory_covers_all_three():
    assert set(BASELINE_FACTORIES.keys()) == {"climatology", "persistence", "seasonal_mean"}
    model = build_baseline("persistence", {"source_column": "dust_optical_depth"})
    assert isinstance(model, PersistenceBaseline)


def test_build_baseline_unknown_name_raises():
    with pytest.raises(ValueError):
        build_baseline("not_a_baseline", {})


def test_baselines_are_config_driven_not_hardcoded():
    # Changing the configured source_column changes what's actually read --
    # proves the parameter is used at runtime, not a hardcoded column name.
    model_a = PersistenceBaseline(source_column="dust_optical_depth")
    model_b = PersistenceBaseline(source_column="surface_temperature_k")
    df = pd.DataFrame({"dust_optical_depth": [1.0], "surface_temperature_k": [200.0]})
    assert model_a.predict(df).tolist() == [1.0]
    assert model_b.predict(df).tolist() == [200.0]
