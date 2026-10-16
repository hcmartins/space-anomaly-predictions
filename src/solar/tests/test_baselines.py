"""Tests for the baseline classifiers: climatology (train-mean), persistence
-- both dataframe-aware, config-driven, and fit-on-train-only."""
import numpy as np
import pandas as pd
import pytest

from modeling.baselines import BASELINE_FACTORIES, ClimatologyBaseline, PersistenceBaseline, build_baseline


def test_persistence_baseline_predicts_current_value():
    model = PersistenceBaseline(source_column="feat_persistence_elevated_flux_run_steps")
    train_df = pd.DataFrame({"feat_persistence_elevated_flux_run_steps": [0, 3]})
    model.fit(train_df, [0.0, 1.0])
    apply_df = pd.DataFrame({"feat_persistence_elevated_flux_run_steps": [5, 6]})
    assert model.predict(apply_df).tolist() == [5.0, 6.0]


def test_persistence_baseline_predict_proba_clips_to_valid_range():
    model = PersistenceBaseline(source_column="feat_persistence_elevated_flux_run_steps")
    apply_df = pd.DataFrame({"feat_persistence_elevated_flux_run_steps": [0, 5]})
    proba = model.predict_proba(apply_df)
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0].tolist() == pytest.approx([1.0, 0.0])
    assert proba[1].tolist() == pytest.approx([0.0, 1.0])  # clipped: any run >0 -> full probability


def test_climatology_baseline_train_mean_mode():
    model = ClimatologyBaseline(mode="train_mean")
    model.fit(pd.DataFrame(), np.array([0, 1, 1, 0, 1]))  # rate = 0.6
    apply_df = pd.DataFrame({"anything": [1, 2]})
    preds = model.predict(apply_df)
    assert preds.tolist() == pytest.approx([0.6, 0.6])


def test_climatology_baseline_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ClimatologyBaseline(mode="not_a_real_mode")


def test_build_baseline_factory_covers_both():
    assert set(BASELINE_FACTORIES.keys()) == {"climatology", "persistence"}
    model = build_baseline("persistence", {"source_column": "feat_persistence_elevated_flux_run_steps"})
    assert isinstance(model, PersistenceBaseline)


def test_build_baseline_unknown_name_raises():
    with pytest.raises(ValueError):
        build_baseline("not_a_baseline", {})


def test_baselines_are_config_driven_not_hardcoded():
    # Changing the configured source_column changes what's actually read --
    # proves the parameter is used at runtime, not a hardcoded column name.
    model_a = PersistenceBaseline(source_column="col_a")
    model_b = PersistenceBaseline(source_column="col_b")
    df = pd.DataFrame({"col_a": [1.0], "col_b": [0.0]})
    assert model_a.predict(df).tolist() == [1.0]
    assert model_b.predict(df).tolist() == [0.0]
