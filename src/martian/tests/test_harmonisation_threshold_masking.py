"""Tests for the 2026-10-04 pipeline changes: per-retrieval dust harmonisation,
Eq. 6.1 validation threshold selection, and unobserved-horizon masking."""
import numpy as np
import pandas as pd
import pytest

from martian.labeling.anomaly_labels import compute_horizon_labels
from martian.modeling.metrics import classification_metrics, select_threshold
from martian.preprocessing.harmonisation import harmonise_dust_to_reference, load_harmonisation_config


def _cfg(**over):
    cfg = load_harmonisation_config()
    cfg.update(over)
    return cfg


def test_harmonisation_formula_exact():
    df = pd.DataFrame({"dust_optical_depth": [0.39, 0.39], "dust_optical_depth_uncertainty": [0.013, 0.013],
                       "surface_pressure_mbar": [6.1, 3.05]})
    out, summary = harmonise_dust_to_reference(df, _cfg(extinction_to_absorption_ratio=1.3))
    # at p_s = 610 Pa the pressure factor is 1 -> tau* = tau/1.3; at half pressure it doubles
    assert out["dust_optical_depth"].tolist() == pytest.approx([0.3, 0.6])
    assert out["dust_optical_depth_uncertainty"].tolist() == pytest.approx([0.01, 0.02])
    assert summary["enabled"] and summary["n_retrievals_harmonised"] == 2


def test_harmonisation_preserves_relative_uncertainty_and_input():
    df = pd.DataFrame({"dust_optical_depth": [0.2, 0.5], "dust_optical_depth_uncertainty": [0.02, 0.1],
                       "surface_pressure_mbar": [4.0, 9.0]})
    before = df.copy()
    out, _ = harmonise_dust_to_reference(df, _cfg())
    pd.testing.assert_frame_equal(df, before)  # input never mutated
    rel_in = df["dust_optical_depth_uncertainty"] / df["dust_optical_depth"]
    rel_out = out["dust_optical_depth_uncertainty"] / out["dust_optical_depth"]
    assert rel_out.tolist() == pytest.approx(rel_in.tolist())  # QC rule 6 unaffected


def test_harmonisation_refuses_missing_or_nonpositive_pressure():
    df = pd.DataFrame({"dust_optical_depth": [0.2, 0.3], "dust_optical_depth_uncertainty": [0.01, 0.01],
                       "surface_pressure_mbar": [5.0, 0.0]})
    with pytest.raises(ValueError):
        harmonise_dust_to_reference(df, _cfg())
    with pytest.raises(KeyError):
        harmonise_dust_to_reference(df.drop(columns=["surface_pressure_mbar"]), _cfg())


def test_harmonisation_disabled_is_identity():
    df = pd.DataFrame({"dust_optical_depth": [0.2], "dust_optical_depth_uncertainty": [0.01], "surface_pressure_mbar": [5.0]})
    out, summary = harmonise_dust_to_reference(df, _cfg(enabled=False))
    pd.testing.assert_frame_equal(out, df)
    assert summary == {"enabled": False}


def test_select_threshold_maximises_validation_tss():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.full(90, 0.2), np.full(10, 0.35)])  # positives at 0.35: 0.5 misses all of them
    t, src = select_threshold(y, p)
    assert 0.2 < t <= 0.35 and src.startswith("validation_selected")
    assert classification_metrics(y, p, threshold=t)["tss"] == pytest.approx(1.0)
    assert classification_metrics(y, p, threshold=0.5)["tss"] == pytest.approx(0.0)


def test_select_threshold_single_class_returns_none():
    t, reason = select_threshold(np.zeros(10, int), np.linspace(0, 1, 10))
    assert t is None and "single class" in reason


def test_unobserved_horizon_is_nan_not_negative():
    # cell observed on sols 0, 1, 2, 6 only; storm (episode member) on sol 6
    df = pd.DataFrame({"cell_id": [1] * 4, "mars_year": [36] * 4, "sol": [0, 1, 2, 6],
                       "storm_episode": [False, False, False, True]})
    out = compute_horizon_labels(df, horizons_sols=(1, 3, 5)).set_index("sol")
    assert out.loc[0, "horizon_1sol"] == 0.0           # sol 1 observed, no storm
    assert np.isnan(out.loc[2, "horizon_1sol"])          # sol 3 unobserved -> unknown, not 0
    assert np.isnan(out.loc[2, "horizon_3sol"])          # sols 3-5 all unobserved
    assert out.loc[2, "horizon_5sol"] == 1.0             # sol 6 observed and in an episode
    assert np.isnan(out.loc[6, "horizon_5sol"])          # end of record: nothing observed after sol 6
