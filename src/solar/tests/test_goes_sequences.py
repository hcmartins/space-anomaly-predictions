"""Tests for GOES sequence-tensor construction (Chapter 3.2.1's X_G(t0))."""
import numpy as np
import pandas as pd

from preprocessing.goes_sequences import (
    assess_goes_sequence_feasibility,
    build_goes_sequences,
    window_length_steps,
)


def _sequences_config(**overrides):
    config = {
        "feature_columns": ["goes_seq_xrsb_flux_w_m2", "goes_seq_xrsa_flux_w_m2"],
        "look_back_hours": 6,
    }
    config.update(overrides)
    return config


def _preprocessing_config(**overrides):
    config = {"resample_cadence_minutes": 12}
    config.update(overrides)
    return config


def _clean_grid(n_steps=40, start="2018-01-01T00:00:00Z"):
    times = pd.date_range(start, periods=n_steps, freq="12min", tz="UTC")
    return pd.DataFrame(
        {
            "time_utc": times,
            "goes_seq_xrsb_flux_w_m2": np.arange(n_steps, dtype=float),
            "goes_seq_xrsa_flux_w_m2": np.arange(n_steps, dtype=float) * 10.0,
            "goes_seq_interpolated_any": [False] * n_steps,
            "goes_seq_spike_removed_any": [False] * n_steps,
            "goes_seq_gap_minutes": [0.0] * n_steps,
        }
    )


def test_window_length_steps_hand_computed():
    assert window_length_steps(_sequences_config(look_back_hours=6), _preprocessing_config()) == 30
    assert window_length_steps(_sequences_config(look_back_hours=1), _preprocessing_config()) == 5


def test_build_goes_sequences_feasible_window_values_match_source_grid():
    grid = _clean_grid(n_steps=40)
    seq_config = _sequences_config(look_back_hours=1)   # window_steps=5
    issuance_times = [grid["time_utc"].iloc[20]]   # plenty of history before it
    X, meta = build_goes_sequences(grid, issuance_times, sequences_config=seq_config, preprocessing_config=_preprocessing_config())

    assert meta["feasible"].iloc[0]
    assert X.shape == (1, 5, 2)
    # Steps are the 5 most recent grid rows ending at (and including) index 20: 16..20.
    expected_xrsb = grid["goes_seq_xrsb_flux_w_m2"].iloc[16:21].to_numpy()
    np.testing.assert_array_equal(X[0, :, 0], expected_xrsb)


def test_build_goes_sequences_discards_window_with_flagged_gap_inside():
    grid = _clean_grid(n_steps=40)
    grid.loc[18, "goes_seq_interpolated_any"] = True   # falls inside the 5-step window ending at 20
    seq_config = _sequences_config(look_back_hours=1)
    issuance_times = [grid["time_utc"].iloc[20]]
    X, meta = build_goes_sequences(grid, issuance_times, sequences_config=seq_config, preprocessing_config=_preprocessing_config())
    assert not meta["feasible"].iloc[0]
    assert X.shape == (0, 5, 2)


def test_build_goes_sequences_discards_window_with_insufficient_history():
    grid = _clean_grid(n_steps=40)
    seq_config = _sequences_config(look_back_hours=6)   # window_steps=30, needs 29 steps of history
    issuance_times = [grid["time_utc"].iloc[5]]   # only 5 prior steps exist
    X, meta = build_goes_sequences(grid, issuance_times, sequences_config=seq_config, preprocessing_config=_preprocessing_config())
    assert not meta["feasible"].iloc[0]


def test_build_goes_sequences_reports_every_issuance_time_never_silently_drops():
    grid = _clean_grid(n_steps=40)
    seq_config = _sequences_config(look_back_hours=1)
    issuance_times = [grid["time_utc"].iloc[2], grid["time_utc"].iloc[20], grid["time_utc"].iloc[30]]
    X, meta = build_goes_sequences(grid, issuance_times, sequences_config=seq_config, preprocessing_config=_preprocessing_config())
    assert len(meta) == 3
    assert list(meta["feasible"]) == [False, True, True]
    assert X.shape[0] == 2


def test_assess_goes_sequence_feasibility_hand_computed():
    # A single clean run of length 10; window_steps=3 -> 10-3+1=8 feasible windows.
    grid = _clean_grid(n_steps=10)
    result = assess_goes_sequence_feasibility(grid, 3, sequences_config=_sequences_config())
    assert result["n_clean_runs"] == 1
    assert result["max_run_length_steps"] == 10
    assert result["n_feasible_windows"] == 8
    assert result["feasible"] is True


def test_assess_goes_sequence_feasibility_with_a_gap_splits_the_run():
    grid = _clean_grid(n_steps=10)
    grid.loc[5, "goes_seq_interpolated_any"] = True   # splits into runs of length 5 and 4
    result = assess_goes_sequence_feasibility(grid, 3, sequences_config=_sequences_config())
    assert result["n_clean_runs"] == 2
    # run lengths 5 and 4 -> (5-3+1) + (4-3+1) = 3 + 2 = 5
    assert result["n_feasible_windows"] == 5


def test_assess_goes_sequence_feasibility_empty_input():
    empty = pd.DataFrame(columns=["time_utc", "goes_seq_xrsb_flux_w_m2", "goes_seq_xrsa_flux_w_m2", "goes_seq_interpolated_any"])
    result = assess_goes_sequence_feasibility(empty, 5, sequences_config=_sequences_config())
    assert result["feasible"] is False
    assert result["n_feasible_windows"] == 0
