"""Tests for causal LOCF gap-filling of the disk-integrated aligned series."""
import numpy as np
import pandas as pd

from preprocessing.missing_data import apply_trailing_fill, load_missing_data_config, run_missing_data_sensitivity


def _config(max_gap=3):
    config = load_missing_data_config()
    config["max_trailing_fill_gap_steps"] = max_gap
    return config


def test_no_gaps_all_rows_observed():
    df = pd.DataFrame(
        [
            {"time_utc": "2018-01-01T00:00:00Z", "USFLUX": 1.0},
            {"time_utc": "2018-01-01T00:12:00Z", "USFLUX": 2.0},
            {"time_utc": "2018-01-01T00:24:00Z", "USFLUX": 3.0},
        ]
    )
    out = apply_trailing_fill(df, config=_config())
    assert len(out) == 3
    assert out["is_observed"].all()
    assert (out["gap_length_steps"] == 0).all()
    assert not out["interpolation_flag"].any()


def test_short_gap_is_filled_by_locf():
    # Missing the 00:12 step -- a 1-step (12 min) gap, within max_gap=3.
    df = pd.DataFrame(
        [
            {"time_utc": "2018-01-01T00:00:00Z", "USFLUX": 1.0},
            {"time_utc": "2018-01-01T00:24:00Z", "USFLUX": 3.0},
        ]
    )
    out = apply_trailing_fill(df, config=_config())
    assert len(out) == 3
    filled_row = out[out["time_utc"] == pd.Timestamp("2018-01-01T00:12:00Z")].iloc[0]
    assert filled_row["USFLUX"] == 1.0  # carried forward from 00:00, not from the future 3.0
    assert filled_row["is_observed"] == False
    assert filled_row["interpolation_flag"] == True
    assert filled_row["gap_length_steps"] == 1


def test_gap_beyond_max_is_dropped_not_fabricated():
    # Missing 4 consecutive steps (48 min) -- exceeds max_gap=3.
    df = pd.DataFrame(
        [
            {"time_utc": "2018-01-01T00:00:00Z", "USFLUX": 1.0},
            {"time_utc": "2018-01-01T01:00:00Z", "USFLUX": 9.0},
        ]
    )
    out = apply_trailing_fill(df, config=_config(max_gap=3))
    # 00:00 (observed), 00:12/00:24/00:36 (filled, gap<=3), 00:48 dropped (gap=4), 01:00 (observed)
    assert len(out) == 5
    assert pd.Timestamp("2018-01-01T00:48:00Z") not in set(out["time_utc"])
    assert pd.Timestamp("2018-01-01T00:36:00Z") in set(out["time_utc"])


def test_sensitivity_sweep_reports_counts_per_threshold():
    df = pd.DataFrame(
        [
            {"time_utc": "2018-01-01T00:00:00Z", "USFLUX": 1.0},
            {"time_utc": "2018-01-01T01:00:00Z", "USFLUX": 9.0},
        ]
    )
    result = run_missing_data_sensitivity(df, config=_config())
    assert list(result["max_trailing_fill_gap_steps"]) == [1, 3, 6]
    assert (result["n_observed"] == 2).all()
    # Larger max_gap fills more of the 4-step (48 min) gap.
    counts = dict(zip(result["max_trailing_fill_gap_steps"], result["n_filled"]))
    assert counts[1] < counts[3] < counts[6]
