"""Tests for the real GOES+SHARP combined sequence table
(preprocessing/sharp_sequences.py) -- the merge/scaling/gap-flagging glue
between GOES's independently-built grid and SHARP's disk-aggregated
values, closing Chapter 5's described GOES+SHARP solar input gap."""
import numpy as np
import pandas as pd

from preprocessing.goes_sequences import build_goes_sequences
from preprocessing.sharp_sequences import SHARP_SEQ_MISSING_COL, build_combined_sequence_table


def _alignment_config():
    return {
        "sharp_time_column": "T_REC",
        "sharp_keyword_aggregation": {"USFLUX": "sum", "MEANGAM": "max"},
    }


def _scaling_config():
    return {"winsorize_lower_percentile": 1, "winsorize_upper_percentile": 99}


def _goes_grid(n_steps=6, start="2018-01-01T00:00:00Z"):
    times = pd.date_range(start, periods=n_steps, freq="12min", tz="UTC")
    return pd.DataFrame(
        {
            "time_utc": times,
            "goes_seq_xrsb_flux_w_m2": np.arange(n_steps, dtype=float),
            "goes_seq_xrsa_flux_w_m2": np.arange(n_steps, dtype=float) * 10.0,
            "goes_seq_interpolated_any": [False] * n_steps,
        }
    )


def _sharp_raw_rows(rows):
    """`rows`: list of (t_rec_str, harpnum, usflux, meangam)."""
    return pd.DataFrame(
        [{"T_REC": t, "HARPNUM": h, "USFLUX": u, "MEANGAM": g} for t, h, u, g in rows]
    )


def test_combined_table_exact_match_merge_leaves_unmatched_goes_row_nan_and_flagged():
    goes = _goes_grid()
    # SHARP rows at 00:00, 00:12, 00:48, 01:00 -- 00:24 (goes row index 2) and
    # 00:36 (index 3) deliberately have no SHARP row at all.
    sharp_raw = _sharp_raw_rows(
        [
            ("2018.01.01_00:00:00_TAI", 1, 1.0, 2.0),
            ("2018.01.01_00:12:00_TAI", 1, 3.0, 4.0),
            ("2018.01.01_00:48:00_TAI", 1, 7.0, 8.0),
            ("2018.01.01_01:00:00_TAI", 1, 9.0, 10.0),
        ]
    )
    combined, params = build_combined_sequence_table(
        goes, sharp_raw, train_start="2018-01-01", train_end="2018-01-01",
        alignment_config=_alignment_config(), scaling_config=_scaling_config(),
    )
    assert not combined.loc[0, SHARP_SEQ_MISSING_COL]
    assert not combined.loc[1, SHARP_SEQ_MISSING_COL]
    # No exact SHARP match -- a nearest/fuzzy join would have filled this in.
    assert combined.loc[2, SHARP_SEQ_MISSING_COL]
    assert pd.isna(combined.loc[2, "USFLUX"])
    assert not combined.loc[4, SHARP_SEQ_MISSING_COL]
    assert not combined.loc[5, SHARP_SEQ_MISSING_COL]


def test_combined_table_flags_raw_nan_sharp_value_even_when_timestamp_matches():
    goes = _goes_grid()
    # USFLUX uses "sum" aggregation (pandas sums an all-NaN single-HARPNUM
    # group to 0.0, not NaN) -- MEANGAM uses "max", which correctly
    # propagates NaN for an all-NaN group, so that's the column used here to
    # isolate real raw-NaN-value flagging from aggregate_sharp_disk's own
    # (pre-existing, unmodified, out-of-scope) sum-of-all-NaN behavior.
    sharp_raw = _sharp_raw_rows(
        [
            ("2018.01.01_00:00:00_TAI", 1, 1.0, 2.0),
            ("2018.01.01_00:12:00_TAI", 1, 3.0, 4.0),
            ("2018.01.01_00:24:00_TAI", 1, 5.0, 6.0),
            ("2018.01.01_00:36:00_TAI", 1, 6.0, np.nan),   # exact-time match, but raw NaN value
            ("2018.01.01_00:48:00_TAI", 1, 7.0, 8.0),
            ("2018.01.01_01:00:00_TAI", 1, 9.0, 10.0),
        ]
    )
    combined, _ = build_combined_sequence_table(
        goes, sharp_raw, train_start="2018-01-01", train_end="2018-01-01",
        alignment_config=_alignment_config(), scaling_config=_scaling_config(),
    )
    assert combined.loc[3, SHARP_SEQ_MISSING_COL]
    assert not combined.loc[2, SHARP_SEQ_MISSING_COL]


def test_combined_table_scaling_fit_on_train_range_rows_only():
    # Day 1 (train range) -- USFLUX values 1, 2, 3, 4. Day 2 (outside train
    # range) -- an extreme USFLUX=1000 that must NOT influence the fitted
    # median/scale if train-only-fit is honored.
    goes = pd.concat(
        [_goes_grid(n_steps=4, start="2018-01-01T00:00:00Z"), _goes_grid(n_steps=1, start="2018-01-02T00:00:00Z")],
        ignore_index=True,
    )
    sharp_raw = _sharp_raw_rows(
        [
            ("2018.01.01_00:00:00_TAI", 1, 1.0, 1.0),
            ("2018.01.01_00:12:00_TAI", 1, 2.0, 1.0),
            ("2018.01.01_00:24:00_TAI", 1, 3.0, 1.0),
            ("2018.01.01_00:36:00_TAI", 1, 4.0, 1.0),
            ("2018.01.02_00:00:00_TAI", 1, 1000.0, 1.0),
        ]
    )
    combined_train_only, params_train_only = build_combined_sequence_table(
        goes, sharp_raw, train_start="2018-01-01", train_end="2018-01-01",
        alignment_config=_alignment_config(), scaling_config=_scaling_config(),
    )
    # Median of [1,2,3,4] is 2.5 -- if the extreme day-2 row had leaked into
    # the fit, the median would be pulled far higher.
    assert params_train_only["USFLUX"]["median"] == 2.5
    assert combined_train_only["USFLUX"].max() < 100   # day-2's raw 1000 scaled down, not left raw


def test_build_goes_sequences_discards_window_touching_a_sharp_gap_via_extra_gap_flag_columns():
    goes = _goes_grid(n_steps=10)
    # SHARP row missing entirely at index 5 (falls inside a look-back window
    # ending at issuance index 6) -- GOES's own gap flag stays False
    # throughout, so only extra_gap_flag_columns should cause the discard.
    t_recs = [
        f"2018.01.01_{h:02d}:{m:02d}:00_TAI"
        for h, m in [(0, 0), (0, 12), (0, 24), (0, 36), (0, 48), (1, 0), (1, 12), (1, 24), (1, 36), (1, 48)]
    ]
    rows = [(t, 1, 1.0, 1.0) for i, t in enumerate(t_recs) if i != 5]   # skip index 5 (01:00)
    sharp_raw = _sharp_raw_rows(rows)
    combined, _ = build_combined_sequence_table(
        goes, sharp_raw, train_start="2018-01-01", train_end="2018-01-03",
        alignment_config=_alignment_config(), scaling_config=_scaling_config(),
    )
    assert combined[SHARP_SEQ_MISSING_COL].sum() == 1
    assert combined.loc[5, SHARP_SEQ_MISSING_COL]

    seq_config = {
        "feature_columns": ["goes_seq_xrsb_flux_w_m2", "goes_seq_xrsa_flux_w_m2", "USFLUX", "MEANGAM"],
        "look_back_hours": 1,   # window_steps=5 at 12-min cadence
    }
    preprocessing_config = {"resample_cadence_minutes": 12}
    issuance_times = [combined["time_utc"].iloc[6]]   # window spans indices 2..6, includes the gap at 5
    X, meta = build_goes_sequences(
        combined, issuance_times, sequences_config=seq_config, preprocessing_config=preprocessing_config,
        extra_gap_flag_columns=(SHARP_SEQ_MISSING_COL,),
    )
    assert not meta["feasible"].iloc[0]
    assert X.shape == (0, 5, 4)


def test_build_goes_sequences_keeps_window_with_no_sharp_gap():
    goes = _goes_grid(n_steps=10)
    t_recs = [
        f"2018.01.01_{h:02d}:{m:02d}:00_TAI"
        for h, m in [(0, 0), (0, 12), (0, 24), (0, 36), (0, 48), (1, 0), (1, 12), (1, 24), (1, 36), (1, 48)]
    ]
    sharp_raw = _sharp_raw_rows([(t, 1, 1.0, 1.0) for t in t_recs])
    combined, _ = build_combined_sequence_table(
        goes, sharp_raw, train_start="2018-01-01", train_end="2018-01-03",
        alignment_config=_alignment_config(), scaling_config=_scaling_config(),
    )
    assert combined[SHARP_SEQ_MISSING_COL].sum() == 0

    seq_config = {
        "feature_columns": ["goes_seq_xrsb_flux_w_m2", "goes_seq_xrsa_flux_w_m2", "USFLUX", "MEANGAM"],
        "look_back_hours": 1,
    }
    preprocessing_config = {"resample_cadence_minutes": 12}
    issuance_times = [combined["time_utc"].iloc[6]]
    X, meta = build_goes_sequences(
        combined, issuance_times, sequences_config=seq_config, preprocessing_config=preprocessing_config,
        extra_gap_flag_columns=(SHARP_SEQ_MISSING_COL,),
    )
    assert meta["feasible"].iloc[0]
    assert X.shape == (1, 5, 4)
