"""Combines GOES sequence features (preprocessing/goes_sequences.py) with
real SHARP magnetic-descriptor sequences, closing the gap between Chapter
5's described solar input ("the GOES irradiance sequence AND SHARP
magnetic-descriptor sequence") and the GOES-only Chapter 3.6 LSTM built
earlier..

Reuses two real, already-tested pieces rather than inventing new
aggregation/scaling logic:
  - `preprocessing/alignment.py::aggregate_sharp_disk` -- the classical
    pipeline's real disk-level SHARP aggregation (per-keyword sum/max
    across active regions at each real T_REC timestamp).
  - `preprocessing/scaling.py::fit_robust_scaling`/`apply_robust_scaling`
    -- the classical pipeline's real, train-only-fit Winsorize+median/IQR
    scaling for exactly these 18 SHARP fields
    (`data_sources.schema.SHARP_PHYSICAL_KEYWORDS`).

The only genuinely new step here is the merge: SHARP's disk-aggregated
values (SHARP's own native ~12-min T_REC grid) onto GOES's independently
resampled 12-min grid (preprocessing/goes_preprocessing.py's output). This
is an EXACT `time_utc` match -- never a nearest/fuzzy join -- so a GOES
timestamp with no exactly-matching SHARP row, or any raw NaN SHARP value,
becomes a real, flagged gap (`sharp_seq_missing_any`), never silently
filled or interpolated. See preprocessing/goes_sequences.py's
`extra_gap_flag_columns` for how this flag reaches window-feasibility
checking.
"""
from __future__ import annotations

import pandas as pd

from data_sources.schema import SHARP_PHYSICAL_KEYWORDS

from .alignment import aggregate_sharp_disk, load_alignment_config
from .scaling import apply_robust_scaling, fit_robust_scaling, load_scaling_config

SHARP_SEQ_PREFIX = "sharp_seq_"
SHARP_SEQ_MISSING_COL = f"{SHARP_SEQ_PREFIX}missing_any"


def build_combined_sequence_table(
    processed_goes_df: pd.DataFrame,
    sharp_qc_accepted_df: pd.DataFrame,
    *,
    train_start: str,
    train_end: str,
    alignment_config: dict | None = None,
    scaling_config: dict | None = None,
    time_col: str = "time_utc",
) -> tuple[pd.DataFrame, dict]:
    """Returns (combined_df, scaling_params). combined_df is GOES's
    processed_goes_df with the 18 real SHARP columns (scaled, train-fit)
    and `sharp_seq_missing_any` appended -- a strict superset of
    processed_goes_df's own columns, same row count and order.

    scaling_params is fit_robust_scaling's real output (returned for
    inspection/testing, not just applied silently) -- fit on rows whose
    time_utc falls in [train_start, train_end] (inclusive), matching
    configs/splits.yaml's real train date range, never on val/test rows.
    """
    alignment_config = alignment_config or load_alignment_config()
    scaling_config = scaling_config or load_scaling_config()

    goes = processed_goes_df.copy()
    goes[time_col] = pd.to_datetime(goes[time_col], utc=True)

    sharp_disk = aggregate_sharp_disk(sharp_qc_accepted_df, alignment_config)
    sharp_disk[time_col] = pd.to_datetime(sharp_disk[time_col], utc=True)

    sharp_cols = [c for c in SHARP_PHYSICAL_KEYWORDS if c in sharp_disk.columns]
    merge_cols = [time_col, "n_active_regions_tracked"] + sharp_cols

    # Exact-match merge only -- a GOES timestamp with no identically-timed
    # SHARP row gets NaN in every SHARP column (real gap), never a
    # nearest-neighbour or interpolated substitute.
    merged = goes.merge(sharp_disk[merge_cols], on=time_col, how="left", validate="one_to_one")

    missing = merged[sharp_cols].isna().any(axis=1) | merged["n_active_regions_tracked"].isna()
    merged[SHARP_SEQ_MISSING_COL] = missing.fillna(True)

    train_start_ts = pd.Timestamp(train_start, tz="UTC")
    train_end_ts = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    train_mask = (merged[time_col] >= train_start_ts) & (merged[time_col] <= train_end_ts)

    scaling_params = fit_robust_scaling(merged.loc[train_mask], sharp_cols, scaling_config)
    merged = apply_robust_scaling(merged, scaling_params, copy=False)

    return merged, scaling_params
