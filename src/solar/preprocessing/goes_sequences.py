"""GOES sequence-tensor construction (Chapter 3.2.1's X_G(t0), feeding
Chapter 3.6's LSTM). Consumes preprocessing/goes_preprocessing.py's
resampled, background-normalized output..

Two responsibilities, mirroring martian/modeling/sequence_feasibility.py's
proven pattern (dataset-level feasibility report, always run and always
reported, never skipped even when the result is negative) adapted to
solar's single, continuous disk-integrated series rather than a
per-(cell, mars_year)-grouped grid:

  1. `assess_goes_sequence_feasibility` -- a dataset-level sanity check run
     BEFORE committing to training: what fraction of possible windows are
     actually usable, given real data gaps.
  2. `build_goes_sequences` -- the actual per-issuance-time window builder.
     Algorithm 3.1 step 9's literal policy: "if it contains a flagged gap
     then discard and continue" -- zero tolerance, not a coverage-fraction
     threshold. Infeasible issuance times are reported (a `feasible` column
     covering EVERY issuance time), never silently dropped without a trace.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR

from .goes_preprocessing import SEQ_PREFIX, load_goes_preprocessing_config


def load_goes_sequences_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "goes_sequences.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def window_length_steps(sequences_config: dict, preprocessing_config: dict) -> int:
    """T_G: the number of resampled steps spanning look_back_hours,
    inclusive of the issuance step itself -- e.g. 6h look-back at 12-min
    cadence gives 30 steps spanning 29*12=348 minutes (~5.8h), a documented
    rounding consistent with this project's existing trailing-window
    conventions (see preprocessing/features.py's trailing_window_steps)."""
    cadence_minutes = preprocessing_config["resample_cadence_minutes"]
    return max(1, round(sequences_config["look_back_hours"] * 60 / cadence_minutes))


def _consecutive_run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs in a 1D boolean array -- a
    vectorized boundary-detection idiom (pad with False at both ends, find
    sign changes), not a per-row Python loop."""
    if len(mask) == 0:
        return np.array([], dtype=int)
    padded = np.concatenate(([0], mask.astype(int), [0]))
    boundaries = np.flatnonzero(np.diff(padded))
    return boundaries[1::2] - boundaries[0::2]


def _clean_mask(
    processed_goes_df: pd.DataFrame, feature_columns: list[str], *, extra_gap_flag_columns: tuple[str, ...] = ()
) -> np.ndarray:
    interpolated = processed_goes_df[f"{SEQ_PREFIX}interpolated_any"].fillna(True).astype(bool)
    for col in extra_gap_flag_columns:
        interpolated = interpolated | processed_goes_df[col].fillna(True).astype(bool)
    has_values = processed_goes_df[feature_columns].notna().all(axis=1)
    return (~interpolated & has_values).to_numpy()


def assess_goes_sequence_feasibility(
    processed_goes_df: pd.DataFrame,
    window_steps: int,
    *,
    sequences_config: dict | None = None,
    extra_gap_flag_columns: tuple[str, ...] = (),
) -> dict:
    """`extra_gap_flag_columns`: additional boolean gap-flag column names
    (e.g. sharp_sequences.py's SHARP_SEQ_MISSING_COL) OR'd into the GOES
    gap flag -- default empty means zero behavior change for every
    existing GOES-only caller."""
    sequences_config = sequences_config or load_goes_sequences_config()
    feature_columns = sequences_config["feature_columns"]
    if processed_goes_df.empty:
        return {
            "window_length_steps": window_steps, "n_total_steps": 0, "n_clean_runs": 0,
            "max_run_length_steps": 0, "n_feasible_windows": 0, "feasible": False,
        }

    clean = _clean_mask(processed_goes_df, feature_columns, extra_gap_flag_columns=extra_gap_flag_columns)
    run_lengths = _consecutive_run_lengths(clean)
    n_feasible_windows = int(np.sum(np.maximum(run_lengths - window_steps + 1, 0)))
    return {
        "window_length_steps": window_steps,
        "n_total_steps": int(len(processed_goes_df)),
        "n_clean_runs": int(len(run_lengths)),
        "max_run_length_steps": int(run_lengths.max()) if len(run_lengths) else 0,
        "n_feasible_windows": n_feasible_windows,
        "feasible": n_feasible_windows > 0,
    }


def build_goes_sequences(
    processed_goes_df: pd.DataFrame,
    issuance_times,
    *,
    sequences_config: dict | None = None,
    preprocessing_config: dict | None = None,
    time_col: str = "time_utc",
    extra_gap_flag_columns: tuple[str, ...] = (),
) -> tuple[np.ndarray, pd.DataFrame]:
    """Returns (X, meta): X has shape (n_feasible, T_G, d_G) in the SAME
    row order as meta.loc[meta['feasible']] -- meta covers EVERY issuance
    time (never silently dropped without a trace), each row's `feasible`
    flag stating whether Algorithm 3.1 step 9's window would be discarded.

    Vectorized via window_steps reindex passes (T_G ~= 30), never a
    per-issuance-time Python loop -- exact-timestamp lookup on
    processed_goes_df's own regular resampled grid, the same technique
    preprocessing/features.py already uses for lag/trailing lookups.

    `extra_gap_flag_columns`: additional boolean gap-flag column names
    (e.g. sharp_sequences.py's SHARP_SEQ_MISSING_COL) OR'd into the
    per-step gap flag alongside GOES's own -- default empty means zero
    behavior change for every existing GOES-only caller."""
    sequences_config = sequences_config or load_goes_sequences_config()
    preprocessing_config = preprocessing_config or load_goes_preprocessing_config()
    feature_columns = sequences_config["feature_columns"]
    cadence = pd.Timedelta(minutes=preprocessing_config["resample_cadence_minutes"])
    window_steps = window_length_steps(sequences_config, preprocessing_config)

    grid = processed_goes_df.set_index(time_col)
    issuance_index = pd.DatetimeIndex(pd.to_datetime(pd.Index(issuance_times), utc=True))
    n = len(issuance_index)
    d_G = len(feature_columns)

    X = np.full((n, window_steps, d_G), np.nan, dtype=float)
    flagged = np.zeros((n, window_steps), dtype=bool)

    for k in range(window_steps):
        lag_times = issuance_index - (window_steps - 1 - k) * cadence
        rows = grid.reindex(lag_times)
        X[:, k, :] = rows[feature_columns].to_numpy()
        step_interpolated = rows[f"{SEQ_PREFIX}interpolated_any"].fillna(True).astype(bool).to_numpy()
        for col in extra_gap_flag_columns:
            step_interpolated = step_interpolated | rows[col].fillna(True).astype(bool).to_numpy()
        step_missing = rows[feature_columns].isna().any(axis=1).to_numpy()
        flagged[:, k] = step_interpolated | step_missing

    any_nan = np.isnan(X).any(axis=(1, 2))
    any_flagged = flagged.any(axis=1)
    feasible = ~(any_nan | any_flagged)

    meta = pd.DataFrame({time_col: issuance_index, "feasible": feasible})
    X_feasible = X[feasible]
    return X_feasible, meta
