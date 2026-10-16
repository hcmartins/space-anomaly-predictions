"""GOES sequence preprocessing (Chapter 3.4.1, Algorithm 3.1): log10
transform, MAD-based despiking, gap interpolation, causal moving-average
smoothing, rolling-percentile background normalization, resample to a
common cadence.

Feeds ONLY the new sequence-tensor construction path for the LSTM
(preprocessing/goes_sequences.py) -- does not touch
preprocessing/alignment.py::aggregate_goes_disk or
attach_goes_trailing_state, whose already-reported real numbers for the
classical/tabular pipeline stay untouched (same purely-additive principle
Phase 1/Phase 2 followed for per-HARP forecasting)..

Despiking uses a CENTERED window (Chapter 3.4.1's own formula: a symmetric
median/MAD window). This is safe for causality despite looking at
"future" samples within the local window: it is a one-time, offline
cleaning pass over the HISTORICAL ARCHIVE (removing real instrument
artifacts before any forecast instance is ever constructed), not a
per-forecast feature computed at issuance time -- no future information
about a specific flare ever reaches a specific forecast instance through
this step. Every step AFTER despiking (interpolation, smoothing,
background normalization) is trailing/causal only, exactly as
Chapter 3.4.1 specifies, and IS safe to use directly as a forecast-time
input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR

SEQ_PREFIX = "goes_seq_"


def load_goes_preprocessing_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "goes_preprocessing.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _rolling_mad(series: pd.Series, kernel_steps: int, min_periods: int) -> tuple[pd.Series, pd.Series]:
    """Two chained rolling-median passes (both O(n log k) via pandas'
    native rolling-median implementation, never a per-window Python
    .apply) -- returns (centered rolling median, rolling MAD around it),
    the exact quantities Chapter 3.4.1's despiking formula needs."""
    center = series.rolling(kernel_steps, center=True, min_periods=min_periods).median()
    mad = (series - center).abs().rolling(kernel_steps, center=True, min_periods=min_periods).median()
    return center, mad


def preprocess_goes_sequence(
    goes_disk_df: pd.DataFrame,
    *,
    config: dict | None = None,
    time_col: str = "time_utc",
) -> pd.DataFrame:
    """Runs Algorithm 3.1 steps 2-7 on preprocessing/alignment.py::
    aggregate_goes_disk's output (already QC'd and satellite-collapsed --
    step 1 is that upstream QC, reused unchanged).

    Returns a new dataframe at resample_cadence_minutes cadence with one
    `{SEQ_PREFIX}{var}` column per configured log_variable (the final
    despiked/interpolated/smoothed/background-normalized, resampled log
    flux) plus bookkeeping columns
    (`{SEQ_PREFIX}interpolated_any`/`{SEQ_PREFIX}gap_minutes_max`) --
    never fabricates a bin with no real underlying observation (an
    all-NaN bin's mean stays NaN, not silently zero-filled)."""
    config = config or load_goes_preprocessing_config()
    cadence = pd.Timedelta(minutes=config["native_cadence_minutes"])
    despike_kernel_steps = max(1, round(config["despike_kernel_minutes"] / config["native_cadence_minutes"]))
    n_sigma = config["despike_n_sigma"]
    g_max_steps = max(1, round(config["max_interpolated_gap_minutes"] / config["native_cadence_minutes"]))
    smoothing_steps = max(1, round(config["smoothing_window_minutes"] / config["native_cadence_minutes"]))
    bg_percentile = config["background_percentile"] / 100.0
    bg_window_steps = max(1, round(config["background_window_minutes"] / config["native_cadence_minutes"]))
    log_variables = [v for v in config["log_variables"] if v in goes_disk_df.columns]

    df = goes_disk_df.sort_values(time_col).reset_index(drop=True).copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    if df.empty or not log_variables:
        return pd.DataFrame(columns=[time_col] + [f"{SEQ_PREFIX}{v}" for v in log_variables])

    df["_observed"] = True
    full_index = pd.date_range(df[time_col].min(), df[time_col].max(), freq=cadence, tz="UTC")
    grid = df.set_index(time_col).reindex(full_index)
    grid.index.name = time_col
    observed = grid["_observed"].fillna(False).astype(bool)
    grid = grid.drop(columns=["_observed"])

    seq_cols: dict[str, pd.Series] = {}
    interpolated_any = pd.Series(False, index=full_index)
    spike_any = pd.Series(False, index=full_index)
    gap_minutes = pd.Series(0.0, index=full_index)

    for var in log_variables:
        raw = grid[var].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ell = np.log10(raw.where(raw > 0))

        center, mad = _rolling_mad(ell, despike_kernel_steps, min_periods=despike_kernel_steps // 2 + 1)
        threshold = n_sigma * 1.4826 * mad
        is_spike = (ell - center).abs() > threshold
        spike_any |= is_spike.fillna(False)
        ell_despiked = ell.where(~is_spike)

        interpolated = ell_despiked.interpolate(method="linear", limit=g_max_steps, limit_area="inside")
        was_interpolated = interpolated.notna() & ell_despiked.isna()
        interpolated_any |= was_interpolated.fillna(False)

        smoothed = interpolated.rolling(smoothing_steps, min_periods=1).mean()
        background = interpolated.rolling(bg_window_steps, min_periods=max(1, bg_window_steps // 4)).quantile(
            bg_percentile
        )
        bg_normalized = smoothed - background
        seq_cols[f"{SEQ_PREFIX}{var}"] = bg_normalized

    # gap length (minutes since the last genuinely observed sample, for
    # bookkeeping/reporting -- mirrors missing_data.py's gap_length_steps
    # convention, expressed in minutes here since this module's own
    # cadence is finer than the SHARP-aligned pipeline's.
    step_idx = np.arange(len(full_index))
    observed_step = np.where(observed.to_numpy(), step_idx, np.nan)
    last_observed_step = pd.Series(observed_step, index=full_index).ffill().to_numpy()
    with np.errstate(invalid="ignore"):
        gap_steps = step_idx - last_observed_step
    gap_minutes = pd.Series(
        np.where(np.isfinite(gap_steps), gap_steps * config["native_cadence_minutes"], np.nan),
        index=full_index,
    )

    out = pd.DataFrame(seq_cols, index=full_index)
    out[f"{SEQ_PREFIX}interpolated_any"] = interpolated_any
    out[f"{SEQ_PREFIX}spike_removed_any"] = spike_any
    out[f"{SEQ_PREFIX}gap_minutes"] = gap_minutes
    out.index.name = time_col
    out = out.reset_index()

    resample_cadence = pd.Timedelta(minutes=config["resample_cadence_minutes"])
    agg_map = {f"{SEQ_PREFIX}{v}": config["resample_aggregation"] for v in log_variables}
    agg_map[f"{SEQ_PREFIX}interpolated_any"] = "any"
    agg_map[f"{SEQ_PREFIX}spike_removed_any"] = "any"
    agg_map[f"{SEQ_PREFIX}gap_minutes"] = "max"

    resampled = out.set_index(time_col).resample(resample_cadence).agg(agg_map).reset_index()
    return resampled
