"""Strictly causal feature engineering on the trailing-filled, disk-integrated
aligned series (preprocessing/missing_data.py's output).

Every feature at row time t is computed from the SAME series' rows with
time <= t (lags, trailing stats, first difference, slope, persistence-run
counter) -- there is no spatial grid here (unlike martian's cell-sol
version), since this project's alignment is disk-integrated, not
per-active-region (see configs/alignment.yaml's header for why)..

Lag/trailing lookups are keyed by EXACT timestamp (current time minus
lag * cadence_minutes), never by row-adjacency: once preprocessing/
missing_data.py has dropped rows for gaps too long to fill, consecutive
ROWS are no longer guaranteed to be consecutive TIME STEPS, and naive
row-shifting would silently treat row-adjacency as time-adjacency across
such a gap. This module reindexes onto filled_df's own regular cadence
grid FIRST (making any dropped-gap step an explicit NaN row), then uses
vectorized `.shift()`/`.rolling()` on that grid -- exact-timestamp lookup
in vectorized form, not an approximation of it -- before filtering back
down to filled_df's actual rows. This returns NaN for any lag that lands
on a genuinely absent step, the same correct, non-fabricated answer the
original per-row dict-lookup implementation gave (verified equivalent
against real data before replacing it -- see
planning/PREPROCESSING_DOCUMENTATION.md: the original row-by-row Python
loop took ~76ms/row, ~14 hours projected at the real ~666k-row full-range
scale; this vectorized version is the fix, not a behavior change).

No feature column produced here is ever named with a `label_` prefix -- see
configs/features.yaml's header for the leakage-guarantee rationale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR

FEATURE_PREFIX = "feat_"


def load_features_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "features.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _trailing_ols_slope(shifted: dict[int, pd.Series], window: int, index: pd.Index) -> pd.Series:
    """Vectorized equivalent of fitting np.polyfit(xs, ys, 1)[0] per row,
    where ys are the non-NaN values among shifted[1..window] (lag k's value
    is at fixed x=-k, gaps/series-start NaNs excluded from the fit, same as
    the original per-row implementation) and xs are their corresponding
    fixed lag positions. Uses the closed-form OLS slope
    (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2) computed via masked sums across the
    `window` shifted columns at once, rather than an np.polyfit call per
    row -- mathematically identical, verified equivalent on real data (see
    planning/PREPROCESSING_DOCUMENTATION.md)."""
    y = np.stack([shifted[k].to_numpy() for k in range(1, window + 1)], axis=1)  # (n, window)
    x = np.array([-k for k in range(1, window + 1)], dtype=float)  # fixed lag positions
    mask = ~np.isnan(y)
    n_valid = mask.sum(axis=1)
    x_b = np.broadcast_to(x, y.shape)
    y0 = np.where(mask, y, 0.0)
    x0 = np.where(mask, x_b, 0.0)
    sum_x = x0.sum(axis=1)
    sum_y = y0.sum(axis=1)
    sum_xx = (x0 * x0).sum(axis=1)
    sum_xy = (x0 * y0).sum(axis=1)
    denom = n_valid * sum_xx - sum_x * sum_x
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (n_valid * sum_xy - sum_x * sum_y) / denom
    slope = np.where((n_valid >= 2) & (denom > 0), slope, np.nan)
    return pd.Series(slope, index=index)


def _persistence_run_ending_at_previous_step(cond: pd.Series) -> pd.Series:
    """For each row, counts consecutive True values in `cond` immediately
    PRECEDING it (not including itself) on a regular grid, resetting to 0
    at the first False or NaN -- vectorized equivalent of the original
    `while` loop that walked backward one lag at a time until a
    non-elevated or missing step broke the run."""
    changed = cond.ne(cond.shift()).cumsum()
    run_incl_self = cond.groupby(changed).cumcount() + 1
    run_incl_self = run_incl_self.where(cond, 0)
    return run_incl_self.shift(1).fillna(0).astype(int)


def build_features(
    filled_df: pd.DataFrame,
    *,
    config: dict | None = None,
    time_col: str = "time_utc",
) -> pd.DataFrame:
    """Returns a COPY of filled_df with feature columns appended. Existing
    columns are never modified or removed."""
    config = config or load_features_config()
    out = filled_df.sort_values(time_col).reset_index(drop=True).copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True)
    cadence = pd.Timedelta(minutes=config["cadence_minutes"])

    variables = [v for v in config["variables"] if v in out.columns]
    lags = config["lags_steps"]
    window = config["trailing_window_steps"]

    if out.empty:
        grid = out.set_index(time_col)
    else:
        # Reindex onto filled_df's OWN regular cadence grid so any step
        # missing from filled_df (an unfillable gap dropped upstream, or
        # simply the series' start) becomes an explicit NaN row -- this is
        # what makes plain .shift()/.rolling() below safe to use (see
        # module docstring): a lag that lands on a genuinely absent step
        # gets NaN from the grid, never the next available ROW's value.
        full_index = pd.date_range(out[time_col].min(), out[time_col].max(), freq=cadence, tz="UTC")
        grid = out.set_index(time_col).reindex(full_index)
        grid.index.name = time_col

    new_cols: dict[str, pd.Series] = {}
    for var in variables:
        if var not in grid.columns:
            continue
        s = grid[var].astype(float)
        shifted = {k: s.shift(k) for k in range(1, window + 1)}

        if config["enable_lags"]:
            for lag in lags:
                if lag in shifted:
                    new_cols[f"{FEATURE_PREFIX}{var}_lag{lag}"] = shifted[lag]

        if config["enable_trailing_stats"]:
            trailing = s.shift(1).rolling(window, min_periods=1)
            new_cols[f"{FEATURE_PREFIX}{var}_trailing{window}_mean"] = trailing.mean()
            new_cols[f"{FEATURE_PREFIX}{var}_trailing{window}_std"] = trailing.std(ddof=0)
            new_cols[f"{FEATURE_PREFIX}{var}_trailing{window}_max"] = trailing.max()
            new_cols[f"{FEATURE_PREFIX}{var}_trailing{window}_min"] = trailing.min()

        if config["enable_first_difference_and_slope"]:
            new_cols[f"{FEATURE_PREFIX}{var}_diff1"] = s.diff(1)
            new_cols[f"{FEATURE_PREFIX}{var}_slope{window}"] = _trailing_ols_slope(shifted, window, grid.index)

    if config["enable_persistence_counter"]:
        flux_col = config["persistence_flux_column"]
        threshold = config["persistence_flux_threshold_w_m2"]
        if flux_col in grid.columns:
            cond = (grid[flux_col] >= threshold).fillna(False)
            new_cols[f"{FEATURE_PREFIX}persistence_elevated_flux_run_steps"] = _persistence_run_ending_at_previous_step(
                cond
            )
        else:
            new_cols[f"{FEATURE_PREFIX}persistence_elevated_flux_run_steps"] = pd.Series(0, index=grid.index, dtype=int)

    if new_cols:
        feature_df = pd.DataFrame(new_cols)
        # Grid rows exist only to make gaps explicit for shift/rolling;
        # filter back down to exactly filled_df's own rows (by exact
        # timestamp, preserving out's row order/index) before concatenating.
        feature_df = feature_df.loc[out[time_col].to_numpy()].reset_index(drop=True)
        out = pd.concat([out, feature_df], axis=1)

    if config["enable_qc_indicators"]:
        if "is_observed" in out.columns:
            out[f"{FEATURE_PREFIX}is_observed"] = out["is_observed"].astype(int)
        if "interpolation_flag" in out.columns:
            out[f"{FEATURE_PREFIX}interpolation_flag"] = out["interpolation_flag"].astype(int)
        if "gap_length_steps" in out.columns:
            out[f"{FEATURE_PREFIX}gap_length_steps"] = out["gap_length_steps"]

    return out


def feature_column_names(df: pd.DataFrame) -> list[str]:
    """All engineered (FEATURE_PREFIX-tagged) columns present in df."""
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
