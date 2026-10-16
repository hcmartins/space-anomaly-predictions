"""Shared, strictly causal feature engineering for Phase 5 modeling tasks.

Every feature at (cell, sol=s) is computed from:
  - the SAME cell's rows with sol <= s (current state, lags, trailing
    stats, first difference, slope, persistence-run counter), or
  - a SPATIAL NEIGHBOUR's row at sol == s (spatial-neighbour stats -- this
    is legitimate "currently known" information at prediction time, never a
    future sol or a future cell-sol).

No feature column produced here is ever named with a `target_` prefix --
that prefix is reserved exclusively for task-specific targets built by
modeling/datasets.py. This is a structural guarantee against cross-task
target leakage: the feature table itself never contains any task's target,
so a task's dataset builder cannot accidentally pull in another task's
target as a feature.

Operates on the TRAILING-FILLED cell-sol series (preprocessing/missing_data.py)
rather than the raw aggregated series, so lag/rolling features are denser
than the raw observation sparsity would otherwise allow -- trailing-fill is
itself strictly causal (last-observation-carried-forward, max 2-sol gap by
default), so this does not introduce any future information. Each row's
`is_observed`/`interpolation_flag` columns (already present on the filled
series) are surfaced as the QC/interpolation-indicator feature family so a
model can distinguish a genuine observation from a carried-forward one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR

from .grid import GridDefinition, bands_from_cell_id, cell_id as grid_cell_id, neighbours

FEATURE_PREFIX = "feat_"


def load_features_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "features.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _build_lookup(df: pd.DataFrame, cell_col: str, mars_year_col: str, sol_col: str) -> dict:
    lookup = {}
    for idx, cid, my, sol in df[[cell_col, mars_year_col, sol_col]].itertuples(name=None):
        lookup[(int(cid), int(my), int(sol))] = idx
    return lookup


def build_features(
    filled_df: pd.DataFrame,
    grid: GridDefinition,
    *,
    config: dict | None = None,
    cell_col: str = "cell_id",
    mars_year_col: str = "mars_year",
    sol_col: str = "sol",
) -> pd.DataFrame:
    """Returns a COPY of filled_df with feature columns appended. Existing
    columns are never modified or removed."""
    config = config or load_features_config()
    out = filled_df.copy().reset_index(drop=True)
    lookup = _build_lookup(out, cell_col, mars_year_col, sol_col)

    variables = config["variables"]
    z_vars = config["z_score_variables"]
    lags = config["lags_sols"]
    window = config["trailing_window_sols"]
    min_completeness = config.get("min_window_completeness_frac")

    lat_band_arr, lon_band_arr = bands_from_cell_id(out[cell_col].to_numpy(), grid)

    # Preallocate feature columns.
    for var in variables + z_vars:
        if config["enable_lags"]:
            for lag in lags:
                out[f"{FEATURE_PREFIX}{var}_lag{lag}"] = np.nan
        if config["enable_trailing_stats"]:
            for stat in ("mean", "std", "max", "min"):
                out[f"{FEATURE_PREFIX}{var}_trailing{window}_{stat}"] = np.nan
        if config["enable_first_difference_and_slope"]:
            out[f"{FEATURE_PREFIX}{var}_diff1"] = np.nan
            out[f"{FEATURE_PREFIX}{var}_slope{window}"] = np.nan
        if config["enable_spatial_neighbour_stats"]:
            out[f"{FEATURE_PREFIX}{var}_neighbour_mean"] = np.nan
            out[f"{FEATURE_PREFIX}{var}_neighbour_max"] = np.nan

    if config["enable_persistence_counter"]:
        out[f"{FEATURE_PREFIX}persistence_run_sols"] = 0

    if config["enable_spatial_neighbour_stats"]:
        out[f"{FEATURE_PREFIX}neighbour_exceedance_count"] = 0

    if config["enable_ls_ltst_harmonics"]:
        ls_rad = np.radians(out["solar_longitude_ls"].to_numpy(dtype=float))
        ltst_rad = np.radians(out["local_true_solar_time"].to_numpy(dtype=float) / 24.0 * 360.0)
        out[f"{FEATURE_PREFIX}ls_sin"] = np.sin(ls_rad)
        out[f"{FEATURE_PREFIX}ls_cos"] = np.cos(ls_rad)
        out[f"{FEATURE_PREFIX}ltst_sin"] = np.sin(ltst_rad)
        out[f"{FEATURE_PREFIX}ltst_cos"] = np.cos(ltst_rad)

    if config["enable_qc_indicators"]:
        if "is_observed" in out.columns:
            out[f"{FEATURE_PREFIX}is_observed"] = out["is_observed"].astype(int)
        if "interpolation_flag" in out.columns:
            out[f"{FEATURE_PREFIX}interpolation_flag"] = out["interpolation_flag"].astype(int)
        if "low_support_flag" in out.columns:
            out[f"{FEATURE_PREFIX}low_support_flag"] = out["low_support_flag"].astype(int)
        if "observation_count" in out.columns:
            out[f"{FEATURE_PREFIX}observation_count"] = out["observation_count"]

    for row_pos, (idx, cid, my, sol) in enumerate(
        out[[cell_col, mars_year_col, sol_col]].itertuples(name=None)
    ):
        cid, my, sol = int(cid), int(my), int(sol)

        # ---- lags, trailing stats, first difference, slope (same cell, past only) ----
        for var in variables + z_vars:
            history = []  # (lag_offset, value) for lag in 1..window, oldest last
            for lag in range(1, window + 1):
                prior_idx = lookup.get((cid, my, sol - lag))
                val = out.at[prior_idx, var] if prior_idx is not None else np.nan
                history.append(val)
                if config["enable_lags"] and lag in lags:
                    out.at[idx, f"{FEATURE_PREFIX}{var}_lag{lag}"] = val

            hist_arr = np.array(history, dtype=float)
            valid = ~np.isnan(hist_arr)
            meets_completeness = min_completeness is None or (valid.sum() / window) >= min_completeness
            if config["enable_trailing_stats"] and valid.any() and meets_completeness:
                vals = hist_arr[valid]
                out.at[idx, f"{FEATURE_PREFIX}{var}_trailing{window}_mean"] = float(np.mean(vals))
                out.at[idx, f"{FEATURE_PREFIX}{var}_trailing{window}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
                out.at[idx, f"{FEATURE_PREFIX}{var}_trailing{window}_max"] = float(np.max(vals))
                out.at[idx, f"{FEATURE_PREFIX}{var}_trailing{window}_min"] = float(np.min(vals))

            if config["enable_first_difference_and_slope"]:
                lag1 = history[0] if len(history) >= 1 else np.nan
                current_val = out.at[idx, var]
                if not np.isnan(lag1) and not pd.isna(current_val):
                    out.at[idx, f"{FEATURE_PREFIX}{var}_diff1"] = float(current_val) - float(lag1)
                if valid.sum() >= 2:
                    xs = np.array([-(i + 1) for i, v in enumerate(history) if not np.isnan(v)], dtype=float)
                    ys = hist_arr[valid]
                    slope = float(np.polyfit(xs, ys, 1)[0])
                    out.at[idx, f"{FEATURE_PREFIX}{var}_slope{window}"] = slope

        # ---- persistence-run counter (same cell, past only, atmospheric_anomaly) ----
        if config["enable_persistence_counter"] and "atmospheric_anomaly" in out.columns:
            run = 0
            lag = 1
            while True:
                prior_idx = lookup.get((cid, my, sol - lag))
                if prior_idx is None:
                    break
                if bool(out.at[prior_idx, "atmospheric_anomaly"]):
                    run += 1
                    lag += 1
                else:
                    break
            out.at[idx, f"{FEATURE_PREFIX}persistence_run_sols"] = run

        # ---- spatial-neighbour stats (same sol, neighbouring cells) ----
        if config["enable_spatial_neighbour_stats"]:
            lat_band, lon_band = int(lat_band_arr[row_pos]), int(lon_band_arr[row_pos])
            neighbour_cells = [int(grid_cell_id(nla, nlo, grid)) for nla, nlo in neighbours(lat_band, lon_band, grid)]
            n_exceeding = 0
            for var in variables:
                neighbour_vals = []
                for ncid in neighbour_cells:
                    nidx = lookup.get((ncid, my, sol))
                    if nidx is not None:
                        neighbour_vals.append(out.at[nidx, var])
                if neighbour_vals:
                    arr = np.array(neighbour_vals, dtype=float)
                    out.at[idx, f"{FEATURE_PREFIX}{var}_neighbour_mean"] = float(np.mean(arr))
                    out.at[idx, f"{FEATURE_PREFIX}{var}_neighbour_max"] = float(np.max(arr))
            if "atmospheric_anomaly" in out.columns:
                for ncid in neighbour_cells:
                    nidx = lookup.get((ncid, my, sol))
                    if nidx is not None and bool(out.at[nidx, "atmospheric_anomaly"]):
                        n_exceeding += 1
                out.at[idx, f"{FEATURE_PREFIX}neighbour_exceedance_count"] = n_exceeding

    return out


def feature_column_names(df: pd.DataFrame) -> list[str]:
    """All engineered (FEATURE_PREFIX-tagged) columns present in df."""
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
