"""Cell-sol aggregation of QC-accepted Martian retrievals.

Aggregates by (cell_id, mars_year, sol) using uncertainty-weighted
(inverse-variance) means for variables with a documented uncertainty
companion, and simple arithmetic means otherwise. Settings (which variables
get which treatment, minimum-support thresholds) live in
configs/aggregation.yaml — see that file's header comment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.provenance import stamp_processed_dataset
from data_sources.schema import CONFIG_DIR

ATM_LEVEL_PREFIX = "atm_temp_k_level_"


def load_aggregation_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "aggregation.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _atm_level_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        c for c in df.columns
        if c.startswith(ATM_LEVEL_PREFIX) and not c.endswith("_uncertainty")
    )


def _weighted_mean_and_combined_uncertainty(values: np.ndarray, uncertainties: np.ndarray, epsilon: float):
    """Inverse-variance weighted mean over the non-NaN entries of `values`
    (paired with `uncertainties`). Returns (weighted_mean, combined_uncertainty,
    n) where n is the count of retrievals actually used. NaN mean/uncertainty
    and n=0 if no valid (value, uncertainty) pair exists."""
    mask = ~(np.isnan(values) | np.isnan(uncertainties))
    if not mask.any():
        return np.nan, np.nan, 0
    v = values[mask]
    u = np.clip(uncertainties[mask], epsilon, None)
    weights = 1.0 / (u ** 2)
    weighted_mean = float(np.sum(v * weights) / np.sum(weights))
    combined_uncertainty = float(np.sqrt(1.0 / np.sum(weights)))
    return weighted_mean, combined_uncertainty, int(mask.sum())


def _simple_mean(values: np.ndarray):
    mask = ~np.isnan(values)
    if not mask.any():
        return np.nan, 0
    return float(np.mean(values[mask])), int(mask.sum())


def compute_low_support_flag(
    df: pd.DataFrame,
    *,
    min_obs_per_cell_sol: int,
    min_pressure_level_coverage_frac: float,
) -> pd.Series:
    """Recompute the low_support_flag purely from already-aggregated
    observation_count / pressure_level_coverage_frac columns, without
    re-aggregating raw data. Used both for the primary run's retained
    low_support_flag column and by preprocessing/sensitivity.py to sweep
    thresholds cheaply.

    NaN pressure_level_coverage_frac (no atm_temp columns were aggregated at
    all, e.g. an empty vertical profile schema) is treated as "not
    applicable" rather than a failure, since the coverage requirement cannot
    be evaluated in that case.
    """
    low_obs = df["observation_count"] < min_obs_per_cell_sol
    coverage = df["pressure_level_coverage_frac"]
    low_coverage = coverage.notna() & (coverage < min_pressure_level_coverage_frac)
    return low_obs | low_coverage


def aggregate_by_cell_sol(
    accepted_df: pd.DataFrame,
    *,
    min_obs_per_cell_sol: int,
    min_pressure_level_coverage_frac: float,
    config: dict | None = None,
) -> pd.DataFrame:
    """Aggregate QC-accepted retrievals to one row per (cell_id, mars_year,
    sol). Requires a `cell_id` column (see preprocessing.grid.assign_cells)."""
    if "cell_id" not in accepted_df.columns:
        raise KeyError(
            "aggregate_by_cell_sol requires a 'cell_id' column — run "
            "preprocessing.grid.assign_cells on the accepted data first."
        )

    config = config or load_aggregation_config()
    epsilon = config["epsilon"]
    weighted_vars = config["uncertainty_weighted_variables"]
    simple_vars = config["simple_mean_variables"]
    atm_cols = _atm_level_columns(accepted_df)

    rows: list[dict] = []
    for (cell_id, mars_year, sol), group in accepted_df.groupby(["cell_id", "mars_year", "sol"], sort=False):
        row: dict = {
            "cell_id": cell_id,
            "mars_year": mars_year,
            "sol": sol,
            "observation_count": len(group),
        }

        combined_uncertainties: list[float] = []
        for var in weighted_vars:
            unc_col = f"{var}_uncertainty"
            mean, unc, n = _weighted_mean_and_combined_uncertainty(
                group[var].to_numpy(dtype=float),
                group[unc_col].to_numpy(dtype=float),
                epsilon,
            )
            row[var] = mean
            row[f"{var}_uncertainty"] = unc
            row[f"{var}_n"] = n
            if not np.isnan(unc):
                combined_uncertainties.append(unc)

        for var in simple_vars:
            mean, n = _simple_mean(group[var].to_numpy(dtype=float))
            row[var] = mean
            row[f"{var}_n"] = n

        levels_with_any_data = 0
        for col in atm_cols:
            unc_col = f"{col}_uncertainty"
            mean, unc, n = _weighted_mean_and_combined_uncertainty(
                group[col].to_numpy(dtype=float),
                group[unc_col].to_numpy(dtype=float),
                epsilon,
            )
            row[col] = mean
            row[unc_col] = unc
            if n > 0:
                levels_with_any_data += 1
                combined_uncertainties.append(unc)

        row["pressure_level_coverage_frac"] = (
            levels_with_any_data / len(atm_cols) if atm_cols else np.nan
        )
        row["uncertainty_summary"] = (
            float(np.mean(combined_uncertainties)) if combined_uncertainties else np.nan
        )
        row["interpolation_flag"] = False  # see configs/aggregation.yaml note
        rows.append(row)

    result = pd.DataFrame(rows)
    if len(result):
        result["low_support_flag"] = compute_low_support_flag(
            result,
            min_obs_per_cell_sol=min_obs_per_cell_sol,
            min_pressure_level_coverage_frac=min_pressure_level_coverage_frac,
        )
    else:
        result["low_support_flag"] = pd.Series([], dtype=bool)

    return stamp_processed_dataset(result, "SIMULATED-AGGREGATED-CELL-SOL")
