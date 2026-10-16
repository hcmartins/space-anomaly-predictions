"""Anomaly/storm-episode labels (Eq. 4.1-4.3) and forecast-horizon targets.

Dust-only: no real climatological source exists for surface_temperature_k
(RealMCDClimatologyProvider covers dust_optical_depth only, built from real
Montabone et al. data -- see data_sources/real_mcd_provider.py), so this
pipeline was descoped to dust rather than faking a temperature climatology.

Pipeline, all thresholds read from configs/anomaly_definition.yaml:

  1. z_v(c,s) = (x_v - mu_v) / max(sigma_v, epsilon)                 [Eq. 4.1]
  2. dust_anomaly(c,s)        = z_dust > tau                 (one-sided high)
     atmospheric_anomaly(c,s) = dust_anomaly                         [Eq. 4.2]
  3. storm_episode(c,s): atmospheric_anomaly cell-sols are grouped into
     spatiotemporal episodes (preprocessing.grid 8-neighbour + wraparound
     spatial connectivity, plus same-cell/spatial-neighbour temporal linking
     -- see labeling/connected_components.py); an episode's member cell-sols
     get storm_episode=True iff duration_sols >= persistence_sols AND
     peak_area_cells >= min_area_cells                              [Eq. 4.3]
  4. horizon_{h}sol(c,s) = any(storm_episode(c, s') for s < s' <= s+h),
     same cell, same mars_year -- "any event within the future horizon."

Operates ONLY on directly observed cell-sols (never trailing-filled ones --
see `_exclude_interpolated_rows`), so a carried-forward value is never
mistaken for a detected anomaly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR
from preprocessing.grid import GridDefinition

from .climatology_deviation import compute_z_scores
from .connected_components import assign_episode_ids, summarize_episodes

DUST_VARIABLE = "dust_optical_depth"


def load_anomaly_definition_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "anomaly_definition.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _exclude_interpolated_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Anomaly detection must never run on a trailing-filled (carried-
    forward) cell-sol -- that would flag our own imputation as a detected
    event. If the input carries an `interpolation_flag` or `is_observed`
    column (i.e. it came from preprocessing.missing_data), filter to real
    observations only; otherwise (a fresh aggregation.py output, which is
    always real observations) pass through unchanged."""
    out = df
    if "interpolation_flag" in out.columns:
        out = out.loc[~out["interpolation_flag"].astype(bool)]
    if "is_observed" in out.columns:
        out = out.loc[out["is_observed"].astype(bool)]
    return out.reset_index(drop=True)


def compute_exceedance(z_df: pd.DataFrame, *, tau: float, config: dict | None = None) -> pd.DataFrame:
    config = config or load_anomaly_definition_config()
    out = z_df.copy()
    out["dust_anomaly"] = out[f"{DUST_VARIABLE}_z"] > tau
    out["atmospheric_anomaly"] = out["dust_anomaly"]
    return out


def compute_storm_episodes(
    df_with_exceedance: pd.DataFrame,
    grid: GridDefinition,
    *,
    persistence_sols: int,
    min_area_cells: int,
    cell_id_col: str = "cell_id",
    mars_year_col: str = "mars_year",
    sol_col: str = "sol",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df_with_storm_episode_column, episode_catalogue). The
    catalogue includes EVERY candidate episode (connected component of
    atmospheric_anomaly cell-sols), with a `qualifies` column recording
    whether it passed the persistence+area gate; only qualifying episodes'
    member rows get storm_episode=True in the returned dataframe. An
    `episode_id` column is also attached (candidate id for every exceedance
    row, `pd.NA` elsewhere) so downstream code -- e.g.
    labeling/catalogue_validation.py -- can recover a qualifying episode's
    exact member cells via `out.loc[out["storm_episode"]].groupby("episode_id")`
    without recomputing connectivity."""
    out = df_with_exceedance.copy()
    out["storm_episode"] = False
    out["episode_id"] = pd.NA

    exceedance_rows = out.loc[out["atmospheric_anomaly"]]
    if exceedance_rows.empty:
        empty_catalogue = pd.DataFrame(
            columns=[
                "episode_id", "mars_year", "onset_sol", "end_sol", "duration_sols",
                "n_member_cell_sols", "n_distinct_cells", "peak_area_cells",
                "centroid_lat", "centroid_lon", "longitude_sectors_spanned", "qualifies",
            ]
        )
        return out, empty_catalogue

    episode_ids = assign_episode_ids(
        exceedance_rows, grid, cell_id_col=cell_id_col, mars_year_col=mars_year_col, sol_col=sol_col
    )
    exceedance_rows = exceedance_rows.copy()
    exceedance_rows["episode_id"] = episode_ids

    catalogue = summarize_episodes(
        exceedance_rows,
        cell_id_col=cell_id_col,
        mars_year_col=mars_year_col,
        sol_col=sol_col,
        lat_col=lat_col,
        lon_col=lon_col,
    )
    catalogue["qualifies"] = (catalogue["duration_sols"] >= persistence_sols) & (
        catalogue["peak_area_cells"] >= min_area_cells
    )

    out.loc[exceedance_rows.index, "episode_id"] = exceedance_rows["episode_id"]

    qualifying_ids = set(catalogue.loc[catalogue["qualifies"], "episode_id"])
    qualifying_row_index = exceedance_rows.loc[exceedance_rows["episode_id"].isin(qualifying_ids)].index
    out.loc[qualifying_row_index, "storm_episode"] = True

    return out, catalogue


def compute_horizon_labels(
    df_with_storm_episode: pd.DataFrame,
    *,
    horizons_sols: tuple[int, ...] = (1, 3, 5),
    group_cols: tuple[str, ...] = ("cell_id", "mars_year"),
    sol_col: str = "sol",
    storm_col: str = "storm_episode",
) -> pd.DataFrame:
    """horizon_{h}sol(c,s) = any(storm_episode(c,s') for s < s' <= s+h),
    same cell + mars_year, over OBSERVED sols only; NaN when no observed sol
    of that cell falls in (s, s+h] (target undefined, never assumed 0).
    Membership target, not onset: a cell already inside a continuing episode
    is positive. Computed per group via binary search on the
    group's sorted sol array (fast even though sampling is irregular)."""
    out = df_with_storm_episode.copy()
    # Unobserved-horizon policy (2026-10-04): a target is DEFINED only if the
    # same cell has at least one OBSERVED sol in (s, s+h]; otherwise it is NaN
    # ("unknown") and modeling.train._horizon_frame drops it. Previously such
    # rows were silently labelled 0 ("no episode") -- 64.9% / 23.6% / 15.5% of
    # h = 1 / 3 / 5 targets in the 2026-08-13 run (martian_correction_log_audit.md).
    for h in horizons_sols:
        out[f"horizon_{h}sol"] = np.nan

    for _, group in out.groupby(list(group_cols), sort=False):
        idx = group.index.to_numpy()
        sols = group[sol_col].to_numpy()
        order = np.argsort(sols)
        sorted_sols = sols[order]
        sorted_storm = group[storm_col].to_numpy()[order]
        sorted_idx = idx[order]

        # cumulative-OR-from-the-right so any(storm[i+1:j]) is an O(1) lookup
        suffix_any = np.zeros(len(sorted_storm) + 1, dtype=bool)
        for i in range(len(sorted_storm) - 1, -1, -1):
            suffix_any[i] = sorted_storm[i] or suffix_any[i + 1]

        for h in horizons_sols:
            upper_bound = np.searchsorted(sorted_sols, sorted_sols + h, side="right")
            for i in range(len(sorted_sols)):
                lo, hi = i + 1, upper_bound[i]
                if lo < hi:  # >= 1 observed sol in (s, s+h] -> defined target; else stays NaN
                    out.loc[sorted_idx[i], f"horizon_{h}sol"] = float(bool(sorted_storm[lo:hi].any()))

    return out


def run_full_labeling_pipeline(
    aggregated_df: pd.DataFrame,
    mcd_provider,
    grid: GridDefinition,
    *,
    config: dict | None = None,
    tau: float | None = None,
    persistence_sols: int | None = None,
    min_area_cells: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: exclude interpolated rows -> z-scores -> exceedance ->
    storm episodes -> horizon labels. Returns (labeled_df, episode_catalogue).
    Defaults to the PRIMARY settings in configs/anomaly_definition.yaml
    unless overridden (used by the sensitivity sweep)."""
    config = config or load_anomaly_definition_config()
    tau = tau if tau is not None else config["tau_primary"]
    persistence_sols = persistence_sols if persistence_sols is not None else config["persistence_sols_primary"]
    min_area_cells = min_area_cells if min_area_cells is not None else config["min_area_cells_primary"]

    observed = _exclude_interpolated_rows(aggregated_df)
    z_df = compute_z_scores(observed, mcd_provider, epsilon=config["epsilon"])
    exceedance_df = compute_exceedance(z_df, tau=tau, config=config)
    labeled_df, catalogue = compute_storm_episodes(
        exceedance_df, grid, persistence_sols=persistence_sols, min_area_cells=min_area_cells
    )
    labeled_df = compute_horizon_labels(labeled_df, horizons_sols=tuple(config["horizons_sols"]))
    return labeled_df, catalogue


def run_anomaly_sensitivity(
    aggregated_df: pd.DataFrame,
    mcd_provider,
    grid: GridDefinition,
    *,
    config: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """One-factor-at-a-time sensitivity sweeps (mirrors
    preprocessing/sensitivity.py's convention): tau, persistence_sols, and
    min_area_cells swept independently, holding the other two at their
    primary value. z-scores are computed once and reused across the tau
    sweep (z doesn't depend on tau/persistence/area)."""
    config = config or load_anomaly_definition_config()
    observed = _exclude_interpolated_rows(aggregated_df)
    z_df = compute_z_scores(observed, mcd_provider, epsilon=config["epsilon"])

    primary_tau = config["tau_primary"]
    primary_persistence = config["persistence_sols_primary"]
    primary_min_area = config["min_area_cells_primary"]
    proposed = config["sensitivity"]

    def _summarize(tau: float, persistence_sols: int, min_area_cells: int) -> dict:
        exceedance_df = compute_exceedance(z_df, tau=tau, config=config)
        labeled_df, catalogue = compute_storm_episodes(
            exceedance_df, grid, persistence_sols=persistence_sols, min_area_cells=min_area_cells
        )
        labeled_df = compute_horizon_labels(labeled_df, horizons_sols=tuple(config["horizons_sols"]))
        n = len(labeled_df)
        return {
            "n_cell_sols": n,
            "n_dust_anomaly": int(labeled_df["dust_anomaly"].sum()),
            "n_atmospheric_anomaly": int(labeled_df["atmospheric_anomaly"].sum()),
            "n_storm_episode": int(labeled_df["storm_episode"].sum()),
            "n_candidate_episodes": int(len(catalogue)),
            "n_qualifying_episodes": int(catalogue["qualifies"].sum()) if len(catalogue) else 0,
        }

    tau_rows = []
    for tau in config["tau_sensitivity"]:
        row = {"tau": tau}
        row.update(_summarize(tau, primary_persistence, primary_min_area))
        tau_rows.append(row)

    persistence_rows = []
    for persistence_sols in proposed["persistence_sols"]:
        row = {"persistence_sols": persistence_sols}
        row.update(_summarize(primary_tau, persistence_sols, primary_min_area))
        persistence_rows.append(row)

    min_area_rows = []
    for min_area_cells in proposed["min_area_cells"]:
        row = {"min_area_cells": min_area_cells}
        row.update(_summarize(primary_tau, primary_persistence, min_area_cells))
        min_area_rows.append(row)

    return {
        "tau_sensitivity": pd.DataFrame(tau_rows),
        "persistence_sensitivity": pd.DataFrame(persistence_rows),
        "min_area_sensitivity": pd.DataFrame(min_area_rows),
    }
