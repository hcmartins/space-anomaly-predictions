"""One-factor-at-a-time sensitivity analyses for the Chapter 4 preprocessing
pipeline: grid resolution, minimum observations per cell-sol, and minimum
pressure-level coverage. Settings swept live in configs/aggregation.yaml
(`sensitivity:` block) / configs/grid.yaml (`sensitivity_resolutions_deg`).

Grid-resolution sensitivity requires re-running cell assignment and
aggregation, since cell_id itself depends on grid resolution. Min-obs and
pressure-coverage sensitivity are cheap post-hoc recomputations of
low_support_flag from an already-aggregated (primary-grid) dataframe, via
preprocessing.aggregation.compute_low_support_flag — they do NOT re-aggregate
raw retrievals for every threshold combination.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .aggregation import (
    aggregate_by_cell_sol,
    compute_low_support_flag,
    load_aggregation_config,
)
from .grid import assign_cells, load_primary_grid, load_sensitivity_grids


def grid_resolution_sensitivity(
    accepted_df: pd.DataFrame,
    *,
    min_obs_per_cell_sol: int,
    min_pressure_level_coverage_frac: float,
    grid_config_path: Path | None = None,
    aggregation_config: dict | None = None,
) -> pd.DataFrame:
    """Re-grid and re-aggregate the accepted retrievals at the primary grid
    resolution plus every configured sensitivity resolution, reporting how
    populated cell-sol coverage and low-support rates change with resolution.
    If a sensitivity resolution equals the primary resolution, it is
    evaluated once and flagged is_primary=True (not duplicated)."""
    aggregation_config = aggregation_config or load_aggregation_config()
    primary_grid = load_primary_grid(grid_config_path)
    sensitivity_grids = load_sensitivity_grids(grid_config_path)

    grids_to_run = [primary_grid]
    for g in sensitivity_grids:
        if g.resolution_deg != primary_grid.resolution_deg:
            grids_to_run.append(g)

    rows: list[dict] = []
    for grid in grids_to_run:
        gridded = assign_cells(accepted_df, grid)
        aggregated = aggregate_by_cell_sol(
            gridded,
            min_obs_per_cell_sol=min_obs_per_cell_sol,
            min_pressure_level_coverage_frac=min_pressure_level_coverage_frac,
            config=aggregation_config,
        )
        n_populated = len(aggregated)
        rows.append(
            {
                "resolution_deg": grid.resolution_deg,
                "n_cells_nominal": grid.n_cells,
                "n_cell_sols_populated": n_populated,
                "n_cell_sols_low_support": int(aggregated["low_support_flag"].sum()) if n_populated else 0,
                "mean_observations_per_cell_sol": (
                    float(aggregated["observation_count"].mean()) if n_populated else float("nan")
                ),
                "is_primary": grid.resolution_deg == primary_grid.resolution_deg,
            }
        )

    return pd.DataFrame(rows)


def min_obs_sensitivity(
    aggregated_primary_grid_df: pd.DataFrame,
    *,
    min_obs_values: list[int],
    min_pressure_level_coverage_frac: float,
) -> pd.DataFrame:
    """Sweep the minimum-observations-per-cell-sol threshold (holding
    pressure-coverage fixed at its primary value) against an already-aggregated
    primary-grid dataframe. Cheap: recomputes low_support_flag only."""
    n_total = len(aggregated_primary_grid_df)
    rows: list[dict] = []
    for min_obs in min_obs_values:
        flag = compute_low_support_flag(
            aggregated_primary_grid_df,
            min_obs_per_cell_sol=min_obs,
            min_pressure_level_coverage_frac=min_pressure_level_coverage_frac,
        )
        n_low_support = int(flag.sum())
        n_retained = n_total - n_low_support
        rows.append(
            {
                "min_obs_per_cell_sol": min_obs,
                "n_cell_sols_total": n_total,
                "n_cell_sols_low_support": n_low_support,
                "n_cell_sols_retained": n_retained,
                "retained_fraction": (n_retained / n_total) if n_total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def pressure_coverage_sensitivity(
    aggregated_primary_grid_df: pd.DataFrame,
    *,
    coverage_values: list[float],
    min_obs_per_cell_sol: int,
) -> pd.DataFrame:
    """Sweep the minimum-pressure-level-coverage threshold (holding min-obs
    fixed at its primary value) against an already-aggregated primary-grid
    dataframe. Cheap: recomputes low_support_flag only."""
    n_total = len(aggregated_primary_grid_df)
    rows: list[dict] = []
    for coverage in coverage_values:
        flag = compute_low_support_flag(
            aggregated_primary_grid_df,
            min_obs_per_cell_sol=min_obs_per_cell_sol,
            min_pressure_level_coverage_frac=coverage,
        )
        n_low_support = int(flag.sum())
        n_retained = n_total - n_low_support
        rows.append(
            {
                "min_pressure_level_coverage_frac": coverage,
                "n_cell_sols_total": n_total,
                "n_cell_sols_low_support": n_low_support,
                "n_cell_sols_retained": n_retained,
                "retained_fraction": (n_retained / n_total) if n_total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def run_all_sensitivity_analyses(
    accepted_df: pd.DataFrame,
    *,
    aggregation_config: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Orchestrate all three sensitivity analyses at their configured
    settings, returning a dict keyed by analysis name."""
    config = aggregation_config or load_aggregation_config()
    primary = config["primary"]
    sensitivity = config["sensitivity"]

    grid_sens = grid_resolution_sensitivity(
        accepted_df,
        min_obs_per_cell_sol=primary["min_obs_per_cell_sol"],
        min_pressure_level_coverage_frac=primary["min_pressure_level_coverage_frac"],
        aggregation_config=config,
    )

    primary_grid = load_primary_grid()
    gridded_primary = assign_cells(accepted_df, primary_grid)
    aggregated_primary = aggregate_by_cell_sol(
        gridded_primary,
        min_obs_per_cell_sol=primary["min_obs_per_cell_sol"],
        min_pressure_level_coverage_frac=primary["min_pressure_level_coverage_frac"],
        config=config,
    )

    min_obs_sens = min_obs_sensitivity(
        aggregated_primary,
        min_obs_values=sensitivity["min_obs_per_cell_sol"],
        min_pressure_level_coverage_frac=primary["min_pressure_level_coverage_frac"],
    )
    coverage_sens = pressure_coverage_sensitivity(
        aggregated_primary,
        coverage_values=sensitivity["min_pressure_level_coverage_frac"],
        min_obs_per_cell_sol=primary["min_obs_per_cell_sol"],
    )

    return {
        "grid_resolution_sensitivity": grid_sens,
        "min_obs_sensitivity": min_obs_sens,
        "pressure_coverage_sensitivity": coverage_sens,
    }
