"""Causal (trailing-only) missing-data handling for cell-sol aggregated data.

Processes each (cell_id, mars_year) sol-series strictly chronologically
(sol-ascending). A missing sol is filled ONLY by carrying the most recent
PRIOR observation forward (last-observation-carried-forward), and only for
columns declared in configs/missing_data.yaml's `fill_variables` list
(forecasting inputs). Labels are never listed there and are therefore never
filled -- see the reserved-pattern guard in `_validate_fill_variables`.

If the gap since the last real observation exceeds `max_trailing_fill_gap_sols`,
the cell-sol is NOT fabricated: it is simply absent from the output, per the
task requirement "do not interpolate a cell-sol into existence when no
defensible observation is available." Every retained filled row carries
`gap_length_sols` (0 for a genuine observation) and `interpolation_flag`
(True only for a filled row) so downstream code can always distinguish real
from carried-forward data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR

from .aggregation import ATM_LEVEL_PREFIX, compute_low_support_flag, load_aggregation_config

_STRUCTURAL_CARRY_COLUMNS = ("data_source",)  # bookkeeping, always carried forward
FILLED_DATA_SOURCE_SUFFIX = "-TRAILING-FILLED"


def load_missing_data_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "missing_data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_fill_variables(config["fill_variables"], config["reserved_label_patterns"])
    return config


def _validate_fill_variables(fill_variables: list[str], reserved_patterns: list[str]) -> None:
    for var in fill_variables:
        for pattern in reserved_patterns:
            if pattern in var:
                raise ValueError(
                    f"fill_variables contains {var!r}, which matches the reserved "
                    f"label-column pattern {pattern!r}. Labels must never be "
                    f"trailing-filled or interpolated -- see configs/missing_data.yaml."
                )


def _expand_fill_variables(df: pd.DataFrame, configured: list[str]) -> list[str]:
    """Add present atm_temp_k_level_NN (+ uncertainty) columns programmatically,
    mirroring the pattern used by configs/aggregation.yaml, so the 14-level
    list is never duplicated by hand."""
    atm_cols = [c for c in df.columns if c.startswith(ATM_LEVEL_PREFIX)]
    expanded = [v for v in configured if v in df.columns] + [c for c in atm_cols if c not in configured]
    return expanded


def _fill_one_series(
    group: pd.DataFrame,
    *,
    group_cols: tuple[str, ...],
    sol_col: str,
    fill_variables: list[str],
    max_gap_sols: int,
) -> pd.DataFrame:
    group = group.sort_values(sol_col)
    group_key_values = {c: group[c].iloc[0] for c in group_cols}

    observed_sols_arr = group[sol_col].to_numpy(dtype=int)
    observed_sols = set(int(s) for s in observed_sols_arr)

    # Only materialise sols that are either real observations or within
    # max_gap_sols of the immediately PRECEDING real observation -- i.e.
    # never the full [min_sol, max_sol] span. A gap wider than max_gap_sols
    # is never even constructed as a candidate row, let alone filled; this
    # keeps the per-group work proportional to the number of real
    # observations rather than to how many sols the series happens to span
    # (important since a sparsely-observed cell can span most of a Mars
    # year while containing very few real retrievals).
    allowed_sols = set(observed_sols)
    for a, b in zip(observed_sols_arr[:-1], observed_sols_arr[1:]):
        gap = int(b - a)
        if gap > 1:
            for k in range(1, min(gap - 1, max_gap_sols) + 1):
                allowed_sols.add(int(a) + k)

    full_index = pd.Index(sorted(allowed_sols), name=sol_col)
    reindexed = group.set_index(sol_col).reindex(full_index)

    is_observed = pd.Series(full_index.isin(observed_sols), index=full_index)

    # Sol of the most recent PRIOR (or current) real observation, carried
    # forward strictly chronologically -- never looking ahead.
    last_observed_sol = pd.Series(
        np.where(is_observed, full_index.to_numpy(), np.nan), index=full_index
    ).ffill()
    gap_length = pd.Series(full_index.to_numpy(), index=full_index) - last_observed_sol
    # Sols before the first real observation in this series have no prior
    # observation to carry forward at all -- never fillable, regardless of
    # max_gap_sols.
    gap_length = gap_length.where(last_observed_sol.notna(), np.inf)

    # Always true by construction (allowed_sols only ever contains sols that
    # satisfy this), recomputed explicitly as a safety net rather than
    # assumed.
    keep_mask = is_observed | (gap_length <= max_gap_sols)

    for col in fill_variables:
        if col in reindexed.columns:
            reindexed[col] = reindexed[col].ffill()
    for col in _STRUCTURAL_CARRY_COLUMNS:
        if col in reindexed.columns:
            reindexed[col] = reindexed[col].ffill()
    for col, value in group_key_values.items():
        reindexed[col] = value

    reindexed["is_observed"] = is_observed
    reindexed["gap_length_sols"] = gap_length.where(np.isfinite(gap_length), np.nan)
    reindexed["interpolation_flag"] = (~is_observed) & keep_mask

    if "observation_count" in reindexed.columns:
        reindexed.loc[reindexed["interpolation_flag"], "observation_count"] = 0

    if "data_source" in reindexed.columns:
        filled_mask = reindexed["interpolation_flag"]
        reindexed.loc[filled_mask, "data_source"] = (
            reindexed.loc[filled_mask, "data_source"].astype(str) + FILLED_DATA_SOURCE_SUFFIX
        )

    return reindexed.loc[keep_mask].reset_index()


def apply_trailing_fill(
    aggregated_df: pd.DataFrame,
    *,
    max_gap_sols: int,
    config: dict | None = None,
    group_cols: tuple[str, ...] = ("cell_id", "mars_year"),
    sol_col: str = "sol",
) -> pd.DataFrame:
    """Trailing-fill short gaps in a cell-sol aggregated dataframe.

    Processes each (cell_id, mars_year) group strictly chronologically
    (sol-ascending). Returns a dataframe with `is_observed`, `gap_length_sols`,
    and `interpolation_flag` columns; `low_support_flag` is recomputed to
    reflect the post-fill `observation_count` (0 for filled rows).
    """
    config = config or load_missing_data_config()
    fill_variables = _expand_fill_variables(aggregated_df, config["fill_variables"])

    groups = [
        _fill_one_series(
            group,
            group_cols=group_cols,
            sol_col=sol_col,
            fill_variables=fill_variables,
            max_gap_sols=max_gap_sols,
        )
        for _, group in aggregated_df.groupby(list(group_cols), sort=False)
    ]
    result = (
        pd.concat(groups, ignore_index=True)
        if groups
        else aggregated_df.assign(is_observed=pd.Series(dtype=bool), gap_length_sols=pd.Series(dtype=float), interpolation_flag=pd.Series(dtype=bool))
    )

    if len(result) and "pressure_level_coverage_frac" in result.columns and "observation_count" in result.columns:
        agg_config = load_aggregation_config()
        result["low_support_flag"] = compute_low_support_flag(
            result,
            min_obs_per_cell_sol=agg_config["primary"]["min_obs_per_cell_sol"],
            min_pressure_level_coverage_frac=agg_config["primary"]["min_pressure_level_coverage_frac"],
        )

    return result


def run_missing_data_sensitivity(
    aggregated_df: pd.DataFrame,
    *,
    config: dict | None = None,
) -> pd.DataFrame:
    """Sweep max_trailing_fill_gap_sols over configs/missing_data.yaml's
    sensitivity list, reporting observed/filled/total counts at each
    threshold."""
    config = config or load_missing_data_config()
    rows = []
    for max_gap in config["sensitivity"]["max_trailing_fill_gap_sols"]:
        filled = apply_trailing_fill(aggregated_df, max_gap_sols=max_gap, config=config)
        n_observed = int(filled["is_observed"].sum()) if len(filled) else 0
        n_filled = int(filled["interpolation_flag"].sum()) if len(filled) else 0
        rows.append(
            {
                "max_trailing_fill_gap_sols": max_gap,
                "n_observed": n_observed,
                "n_filled": n_filled,
                "n_total": n_observed + n_filled,
            }
        )
    return pd.DataFrame(rows)
