"""Causal (trailing-only) gap-filling for the disk-integrated, SHARP-cadence
aligned series (preprocessing/alignment.py's `build_aligned_timeline` output).

Unlike martian/preprocessing/missing_data.py -- which fills independently
within each (cell_id, mars_year) group on a spatial grid -- this operates on
a SINGLE global disk-wide time series, so no grouping is needed: the series
is reindexed onto a regular cadence_minutes grid spanning its own observed
range, and each step is filled by last-observation-carried-forward (LOCF)
from the most recent PRIOR real observation, bounded by
max_trailing_fill_gap_steps. A step whose gap since the last real
observation exceeds that bound is dropped entirely, never fabricated --
matching this project's "do not interpolate a row into existence when no
defensible observation is available" convention (see
planning/PREPROCESSING_DOCUMENTATION.md).

Every retained row carries `is_observed`, `gap_length_steps` (0 for a
genuine observation, NaN if there is no prior observation to carry forward
at all), and `interpolation_flag` (True only for a filled row) so downstream
code can always distinguish real from carried-forward data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR


def load_missing_data_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "missing_data.yaml"
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
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


def apply_trailing_fill(
    aligned_df: pd.DataFrame,
    *,
    max_gap_steps: int | None = None,
    config: dict | None = None,
    time_col: str = "time_utc",
) -> pd.DataFrame:
    """Reindexes aligned_df onto a regular cadence_minutes grid and fills
    short gaps by LOCF, bounded by max_gap_steps. Returns a dataframe with
    `is_observed`, `gap_length_steps`, and `interpolation_flag` columns;
    rows for gaps longer than max_gap_steps (or before the first real
    observation) are absent from the output, not fabricated."""
    config = config or load_missing_data_config()
    max_gap_steps = max_gap_steps if max_gap_steps is not None else config["max_trailing_fill_gap_steps"]
    cadence = pd.Timedelta(minutes=config["cadence_minutes"])
    fill_variables = [c for c in config["fill_variables"] if c in aligned_df.columns]

    df = aligned_df.sort_values(time_col).reset_index(drop=True).copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    if df.empty:
        out = df.copy()
        out["is_observed"] = pd.Series(dtype=bool)
        out["gap_length_steps"] = pd.Series(dtype=float)
        out["interpolation_flag"] = pd.Series(dtype=bool)
        return out

    df["_is_observed_marker"] = True

    full_index = pd.date_range(df[time_col].min(), df[time_col].max(), freq=cadence, tz="UTC")
    reindexed = df.set_index(time_col).reindex(full_index)
    reindexed.index.name = time_col

    is_observed = reindexed["_is_observed_marker"].fillna(False).astype(bool)
    reindexed = reindexed.drop(columns=["_is_observed_marker"])

    step_idx = np.arange(len(full_index))
    observed_step = np.where(is_observed.to_numpy(), step_idx, np.nan)
    last_observed_step = pd.Series(observed_step, index=full_index).ffill().to_numpy()
    with np.errstate(invalid="ignore"):
        gap_length = step_idx - last_observed_step
    gap_length = np.where(np.isnan(last_observed_step), np.inf, gap_length)

    keep_mask = is_observed.to_numpy() | (gap_length <= max_gap_steps)

    for col in fill_variables:
        reindexed[col] = reindexed[col].ffill()

    reindexed["is_observed"] = is_observed
    reindexed["gap_length_steps"] = np.where(np.isfinite(gap_length), gap_length, np.nan)
    reindexed["interpolation_flag"] = (~is_observed.to_numpy()) & keep_mask

    result = reindexed.loc[keep_mask].reset_index().rename(columns={"index": time_col})
    return result


def run_missing_data_sensitivity(aligned_df: pd.DataFrame, *, config: dict | None = None) -> pd.DataFrame:
    """Sweeps configs/missing_data.yaml's sensitivity.max_trailing_fill_gap_steps,
    reporting observed/filled/total counts at each threshold."""
    config = config or load_missing_data_config()
    rows = []
    for max_gap in config["sensitivity"]["max_trailing_fill_gap_steps"]:
        filled = apply_trailing_fill(aligned_df, max_gap_steps=max_gap, config=config)
        n_observed = int(filled["is_observed"].sum()) if len(filled) else 0
        n_filled = int(filled["interpolation_flag"].sum()) if len(filled) else 0
        rows.append(
            {
                "max_trailing_fill_gap_steps": max_gap,
                "n_observed": n_observed,
                "n_filled": n_filled,
                "n_total": n_observed + n_filled,
            }
        )
    return pd.DataFrame(rows)
