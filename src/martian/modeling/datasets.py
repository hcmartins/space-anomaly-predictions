"""Per-task dataset/target construction for the two fully-separated tasks
(dust_regression, storm_detection -- temperature_regression was descoped, no
real climatological source exists for surface_temperature_k). Each task's
target is built independently, with a unique
`target_<...>_h<horizon>` column name, from a GENUINELY OBSERVED future
cell-sol only (never a trailing-filled/fabricated one). This module is the
only place task targets are constructed -- preprocessing/features.py never
produces a `target_`-prefixed column, so a task's feature manifest can never
accidentally include another task's (or its own) target.

Every task's usable rows are restricted to cell-sols that are genuine
observations (present in `labeled_df`, which already excludes trailing-filled
rows -- see labeling/anomaly_labels.py's `_exclude_interpolated_rows`), even
though their FEATURES may draw on trailing-filled history for density. This
means: you can only forecast FROM a moment you actually measured, using
whatever historical context (real or gap-filled) is available -- but you can
never forecast FROM a fabricated moment, and you can never be SCORED against
a fabricated future value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing.features import feature_column_names

CURRENT_STATE_RAW_COLUMNS = [
    "dust_optical_depth", "dust_optical_depth_z",
    "dust_anomaly", "atmospheric_anomaly",
]

TASK_TARGET_PREFIXES = {
    "dust_regression": "target_dust_optical_depth_h",
    "storm_detection": "target_storm_horizon_h",
}


def _exact_future_lookup(
    base_df: pd.DataFrame,
    real_source_df: pd.DataFrame,
    *,
    value_col: str,
    horizon: int,
    cell_col: str = "cell_id",
    mars_year_col: str = "mars_year",
    sol_col: str = "sol",
) -> pd.Series:
    """target(c,s) = real_source_df's value_col at (c, mars_year, s+horizon)
    if that EXACT sol exists as a genuine observation there, else NaN."""
    lookup: dict[tuple[int, int, int], int] = {}
    for idx, cid, my, sol in real_source_df[[cell_col, mars_year_col, sol_col]].itertuples(name=None):
        lookup[(int(cid), int(my), int(sol))] = idx

    values = np.full(len(base_df), np.nan)
    for row_pos, (cid, my, sol) in enumerate(
        base_df[[cell_col, mars_year_col, sol_col]].itertuples(index=False, name=None)
    ):
        future_idx = lookup.get((int(cid), int(my), int(sol) + horizon))
        if future_idx is not None:
            values[row_pos] = real_source_df.at[future_idx, value_col]
    return pd.Series(values, index=base_df.index)


def _restrict_to_observed_base_rows(
    features_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    *,
    extra_cols: list[str],
    cell_col: str = "cell_id",
    mars_year_col: str = "mars_year",
    sol_col: str = "sol",
) -> pd.DataFrame:
    key_cols = [cell_col, mars_year_col, sol_col]
    right = labeled_df[key_cols + extra_cols].drop_duplicates(subset=key_cols)
    return features_df.merge(right, on=key_cols, how="inner")


_BOOKKEEPING_METADATA_COLUMNS = [
    "solar_longitude_ls", "dust_optical_depth_mu",
    # Carried for evaluation/failure_analysis.py's breakdown dimensions --
    # never model features (excluded from feature_manifest), just metadata.
    "latitude", "local_true_solar_time", "observation_count",
    "uncertainty_summary", "interpolation_flag", "is_observed", "low_support_flag",
    "episode_id",
]


def _assemble(base_df: pd.DataFrame, feature_manifest: list[str], target_cols: list[str]) -> pd.DataFrame:
    # solar_longitude_ls / *_mu are carried as BOOKKEEPING METADATA (for the
    # seasonal-mean and climatology BASELINES -- see modeling/baselines.py),
    # never as a model feature -- they are deliberately excluded from
    # feature_manifest (the harmonics feat_ls_sin/feat_ls_cos are the
    # model-facing encoding of season instead).
    id_cols = ["cell_id", "mars_year", "sol"] + [c for c in _BOOKKEEPING_METADATA_COLUMNS if c in base_df.columns]
    dataset = base_df[id_cols + feature_manifest + target_cols].copy()
    for col in CURRENT_STATE_RAW_COLUMNS:
        if col in dataset.columns and dataset[col].dtype == bool:
            dataset[col] = dataset[col].astype(int)
    return dataset


def build_dust_regression_dataset(
    aggregated_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    horizons_sols: tuple[int, ...] = (1, 3, 5),
) -> tuple[pd.DataFrame, list[str]]:
    base = _restrict_to_observed_base_rows(features_df, labeled_df, extra_cols=[])
    feature_manifest = feature_column_names(base) + [c for c in CURRENT_STATE_RAW_COLUMNS if c in base.columns]

    target_cols = []
    for h in horizons_sols:
        col = f"{TASK_TARGET_PREFIXES['dust_regression']}{h}"
        base[col] = _exact_future_lookup(base, aggregated_df, value_col="dust_optical_depth", horizon=h)
        target_cols.append(col)

    return _assemble(base, feature_manifest, target_cols), feature_manifest


def build_storm_detection_dataset(
    labeled_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    horizons_sols: tuple[int, ...] = (1, 3, 5),
) -> tuple[pd.DataFrame, list[str]]:
    horizon_cols = [f"horizon_{h}sol" for h in horizons_sols]
    extra_cols = horizon_cols + (["episode_id"] if "episode_id" in labeled_df.columns else [])
    base = _restrict_to_observed_base_rows(features_df, labeled_df, extra_cols=extra_cols)
    feature_manifest = feature_column_names(base) + [c for c in CURRENT_STATE_RAW_COLUMNS if c in base.columns]

    target_cols = []
    for h in horizons_sols:
        src_col = f"horizon_{h}sol"
        col = f"{TASK_TARGET_PREFIXES['storm_detection']}{h}"
        # float with NaN = undefined target (no observed sol in the horizon);
        # dropped per horizon by modeling.train._horizon_frame, never filled
        base[col] = base[src_col].astype(float)
        target_cols.append(col)

    return _assemble(base, feature_manifest, target_cols), feature_manifest
