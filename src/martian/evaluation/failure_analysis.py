"""Failure analysis: breaks down a task's per-row metric performance across
ten dimensions -- season, latitude band, local time, retrieval density,
interpolation status, uncertainty, event category, forecast horizon, label
persistence, and spatial extent.

Not every dimension is meaningful for every task -- e.g. "interpolation
status" is degenerate for all three tasks (every evaluated row is, by
construction, a genuinely observed current sol -- see
modeling/datasets.py's `_restrict_to_observed_base_rows`), and "event
category"/"label persistence"/"spatial extent" only make sense for Task C
(storm_detection), operationalised here as the CURRENT sol's own episode
membership (if any), not the future episode(s) that may drive a positive
horizon target. "Forecast horizon" is satisfied by evaluate_task.py running
this whole module once per horizon, not as an additional binning column
within a single horizon's report. Every function below is honest about
degenerate/inapplicable cases -- it reports a single-category breakdown or
an explicit "not applicable" note rather than fabricating variation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from labeling.reporting import assign_mars_season


def _metric_by_group(
    df: pd.DataFrame, group: pd.Series, *, y_true_col: str, y_pred_col: str, metric_fn
) -> pd.DataFrame:
    rows = []
    for label, idx in df.groupby(group, dropna=False).groups.items():
        subset = df.loc[idx]
        n = len(subset)
        row = {"group": label, "n": n}
        if n > 0:
            row.update(metric_fn(subset[y_true_col].to_numpy(), subset[y_pred_col].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


_NOT_APPLICABLE = pd.DataFrame(columns=["group", "n"])


def breakdown_by_season(df: pd.DataFrame, *, y_true_col: str, y_pred_col: str, metric_fn) -> pd.DataFrame:
    if "solar_longitude_ls" not in df.columns:
        return _NOT_APPLICABLE.copy()
    season = pd.Series(assign_mars_season(df["solar_longitude_ls"]), index=df.index)
    return _metric_by_group(df, season, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn)


def breakdown_by_latitude_band(
    df: pd.DataFrame, *, y_true_col: str, y_pred_col: str, metric_fn, band_width_deg: float = 30.0
) -> pd.DataFrame:
    if "latitude" not in df.columns:
        return _NOT_APPLICABLE.copy()
    band = (np.floor((df["latitude"] + 90.0) / band_width_deg) * band_width_deg - 90.0).astype(int)
    band_label = band.astype(str) + " to " + (band + int(band_width_deg)).astype(str) + " deg"
    return _metric_by_group(df, band_label, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn)


def breakdown_by_local_time(
    df: pd.DataFrame, *, y_true_col: str, y_pred_col: str, metric_fn, bin_width_hours: float = 6.0
) -> pd.DataFrame:
    if "local_true_solar_time" not in df.columns:
        return _NOT_APPLICABLE.copy()
    bin_idx = np.floor(df["local_true_solar_time"] / bin_width_hours) * bin_width_hours
    label = bin_idx.astype(str) + "-" + (bin_idx + bin_width_hours).astype(str) + "h"
    return _metric_by_group(df, label, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn)


def breakdown_by_retrieval_density(
    df: pd.DataFrame, *, y_true_col: str, y_pred_col: str, metric_fn, bin_edges: list[int] = (1, 2, 3, 5)
) -> pd.DataFrame:
    if "observation_count" not in df.columns:
        return _NOT_APPLICABLE.copy()
    labels = [f"{bin_edges[i]}-{bin_edges[i+1]-1}" for i in range(len(bin_edges) - 1)] + [f"{bin_edges[-1]}+"]
    category = pd.cut(
        df["observation_count"], bins=[*bin_edges, np.inf], labels=labels, right=False, include_lowest=True
    )
    return _metric_by_group(df, category, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn)


def breakdown_by_interpolation_status(df: pd.DataFrame, *, y_true_col: str, y_pred_col: str, metric_fn) -> pd.DataFrame:
    """Degenerate by construction for every task (see module docstring) --
    still computed and reported plainly, never hidden."""
    if "is_observed" in df.columns:
        status = df["is_observed"].map({True: "observed", False: "interpolated"})
    else:
        status = pd.Series(["observed"] * len(df), index=df.index)
    return _metric_by_group(df, status, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn)


def breakdown_by_uncertainty(
    df: pd.DataFrame, *, y_true_col: str, y_pred_col: str, metric_fn, n_bins: int = 4
) -> pd.DataFrame:
    if "uncertainty_summary" not in df.columns or df["uncertainty_summary"].notna().sum() < n_bins:
        return pd.DataFrame(columns=["group", "n"])
    quantile_bin = pd.qcut(df["uncertainty_summary"], q=n_bins, duplicates="drop")
    return _metric_by_group(df, quantile_bin, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn)


def breakdown_by_event_category(
    df: pd.DataFrame, episode_catalogue: pd.DataFrame | None, *, y_true_col: str, y_pred_col: str, metric_fn
) -> pd.DataFrame:
    """Task C only -- category of the episode active at the CURRENT sol (not
    a future episode driving the horizon target). Rows with no active
    episode fall in the "quiet" category."""
    if "episode_id" not in df.columns or episode_catalogue is None or episode_catalogue.empty:
        return pd.DataFrame(columns=["group", "n"])
    id_to_class = dict(zip(episode_catalogue["episode_id"], episode_catalogue.get("taxonomy_class", "unclassified")))
    category = df["episode_id"].map(id_to_class).fillna("quiet")
    return _metric_by_group(df, category, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn)


def breakdown_by_label_persistence(
    df: pd.DataFrame, episode_catalogue: pd.DataFrame | None, *, y_true_col: str, y_pred_col: str, metric_fn
) -> pd.DataFrame:
    """Task C only -- duration_sols of the episode active at the current
    sol; "quiet" for rows with none."""
    if "episode_id" not in df.columns or episode_catalogue is None or episode_catalogue.empty:
        return pd.DataFrame(columns=["group", "n"])
    id_to_duration = dict(zip(episode_catalogue["episode_id"], episode_catalogue.get("duration_sols")))
    duration = df["episode_id"].map(id_to_duration)
    category = duration.apply(lambda d: "quiet" if pd.isna(d) else f"{int(d)}sol")
    return _metric_by_group(df, category, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn)


def breakdown_by_spatial_extent(
    df: pd.DataFrame, episode_catalogue: pd.DataFrame | None, *, y_true_col: str, y_pred_col: str, metric_fn
) -> pd.DataFrame:
    """Task C only -- peak_area_cells of the episode active at the current
    sol; "quiet" for rows with none."""
    if "episode_id" not in df.columns or episode_catalogue is None or episode_catalogue.empty:
        return pd.DataFrame(columns=["group", "n"])
    id_to_area = dict(zip(episode_catalogue["episode_id"], episode_catalogue.get("peak_area_cells")))
    area = df["episode_id"].map(id_to_area)
    category = area.apply(lambda a: "quiet" if pd.isna(a) else f"{int(a)}cell(s)")
    return _metric_by_group(df, category, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn)


def run_all_breakdowns(
    df: pd.DataFrame,
    *,
    y_true_col: str,
    y_pred_col: str,
    metric_fn,
    episode_catalogue: pd.DataFrame | None = None,
    config: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Runs every applicable breakdown, returning {dimension_name: table}.
    "forecast_horizon" and (for Tasks A/B) "event_category"/
    "label_persistence"/"spatial_extent" are intentionally absent here --
    see the module docstring for why."""
    config = config or {}
    fa_config = config.get("failure_analysis", {})

    breakdowns = {
        "season": breakdown_by_season(df, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn),
        "latitude_band": breakdown_by_latitude_band(
            df, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn,
            band_width_deg=fa_config.get("latitude_band_width_deg", 30.0),
        ),
        "local_time": breakdown_by_local_time(
            df, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn,
            bin_width_hours=fa_config.get("local_time_bin_width_hours", 6.0),
        ),
        "retrieval_density": breakdown_by_retrieval_density(
            df, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn,
            bin_edges=fa_config.get("retrieval_density_bins", [1, 2, 3, 5]),
        ),
        "interpolation_status": breakdown_by_interpolation_status(
            df, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn
        ),
        "uncertainty": breakdown_by_uncertainty(
            df, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn,
            n_bins=fa_config.get("uncertainty_bins", 4),
        ),
    }

    if episode_catalogue is not None:
        breakdowns["event_category"] = breakdown_by_event_category(
            df, episode_catalogue, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn
        )
        breakdowns["label_persistence"] = breakdown_by_label_persistence(
            df, episode_catalogue, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn
        )
        breakdowns["spatial_extent"] = breakdown_by_spatial_extent(
            df, episode_catalogue, y_true_col=y_true_col, y_pred_col=y_pred_col, metric_fn=metric_fn
        )

    return breakdowns
