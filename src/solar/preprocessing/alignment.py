"""Disk-level alignment: turns QC'd, per-satellite GOES rows and per-active-
region SHARP rows into ONE common timeline at SHARP's native cadence.

See configs/alignment.yaml's header comment for why this is DISK-INTEGRATED
(no real active-region-to-flare attribution catalogue is ingested in this
project) rather than per-active-region. Three steps, each strictly causal:

  1. `aggregate_goes_disk` -- collapse multiple satellites reporting at the
     SAME 1-min timestamp into one full-disk flux value..
  2. `aggregate_sharp_disk` -- collapse multiple active regions (HARPNUMs)
     tracked at the SAME T_REC into disk-wide summary statistics.
  3. `attach_goes_trailing_state` -- for each SHARP-cadence timestamp,
     summarize the native 1-min GOES disk series over the trailing window
     ending AT (and including) that timestamp -- never a future GOES
     observation.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR


def load_alignment_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "alignment.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _parse_tai_string(t_rec: str) -> pd.Timestamp:
    """Parses JSOC's "YYYY.MM.DD_HH:MM:SS_TAI" format. APPROXIMATION: treats
    TAI as UTC, ignoring the ~37s leap-second offset between them --
    immaterial at 12-minute alignment granularity, documented explicitly
    rather than silently assumed correct to sub-minute precision."""
    cleaned = str(t_rec).replace("_TAI", "").replace("TAI", "")
    return pd.Timestamp(datetime.strptime(cleaned, "%Y.%m.%d_%H:%M:%S"), tz="UTC")


def aggregate_goes_disk(qc_accepted_goes_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Collapses rows from multiple satellites reporting at the same
    time_utc into one full-disk flux value per timestamp."""
    config = config or load_alignment_config()
    method = config["goes_satellite_aggregation_method"]
    if method not in ("max", "mean"):
        raise ValueError(f"goes_satellite_aggregation_method must be 'max' or 'mean', got {method!r}")

    df = qc_accepted_goes_df.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    grouped = (
        df.groupby("time_utc")
        .agg(
            xrsa_flux_w_m2=("xrsa_flux_w_m2", method),
            xrsb_flux_w_m2=("xrsb_flux_w_m2", method),
            n_satellites_reporting=("satellite", "nunique"),
        )
        .reset_index()
        .sort_values("time_utc")
        .reset_index(drop=True)
    )
    return grouped


def aggregate_sharp_disk(qc_accepted_sharp_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Collapses rows from multiple active regions (HARPNUMs) tracked at
    the same T_REC into disk-wide summary statistics, per-keyword
    aggregation method from configs/alignment.yaml's `sharp_keyword_aggregation`."""
    config = config or load_alignment_config()
    time_col = config["sharp_time_column"]
    agg_map = config["sharp_keyword_aggregation"]

    df = qc_accepted_sharp_df.copy()
    df["time_utc"] = df[time_col].apply(_parse_tai_string)

    agg_kwargs = {kw: (kw, method) for kw, method in agg_map.items() if kw in df.columns}
    grouped = (
        df.groupby("time_utc")
        .agg(n_active_regions_tracked=("HARPNUM", "nunique"), **agg_kwargs)
        .reset_index()
        .sort_values("time_utc")
        .reset_index(drop=True)
    )
    return grouped


def attach_goes_trailing_state(
    sharp_disk_df: pd.DataFrame, goes_disk_df: pd.DataFrame, config: dict | None = None
) -> pd.DataFrame:
    """For each SHARP-cadence row, summarizes the native GOES disk series
    over the trailing window ending AT (and including) that timestamp.
    Strictly causal: a GOES observation after the SHARP timestamp is never
    included, via searchsorted on sorted time arrays."""
    config = config or load_alignment_config()
    window = pd.Timedelta(minutes=config["goes_trailing_window_minutes"])

    goes_sorted = goes_disk_df.sort_values("time_utc").reset_index(drop=True)
    goes_times = goes_sorted["time_utc"].to_numpy()
    xrsa = goes_sorted["xrsa_flux_w_m2"].to_numpy(dtype=float)
    xrsb = goes_sorted["xrsb_flux_w_m2"].to_numpy(dtype=float)

    out = sharp_disk_df.sort_values("time_utc").reset_index(drop=True).copy()
    n = len(out)
    xrsa_max = np.full(n, np.nan)
    xrsa_mean = np.full(n, np.nan)
    xrsb_max = np.full(n, np.nan)
    xrsb_mean = np.full(n, np.nan)
    n_obs = np.zeros(n, dtype=int)

    sharp_times = out["time_utc"].to_numpy()
    window_start = (pd.Series(sharp_times) - window).to_numpy()
    # side="left" so a GOES timestamp exactly AT window_start is included --
    # the window is documented as inclusive of its start (and of its end,
    # via hi_idx's side="right" below).
    lo_idx = np.searchsorted(goes_times, window_start, side="left")
    hi_idx = np.searchsorted(goes_times, sharp_times, side="right")

    for i in range(n):
        lo, hi = lo_idx[i], hi_idx[i]
        if hi > lo:
            a, b = xrsa[lo:hi], xrsb[lo:hi]
            xrsa_max[i], xrsa_mean[i] = float(np.nanmax(a)), float(np.nanmean(a))
            xrsb_max[i], xrsb_mean[i] = float(np.nanmax(b)), float(np.nanmean(b))
            n_obs[i] = hi - lo

    out["goes_xrsa_flux_trailing_max_w_m2"] = xrsa_max
    out["goes_xrsa_flux_trailing_mean_w_m2"] = xrsa_mean
    out["goes_xrsb_flux_trailing_max_w_m2"] = xrsb_max
    out["goes_xrsb_flux_trailing_mean_w_m2"] = xrsb_mean
    out["goes_n_obs_in_trailing_window"] = n_obs
    return out


def build_aligned_timeline(
    qc_accepted_goes_df: pd.DataFrame, qc_accepted_sharp_df: pd.DataFrame, config: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: aggregate both domains to disk level, then attach GOES
    trailing state onto the SHARP-cadence timeline. Returns (aligned,
    goes_disk_series) -- goes_disk_series (native 1-min disk-aggregated
    GOES) is also returned since labeling/flare_labels.py needs the
    fine-grained series, not the SHARP-coarsened one, to avoid missing a
    flare peak that falls between two SHARP timestamps."""
    config = config or load_alignment_config()
    goes_disk = aggregate_goes_disk(qc_accepted_goes_df, config)
    sharp_disk = aggregate_sharp_disk(qc_accepted_sharp_df, config)
    aligned = attach_goes_trailing_state(sharp_disk, goes_disk, config)
    return aligned, goes_disk
