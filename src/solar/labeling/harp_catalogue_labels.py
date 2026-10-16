"""Per-active-region (HARP-level) forward-looking flare-occurrence
labeling, built from Phase 1's real catalogue attribution
(preprocessing/harp_attribution.py) instead of labeling/flare_labels.py's
disk-wide GOES-threshold proxy..

Kept as a SEPARATE module from flare_labels.py rather than folded in:
the labeling *logic* is fundamentally different (catalogue-attributed vs.
GOES-threshold), and this project's convention throughout is not to
conflate structurally different approaches in one file (mirrors why
preprocessing/ never imports labeling/ -- see flare_labels.py's own
docstring). Reuses flare_labels.py's classify_flux/_class_at_or_above/
LABEL_PREFIX/_CLASS_ORDER and configs/labeling.yaml's thresholds
directly (imported, not duplicated) -- only the WINDOW-SEARCH mechanics
differ (per real HARP-attributed flares instead of the disk-wide
fine-grained GOES series).

Unassigned-flare masking (Chapter 3.2.1's own text: "unattributable
events can neither create positives nor contaminate negatives") is
applied PER CLASS THRESHOLD, not blanket: an unassigned flare's own class
IS known from the catalogue even though its region isn't, so e.g. a
C-class unassigned flare can only ever mask C-or-above's would-be-negative
rows -- masking M-or-above on a C-class event that provably never reached
M-class, regardless of which HARP it came from, would throw away good
negative-label information for no reason. Masking never overrides an
already-genuine positive from a real attributed flare for that same HARP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .flare_labels import LABEL_PREFIX, _CLASS_ORDER, _class_at_or_above, classify_flux, load_labeling_config


def _max_flux_in_windows(
    sorted_times: np.ndarray, sorted_flux: np.ndarray, t0: np.ndarray, t1: np.ndarray
) -> np.ndarray:
    """For each i, the max of sorted_flux among sorted_times in
    (t0[i], t1[i]], or NaN if none. sorted_times/sorted_flux must already
    be sorted ascending by time. Vectorized bounds via searchsorted; the
    per-row max itself is a small Python loop (mirrors
    flare_labels.py::build_labels's identical pattern) -- benchmarked at
    real per-HARP scale before being trusted as fast enough (see
    planning/FLARE_CATALOGUE_DOCUMENTATION.md's Phase 2 section)."""
    n = len(t0)
    out = np.full(n, np.nan)
    if len(sorted_times) == 0:
        return out
    lo_idx = np.searchsorted(sorted_times, t0, side="right")
    hi_idx = np.searchsorted(sorted_times, t1, side="right")
    for i in range(n):
        lo, hi = lo_idx[i], hi_idx[i]
        if hi > lo:
            out[i] = float(np.nanmax(sorted_flux[lo:hi]))
    return out


def build_harp_catalogue_labels(
    per_harp_aligned_df: pd.DataFrame,
    attributed_flares_df: pd.DataFrame,
    *,
    horizon_hours: float | None = None,
    config: dict | None = None,
    time_col: str = "time_utc",
) -> pd.DataFrame:
    """For each (HARPNUM, t0) row in per_harp_aligned_df (from
    preprocessing/harp_alignment.py), labels whether a REAL flare
    attributed to that HARPNUM (attributed_flares_df, from
    preprocessing/harp_attribution.py::attribute_flares_to_harps) peaks
    in (t0, t0+H]. A flare attributed to multiple HARPNUMs (real, ~1.0%
    of cases -- see planning/FLARE_CATALOGUE_DOCUMENTATION.md) credits
    every one of them, not just one arbitrarily.

    No label_coverage_fraction column (unlike flare_labels.py's
    GOES-gap-based one) -- the uncertainty here is catalogue completeness
    (an unassigned flare), a fundamentally different thing, handled via
    per-class masking to NaN instead of a coverage fraction."""
    config = config or load_labeling_config()
    thresholds = config["flare_class_thresholds_w_m2"]
    horizon = pd.Timedelta(hours=horizon_hours if horizon_hours is not None else config["forecast_horizon_hours"])

    out = per_harp_aligned_df.sort_values(["HARPNUM", time_col]).reset_index(drop=True).copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True)
    t0 = out[time_col].to_numpy()
    t1 = (out[time_col] + horizon).to_numpy()
    n = len(out)

    assigned = attributed_flares_df.loc[attributed_flares_df["harpnums"].apply(bool)].copy()
    exploded = assigned.explode("harpnums").rename(columns={"harpnums": "HARPNUM"})
    if len(exploded):
        exploded["HARPNUM"] = exploded["HARPNUM"].astype(int)
        exploded = exploded.sort_values("time_utc_peak")

    unassigned = attributed_flares_df.loc[~attributed_flares_df["harpnums"].apply(bool)].sort_values("time_utc_peak")
    unassigned_times = unassigned["time_utc_peak"].to_numpy()
    unassigned_flux = unassigned["xrsb_irrad_peak_w_m2"].to_numpy(dtype=float)

    attributed_max_flux = np.full(n, np.nan)
    for harpnum, idx in out.groupby("HARPNUM").indices.items():
        harp_flares = exploded.loc[exploded["HARPNUM"] == harpnum] if len(exploded) else exploded
        flare_times = harp_flares["time_utc_peak"].to_numpy() if len(harp_flares) else np.array([])
        flare_flux = harp_flares["xrsb_irrad_peak_w_m2"].to_numpy(dtype=float) if len(harp_flares) else np.array([])
        attributed_max_flux[idx] = _max_flux_in_windows(flare_times, flare_flux, t0[idx], t1[idx])

    unassigned_max_flux = _max_flux_in_windows(unassigned_times, unassigned_flux, t0, t1)

    max_class = np.array([classify_flux(f, thresholds) for f in attributed_max_flux], dtype=object)
    out[f"{LABEL_PREFIX}max_flare_class_in_harp_horizon"] = max_class

    for cls in _CLASS_ORDER:
        positive = np.array([_class_at_or_above(c, cls) for c in max_class])
        # Masked (UNDEFINED, never a fabricated 0): this class's threshold
        # would be a NEGATIVE for this row, but an unassigned flare whose
        # OWN class also reaches this threshold peaks in the same window
        # -- we cannot rule out that unassigned flare being this HARP's.
        masked = (positive == 0.0) & (unassigned_max_flux >= thresholds[cls])
        col = positive.copy()
        col[masked] = np.nan
        out[f"{LABEL_PREFIX}flare_{cls.lower()}_or_above"] = col

    primary = config["primary_target_class"]
    out[f"{LABEL_PREFIX}primary"] = out[f"{LABEL_PREFIX}flare_{primary.lower()}_or_above"]
    out[f"{LABEL_PREFIX}horizon_hours"] = float(horizon / pd.Timedelta(hours=1))
    return out
