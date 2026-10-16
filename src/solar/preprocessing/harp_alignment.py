"""Per-active-region (HARP-level) alignment -- Phase 2 of scoping out
Chapter 3.2.1's per-HARP forecasting task (Phase 1:
preprocessing/harp_attribution.py + data_sources/flare_catalogue_ingestion.py)..

Sits ALONGSIDE preprocessing/alignment.py's disk-integrated pipeline, not
in place of it -- the disk-integrated pipeline's real, already-reported
results must not change. This module reuses
alignment.py::attach_goes_trailing_state and aggregate_goes_disk
UNCHANGED: attach_goes_trailing_state computes its trailing-window stats
purely from each row's own time_utc, with no assumption that timestamps
are unique, so it works correctly on a per-HARP dataframe (many rows
share the same T_REC across different HARPNUMs) with zero modification.

The one genuinely new preprocessing step here is limb exclusion
(configs/alignment.yaml's per_harp section) -- not needed by the
disk-integrated path (which aggregates across the whole disk regardless
of position) but required by Chapter 3.4.2 for per-region forecasting,
where a near-limb HARP's own SHARP parameters are known to be unreliable.
"""
from __future__ import annotations

import pandas as pd

from .alignment import _parse_tai_string, aggregate_goes_disk, attach_goes_trailing_state, load_alignment_config


def prepare_sharp_per_harp(qc_accepted_sharp_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Per-HARP equivalent of alignment.py's aggregate_sharp_disk: parses
    T_REC into time_utc but does NOT aggregate across active regions --
    keeps one row per (HARPNUM, T_REC), the spatial unit Chapter 3.2.1
    requires. Applies limb exclusion (configs/alignment.yaml's
    per_harp.limb_longitude_threshold_deg) using LAT_FWT/LON_FWT, both
    already ingested and confirmed in Stonyhurst coordinates (Bobra et
    al. 2014) -- no conversion needed."""
    config = config or load_alignment_config()
    time_col = config["sharp_time_column"]
    threshold = config["per_harp"]["limb_longitude_threshold_deg"]

    df = qc_accepted_sharp_df.copy()
    df["time_utc"] = df[time_col].apply(_parse_tai_string)

    if "LON_FWT" in df.columns:
        within_limb = df["LON_FWT"].abs() <= threshold
        df = df.loc[within_limb.fillna(False)]

    return df.sort_values(["HARPNUM", "time_utc"]).reset_index(drop=True)


def build_per_harp_timeline(
    qc_accepted_goes_df: pd.DataFrame, qc_accepted_sharp_df: pd.DataFrame, config: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-HARP equivalent of alignment.py's build_aligned_timeline.
    Returns (aligned, goes_disk) in the SAME shape that function does --
    GOES itself stays disk-integrated (it is not per-region data), but
    `aligned` now carries one row per (HARPNUM, T_REC) with the same
    disk-wide GOES trailing-window state broadcast onto every HARP active
    at that timestamp, rather than one row per T_REC."""
    config = config or load_alignment_config()
    goes_disk = aggregate_goes_disk(qc_accepted_goes_df, config)
    sharp_per_harp = prepare_sharp_per_harp(qc_accepted_sharp_df, config)
    aligned = attach_goes_trailing_state(sharp_per_harp, goes_disk, config)
    return aligned, goes_disk
