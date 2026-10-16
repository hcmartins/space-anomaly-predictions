"""Real ingestion of NOAA's GOES Flare Report (NCEI) -- a consolidated,
science-quality, per-flare-event catalogue used to attribute cataloged
flares to NOAA active regions for Chapter 3.2.1's per-active-region
forecasting task (see preprocessing/harp_attribution.py for the next
step: NOAA AR -> HARPNUM). This is a DIFFERENT, additive data source from
data_sources/goes_ingestion.py's raw 1-min XRS irradiance -- that module
still feeds preprocessing/labeling's existing disk-integrated,
GOES-threshold-only labels unchanged; this one is Phase 1 of scoping out
a real catalogue-attributed alternative, not a replacement (yet).

Real archive structure confirmed live 2026-07-25/26 -- see
configs/flare_catalogue.yaml's header for the verification log. Reuses
data_sources.goes_ingestion's `_download` (byte-verified,
retry/truncation-safe, resume-safe -- skips the network call entirely if
the file is already cached) and `resolve_directory_listing` (regex-based
version-suffix resolution, since NCEI republishes this product with
bumped versions -- already observed 1-0-0 -> 1-0-1 during verification)
rather than duplicating that logic.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .goes_ingestion import _download, resolve_directory_listing
from .provenance import IngestedFileRecord, IngestionManifest
from .schema import FLARE_CATALOGUE_NORMALIZED_COLUMNS, load_flare_catalogue_config


def _normalize_active_region(series: pd.Series) -> pd.Series:
    """Real active_region values are read as float64 (the column has NaN
    mixed with numeric-looking values) -- must become "2673", never
    "2673.0" or "nan". Missing -> "" (Algorithm 3.2's unassigned-flare
    case), never fabricated."""
    return series.apply(lambda v: "" if pd.isna(v) else str(int(v)))


def ingest_flare_catalogue(session=None, config: dict | None = None) -> tuple[pd.DataFrame, IngestionManifest]:
    """Downloads the real mission-length GOES Flare Report CSV, normalizes
    it to FLARE_CATALOGUE_NORMALIZED_COLUMNS, and filters to
    config['date_range']. Every fetch (success or failure) is recorded in
    the returned manifest -- a dataset with zero rows is never silently
    substituted for a failed fetch."""
    config = config or load_flare_catalogue_config()
    manifest = IngestionManifest("flare_catalogue")

    if session is None:
        import requests

        session = requests.Session()

    index_url = config["index_url"]
    try:
        filenames = resolve_directory_listing(session, index_url, config["mission_file_regex"])
    except Exception as exc:  # noqa: BLE001 -- a bad index page is recorded, not raised, mirrors goes_ingestion.py
        manifest.add(IngestedFileRecord(
            source="REAL (NOAA GOES Flare Report, NCEI)", request_identifier=index_url,
            local_path=None, status="failed", error=f"directory listing failed: {exc}",
        ))
        return pd.DataFrame(columns=FLARE_CATALOGUE_NORMALIZED_COLUMNS), manifest

    if not filenames:
        manifest.add(IngestedFileRecord(
            source="REAL (NOAA GOES Flare Report, NCEI)", request_identifier=index_url,
            local_path=None, status="failed", error="no matching file in directory listing",
        ))
        return pd.DataFrame(columns=FLARE_CATALOGUE_NORMALIZED_COLUMNS), manifest

    # Lexicographically-last match = highest version (e.g. v1-0-1 after
    # v1-0-0) -- same convention goes_ingestion.py's GOES-R resolution uses.
    filename = sorted(filenames)[-1]
    url = index_url + filename
    dest_path = Path(config["output"]["raw_cache_dir"]) / filename

    record = _download(session, url, dest_path, config)
    manifest.add(record)
    if record.status != "success":
        return pd.DataFrame(columns=FLARE_CATALOGUE_NORMALIZED_COLUMNS), manifest

    raw_df = pd.read_csv(dest_path)
    mapping = config["column_mapping"]
    normalized = pd.DataFrame({norm_col: raw_df[raw_col] for norm_col, raw_col in mapping.items()})
    normalized["active_region"] = _normalize_active_region(normalized["active_region"])
    normalized["flare_id"] = normalized["flare_id"].astype(str)
    for time_col in ("time_utc_peak", "time_utc_start", "time_utc_end"):
        normalized[time_col] = pd.to_datetime(normalized[time_col], utc=True)

    start = pd.Timestamp(config["date_range"]["start"], tz="UTC")
    end = pd.Timestamp(config["date_range"]["end"], tz="UTC") + pd.Timedelta(days=1)
    normalized = normalized.loc[
        (normalized["time_utc_peak"] >= start) & (normalized["time_utc_peak"] < end)
    ].reset_index(drop=True)

    return normalized[FLARE_CATALOGUE_NORMALIZED_COLUMNS], manifest
