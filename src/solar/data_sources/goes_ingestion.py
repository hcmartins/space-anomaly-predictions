"""Real ingestion of GOES X-Ray Sensor (XRS) soft X-ray irradiance data from
NOAA's public archives.

Two structurally different NOAA archives are handled, selected by date
(see configs/goes.yaml for the exact boundary and both URL templates):

  - GOES-R series (GOES-16/17/18, 2017-present): L2 science-quality,
    1-minute-average netCDF files, ONE FILE PER SATELLITE PER DAY.
  - Legacy GOES (GOES-13/14/15, up to ~2020): 1-minute-average netCDF
    files, ONE FILE PER SATELLITE PER MONTH, different variable names.

Both are normalized into the same `GOES_NORMALIZED_COLUMNS` schema
(`schema.py`) so downstream code never needs to know which archive a row
came from. Every file fetch (success or failure) is recorded in an
`IngestionManifest` -- a 404 or malformed file is never silently skipped.
"""
from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

from .provenance import IngestedFileRecord, IngestionManifest, compute_sha256
from .schema import GOES_NORMALIZED_COLUMNS, load_goes_config


def _daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _month_starts(start: date, end: date):
    current = date(start.year, start.month, 1)
    while current <= end:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def resolve_directory_listing(session, index_url: str, filename_regex: str) -> list[str]:
    """Fetches an NCEI directory-index page and returns filenames matching
    `filename_regex`. NCEI publishes plain HTML directory listings (Apache
    autoindex-style); this avoids hardcoding an exact version suffix
    (e.g. `v2-2-0`) that NOAA may bump without notice."""
    response = session.get(index_url, timeout=60)
    response.raise_for_status()
    pattern = re.compile(filename_regex)
    matches = sorted(set(pattern.findall(response.text)))
    return matches


def _download(session, url: str, dest_path: Path, config: dict) -> IngestedFileRecord:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # RESUME: dest_path is only ever left on disk by THIS function after a
    # full, size-verified successful download (a truncated/corrupt download
    # is deleted below, never left in place) -- so a file already present
    # here is safe to treat as already fetched, skipping the network call
    # entirely. Added after a real ~15-year full-range run's background
    # process was killed partway through (a harness/environment issue, not
    # a bug in this function) after ~2,464 real files had already been
    # downloaded; without this, a restart would have silently re-fetched
    # all of them from NOAA instead of resuming from where it stopped.
    if dest_path.exists():
        return IngestedFileRecord(
            source="REAL (NOAA GOES archive)", request_identifier=url,
            local_path=str(dest_path), status="success", http_status=200,
            byte_size=dest_path.stat().st_size, sha256=compute_sha256(dest_path),
        )

    req_cfg = config["request"]
    last_error = None
    last_status = None
    for attempt in range(1, req_cfg["max_retries"] + 1):
        try:
            response = session.get(
                url, timeout=req_cfg["timeout_seconds"],
                headers={"User-Agent": req_cfg["user_agent"]}, stream=True,
            )
            last_status = response.status_code
            if response.status_code == 200:
                expected_size = response.headers.get("Content-Length")
                with open(dest_path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
                actual_size = dest_path.stat().st_size
                # A network stall/truncation can end a stream early without
                # ever raising -- caught here explicitly rather than letting
                # a corrupted file reach the netCDF parser, where it surfaces
                # as an opaque low-level HDF5/netCDF error instead of a clear
                # "incomplete download" one (found via a real flaky-network run).
                if expected_size is not None and actual_size != int(expected_size):
                    dest_path.unlink(missing_ok=True)
                    last_error = f"incomplete download: got {actual_size} bytes, expected {expected_size}"
                else:
                    return IngestedFileRecord(
                        source="REAL (NOAA GOES archive)", request_identifier=url,
                        local_path=str(dest_path), status="success", http_status=200,
                        byte_size=actual_size, sha256=compute_sha256(dest_path),
                    )
            else:
                last_error = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001 -- network errors are recorded, not raised, so one bad file doesn't abort the run
            last_error = str(exc)
        if attempt < req_cfg["max_retries"]:
            import time

            time.sleep(req_cfg["retry_backoff_seconds"])
    return IngestedFileRecord(
        source="REAL (NOAA GOES archive)", request_identifier=url,
        local_path=None, status="failed", http_status=last_status, error=last_error,
    )


def _parse_goes_netcdf(path: Path, variables: list[dict], satellite: int, series: str, time_variable: str) -> pd.DataFrame:
    """Parses a single GOES XRS netCDF file into the normalized schema.
    Requires xarray + netCDF4 (declared as real-data extras in
    requirements.txt, not installed by default for the simulated-only
    Martian pipeline). `time_variable` differs by series -- confirmed
    against real downloaded files: GOES-R uses "time", legacy GOES uses
    "time_tag" (see configs/goes.yaml)."""
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
        raise ImportError(
            "Parsing real GOES netCDF files requires 'xarray' and 'netCDF4'. "
            "Install them (see src/solar/requirements.txt) before running ingestion."
        ) from exc

    var_map = {v["name"]: v["netcdf_variable"] for v in variables}
    with xr.open_dataset(path) as ds:
        time_index = pd.to_datetime(ds[time_variable].values, utc=True)
        data = {"time_utc": time_index}
        for norm_name, nc_name in var_map.items():
            if nc_name in ds.variables:
                data[norm_name.replace("_flux", "_flux_w_m2") if "flux" in norm_name else norm_name] = (
                    ds[nc_name].values
                )
            else:
                data[norm_name.replace("_flux", "_flux_w_m2") if "flux" in norm_name else norm_name] = float("nan")

    df = pd.DataFrame(data)
    df["satellite"] = satellite
    df["satellite_series"] = series
    df["source_file"] = path.name
    for col in GOES_NORMALIZED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[GOES_NORMALIZED_COLUMNS]


def ingest_goes_r(session, config: dict, start: date, end: date) -> tuple[pd.DataFrame, IngestionManifest]:
    """Ingests GOES-R series (16/17/18) daily L2 science-quality XRS files
    for every configured satellite over [start, end]."""
    goes_r_cfg = config["goes_r"]
    manifest = IngestionManifest("goes_r")
    frames: list[pd.DataFrame] = []
    raw_dir = Path(config["output"]["raw_cache_dir"])

    for day in _daterange(start, end):
        for sat in goes_r_cfg["satellites"]:
            sat_base_url = goes_r_cfg["base_url"].format(sat=sat)
            index_url = f"{sat_base_url}/{day.year}/{day.month:02d}/"
            listing_regex = rf'href="({re.escape(day.strftime("sci_xrsf-l2-avg1m_g" + str(sat) + "_d%Y%m%d"))}[\w\.\-]*\.nc)"'
            try:
                matches = resolve_directory_listing(session, index_url, listing_regex)
            except Exception as exc:  # noqa: BLE001
                manifest.add(IngestedFileRecord(
                    source="REAL (NOAA GOES-R L2)", request_identifier=index_url,
                    local_path=None, status="failed", error=f"directory listing failed: {exc}",
                ))
                continue
            if not matches:
                manifest.add(IngestedFileRecord(
                    source="REAL (NOAA GOES-R L2)", request_identifier=index_url,
                    local_path=None, status="failed", error="no matching file in directory listing",
                ))
                continue
            filename = matches[-1]  # latest version if multiple are listed
            file_url = urljoin(index_url, filename)
            dest_path = raw_dir / str(day.year) / f"{day.month:02d}" / filename
            record = _download(session, file_url, dest_path, config)
            manifest.add(record)
            if record.status == "success":
                try:
                    frames.append(_parse_goes_netcdf(
                        dest_path, goes_r_cfg["variables"], sat, "goes_r", goes_r_cfg["time_variable"]
                    ))
                except Exception as exc:  # noqa: BLE001
                    manifest.add(IngestedFileRecord(
                        source="REAL (NOAA GOES-R L2)", request_identifier=file_url,
                        local_path=str(dest_path), status="failed", error=f"parse failed: {exc}",
                    ))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GOES_NORMALIZED_COLUMNS)
    return combined, manifest


def ingest_goes_legacy(session, config: dict, start: date, end: date) -> tuple[pd.DataFrame, IngestionManifest]:
    """Ingests legacy GOES series (13/14/15) monthly XRS files for every
    configured satellite over [start, end]."""
    legacy_cfg = config["goes_legacy"]
    manifest = IngestionManifest("goes_legacy")
    frames: list[pd.DataFrame] = []
    raw_dir = Path(config["output"]["raw_cache_dir"])

    for month_start in _month_starts(start, end):
        month_end = date(month_start.year, month_start.month, monthrange(month_start.year, month_start.month)[1])
        for sat in legacy_cfg["satellites"]:
            filename = legacy_cfg["filename_pattern"].format(sat=sat, month_start=month_start, month_end=month_end)
            file_url = f"{legacy_cfg['base_url']}/{month_start.year}/{month_start.month:02d}/goes{sat}/netcdf/{filename}"
            dest_path = raw_dir / str(month_start.year) / f"{month_start.month:02d}" / filename
            record = _download(session, file_url, dest_path, config)
            manifest.add(record)
            if record.status == "success":
                try:
                    frames.append(_parse_goes_netcdf(
                        dest_path, legacy_cfg["variables"], sat, "goes_legacy", legacy_cfg["time_variable"]
                    ))
                except Exception as exc:  # noqa: BLE001
                    manifest.add(IngestedFileRecord(
                        source="REAL (NOAA GOES legacy)", request_identifier=file_url,
                        local_path=str(dest_path), status="failed", error=f"parse failed: {exc}",
                    ))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GOES_NORMALIZED_COLUMNS)
    return combined, manifest


def ingest_goes(start: date, end: date, config: dict | None = None, session=None) -> tuple[pd.DataFrame, list[IngestionManifest]]:
    """Top-level GOES ingestion: routes each requested date to the correct
    archive (GOES-R vs legacy) based on configs/goes.yaml's `valid_from`/
    `valid_to` boundaries, and concatenates results. Dates outside BOTH
    archives' coverage are reported as failures, never silently dropped."""
    config = config or load_goes_config()
    if session is None:
        import requests

        session = requests.Session()

    goes_r_start = date.fromisoformat(config["goes_r"]["valid_from"])
    legacy_start = date.fromisoformat(config["goes_legacy"]["valid_from"])
    legacy_end = date.fromisoformat(config["goes_legacy"]["valid_to"])

    frames = []
    manifests = []

    legacy_overlap_start = max(start, legacy_start)
    legacy_overlap_end = min(end, legacy_end)
    if legacy_overlap_start <= legacy_overlap_end:
        df, manifest = ingest_goes_legacy(session, config, legacy_overlap_start, legacy_overlap_end)
        frames.append(df)
        manifests.append(manifest)

    r_overlap_start = max(start, goes_r_start)
    r_overlap_end = end
    if r_overlap_start <= r_overlap_end:
        df, manifest = ingest_goes_r(session, config, r_overlap_start, r_overlap_end)
        frames.append(df)
        manifests.append(manifest)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GOES_NORMALIZED_COLUMNS)
    return combined, manifests
