"""Real ingestion of SHARP (Space-weather HMI Active Region Patch) magnetic
field indices from Stanford's JSOC/DRMS interface, via the `drms` Python
package.

Scope: this module ingests the 18 physical keywords + 6 bookkeeping
keywords listed in configs/sharp.yaml (Table 7.1 of the thesis) as a
KEYWORD TABLE, not the underlying FITS images/segments. JSOC allows
keyword-only queries to be pulled directly with `drms.Client.query()`,
without going through the export+poll workflow that FITS/segment
downloads require -- so the `request.export_method` /
`request.poll_*` settings in configs/sharp.yaml are reserved for a later
stage (if this project ever needs raw magnetogram cutouts, not just the
scalar indices) and are intentionally unused here.

KNOWN LIMITATION, now addressed by chunking: JSOC applies undocumented-to-us
row/size limits on very large queries. `ingest_sharp` alone does not attempt
chunking -- for a full-cycle, all-HARPNUM query, use `ingest_sharp_chunked`
instead, which starts from calendar-month windows and ADAPTIVELY BISECTS
any window JSOC actually rejects, concatenating every chunk's rows into one
combined DataFrame/manifest, exactly as if a single call had succeeded.

The limit is genuinely undocumented and NOT a fixed calendar unit --
confirmed empirically against the live service on 2026-07-25: a full
CALENDAR YEAR always fails; a full MONTH succeeds during quieter periods
(e.g. Jan/Sep 2017: ~11.5k/~15.6k rows) but still FAILS during solar-maximum
months with many concurrently-tracked active regions (e.g. Feb/Oct 2014,
near Solar Cycle 24's peak) with `DrmsQueryError(' [status=6]')` -- a fixed
"safe" chunk size cannot be chosen in advance without knowing solar
activity ahead of time. `ingest_sharp_chunked` therefore starts at
month-granularity (cheap: most months succeed on the first try) and
recursively bisects by time ONLY the specific windows JSOC actually
rejects, down to a `_MIN_CHUNK` floor (1 day) before giving up and
recording a genuine, final failure. A query that succeeds but returns
ZERO rows (a legitimately quiet stretch with no tracked active region) is
never bisected further -- retrying a smaller window cannot produce
different real data, so `ingest_sharp`'s own "zero records" failure is
treated as final immediately, not retried.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .provenance import IngestedFileRecord, IngestionManifest, compute_sha256
from .schema import SHARP_NORMALIZED_COLUMNS, load_sharp_config

_TAI_FORMAT = "%Y.%m.%d_%H:%M:%S_TAI"
_MIN_CHUNK = timedelta(days=1)  # bisection floor -- below this, a failure is final, never retried again


def build_query_string(config: dict) -> str:
    """Builds a DRMS record-set query string, e.g.
    `hmi.sharp_cea_720s[][2010.05.01_00:00:00_TAI-2019.12.31_23:59:59_TAI]`.

    The first bracket is the HARPNUM filter (empty = all active regions);
    the second is the T_REC time-range filter. RESEARCHER VERIFICATION
    REQUIRED: confirm this record-set syntax still resolves against the
    live JSOC service before a full-scale run (test with a short time
    window and `harpnum_filter` set to 1-2 known HARPNUMs first)."""
    time_spec = f"{config['time_range']['start']}-{config['time_range']['end']}"
    harp_filter = config.get("harpnum_filter")
    harp_spec = ",".join(str(h) for h in harp_filter) if harp_filter else ""
    return f"{config['series']}[{harp_spec}][{time_spec}]"


def ingest_sharp(config: dict | None = None, client=None) -> tuple[pd.DataFrame, IngestionManifest]:
    """Queries JSOC for the configured SHARP keyword table over the
    configured time range, persists the raw response to disk (for
    checksum/provenance), normalizes it into `SHARP_NORMALIZED_COLUMNS`,
    and records the query in an `IngestionManifest`."""
    config = config or load_sharp_config()
    manifest = IngestionManifest("sharp")

    if client is None:
        import drms

        client = drms.Client(email=config["jsoc_email"])

    query = build_query_string(config)
    all_keywords = config["bookkeeping_keywords"] + config["keywords"]
    key_string = ", ".join(all_keywords)

    try:
        raw_df = client.query(query, key=key_string)
    except Exception as exc:  # noqa: BLE001 -- a failed JSOC query is recorded, not raised, so the caller can inspect the manifest
        manifest.add(IngestedFileRecord(
            source="REAL (JSOC SHARP keyword query)", request_identifier=query,
            local_path=None, status="failed", error=str(exc),
        ))
        return pd.DataFrame(columns=SHARP_NORMALIZED_COLUMNS), manifest

    if raw_df is None or len(raw_df) == 0:
        manifest.add(IngestedFileRecord(
            source="REAL (JSOC SHARP keyword query)", request_identifier=query,
            local_path=None, status="failed", error="query returned zero records",
        ))
        return pd.DataFrame(columns=SHARP_NORMALIZED_COLUMNS), manifest

    raw_dir = Path(config["output"]["raw_cache_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_path = raw_dir / f"sharp_keywords_{timestamp}.csv"
    raw_df.to_csv(raw_path, index=False)

    manifest.add(IngestedFileRecord(
        source="REAL (JSOC SHARP keyword query)", request_identifier=query,
        local_path=str(raw_path), status="success",
        byte_size=raw_path.stat().st_size, sha256=compute_sha256(raw_path),
    ))

    normalized = raw_df.rename(columns=str.upper)
    for col in SHARP_NORMALIZED_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = pd.NA
    normalized = normalized[SHARP_NORMALIZED_COLUMNS]

    return normalized, manifest


def month_chunks(start_tai: str, end_tai: str) -> list[tuple[str, str]]:
    """Splits a [start_tai, end_tai] TAI range (JSOC's
    "YYYY.MM.DD_HH:MM:SS_TAI" format) into calendar-month windows, e.g.
    2010.05.01_00:00:00_TAI .. 2010.07.15_23:59:59_TAI ->
    [(2010.05.01_00:00:00_TAI, 2010.05.31_23:59:59_TAI),
     (2010.06.01_00:00:00_TAI, 2010.06.30_23:59:59_TAI),
     (2010.07.01_00:00:00_TAI, 2010.07.15_23:59:59_TAI)].
    This is the STARTING granularity `ingest_sharp_chunked` queries at --
    most months succeed on the first try; only genuinely dense months get
    bisected further (see that function and the module docstring for why
    month, not year, and why a fixed chunk size can't be chosen a priori).
    A range entirely within one calendar month returns exactly one chunk
    (a no-op for a short pilot window)."""
    start = datetime.strptime(start_tai, _TAI_FORMAT)
    end = datetime.strptime(end_tai, _TAI_FORMAT)
    if start > end:
        raise ValueError(f"time_range start {start_tai!r} is after end {end_tai!r}")

    chunks = []
    chunk_start = start
    while chunk_start <= end:
        if chunk_start.month == 12:
            next_month_start = datetime(chunk_start.year + 1, 1, 1)
        else:
            next_month_start = datetime(chunk_start.year, chunk_start.month + 1, 1)
        month_end = next_month_start - timedelta(seconds=1)
        chunk_end = min(month_end, end)
        chunks.append((chunk_start.strftime(_TAI_FORMAT), chunk_end.strftime(_TAI_FORMAT)))
        chunk_start = next_month_start
    return chunks


def _bisect_tai_range(start_tai: str, end_tai: str) -> tuple[tuple[str, str], tuple[str, str]] | None:
    """Splits [start_tai, end_tai] in half by wall-clock time. Returns None
    if the range is already at or below `_MIN_CHUNK` -- the caller's signal
    that this range cannot be split further."""
    start = datetime.strptime(start_tai, _TAI_FORMAT)
    end = datetime.strptime(end_tai, _TAI_FORMAT)
    if (end - start) <= _MIN_CHUNK:
        return None
    mid = start + (end - start) / 2
    mid_str = mid.strftime(_TAI_FORMAT)
    return (start_tai, mid_str), (mid_str, end_tai)


def _ingest_sharp_adaptive(
    config: dict, client, start_tai: str, end_tai: str, manifest: IngestionManifest, frames: list[pd.DataFrame]
) -> None:
    """Queries [start_tai, end_tai]; on success (including a legitimate
    zero-row result) records it and stops. On any OTHER failure (a real
    JSOC error, e.g. the undocumented size limit), bisects the range and
    recurses on each half -- down to `_MIN_CHUNK`, at which point a
    persistent failure is recorded as final, not retried again."""
    chunk_config = {**config, "time_range": {"start": start_tai, "end": end_tai}}
    chunk_df, chunk_manifest = ingest_sharp(config=chunk_config, client=client)
    record = chunk_manifest.records[0]  # ingest_sharp always adds exactly one record per call

    is_final_empty = record.status == "failed" and record.error == "query returned zero records"
    if record.status == "success" or is_final_empty:
        manifest.add(record)
        if len(chunk_df):
            frames.append(chunk_df)
        return

    split = _bisect_tai_range(start_tai, end_tai)
    if split is None:
        manifest.add(record)  # at the bisection floor -- genuine, final failure
        return

    (left_start, left_end), (right_start, right_end) = split
    _ingest_sharp_adaptive(config, client, left_start, left_end, manifest, frames)
    _ingest_sharp_adaptive(config, client, right_start, right_end, manifest, frames)


def ingest_sharp_chunked(config: dict | None = None, client=None) -> tuple[pd.DataFrame, IngestionManifest]:
    """Starts from configs/sharp.yaml's time_range split into calendar-month
    windows (`month_chunks`), adaptively bisecting (see
    `_ingest_sharp_adaptive`) any window JSOC actually rejects, all over the
    SAME `drms.Client` (one connection, many queries). Concatenates every
    chunk's rows into ONE combined DataFrame and ONE combined
    IngestionManifest -- chunking/bisection is an internal implementation
    detail, invisible to a caller expecting the same shape `ingest_sharp`
    alone would return. A chunk that ultimately fails (even at the
    bisection floor) is recorded in the combined manifest as a failure for
    THAT specific window -- it does not abort the rest of the range, and it
    is never silently absent from the manifest."""
    config = config or load_sharp_config()
    if client is None:
        import drms

        client = drms.Client(email=config["jsoc_email"])

    manifest = IngestionManifest("sharp")
    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in month_chunks(config["time_range"]["start"], config["time_range"]["end"]):
        _ingest_sharp_adaptive(config, client, chunk_start, chunk_end, manifest, frames)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SHARP_NORMALIZED_COLUMNS)
    return combined, manifest
