# Solar (GOES + SHARP) Real Data Ingestion — Documentation

Status: **both GOES and SHARP ingestion have now been run against the live
NOAA/JSOC archives and verified** (see the Verification log below, and
`planning/REAL_PILOT_RUN_2017-09.md` for the first full real end-to-end
pipeline run through modeling). As of 2026-07-26, **both datasets have also
completed a real full-range run (2010-05-01..2025-07-24)** — see
`planning/REAL_FULL_RANGE_RUN.md` for row counts and the genuine (mostly
satellite-lifecycle-explained) gaps that remain. This document records what
is genuinely known, what still needs a researcher to verify against the
live services, and what deliberately is NOT implemented yet.

## Scope

This covers `src/solar/data_sources/goes_ingestion.py` and
`sharp_ingestion.py`, orchestrated by `src/solar/ingest.py`
(`python -m ingest --config configs/solar_ingest.yaml`, run with cwd set
to `src/solar` -- verified; `python -m solar.ingest` from `src/` does NOT
work, see `ingest.py`'s module docstring). It implements Chapter 3 §3.3's
data-collection step only —
preprocessing, feature engineering, labeling, splitting, and modeling
(§3.4 onward) are not part of this delivery. See `src/solar/__init__.py`.

## GOES XRS (soft X-ray irradiance)

Two structurally different NOAA archives, on two different hosts,
selected by date:

| | GOES-R series | Legacy GOES |
|---|---|---|
| Satellites | 16, 17, 18 | 13, 14, 15 |
| Coverage | 2017-02-07 – present | 2008-01-01 – ~2020-03-04 |
| File granularity | one file per satellite per **day** | one file per satellite per **month** |
| Product | L2 science-quality, 1-min avg | 1-min avg |
| Host | `data.ngdc.noaa.gov` | `www.ncei.noaa.gov` |
| Key netCDF variables | `xrsa_flux`, `xrsb_flux` | `A_AVG`, `B_AVG` |
| Time variable | `time` | `time_tag` |

Flare classification uses peak `xrsb_flux` (long channel, 1–8 Å), per
Chapter 4.3's thresholds, encoded in `configs/goes.yaml`:
X ≥ 1.0×10⁻⁴, M ≥ 1.0×10⁻⁵, C ≥ 1.0×10⁻⁶, B ≥ 1.0×10⁻⁷ W/m².

All of the above (hosts, filename patterns, variable names) were
**confirmed against the live archive on 2026-07-25** — see the
Verification log below for what an earlier, unverified transcription got
wrong. NOAA can still reorganize archive paths again in the future: if a
run starts failing every file with HTTP 404 where it previously
succeeded, re-verify before assuming the underlying data is unavailable.

The GOES-R downloader resolves the exact filename (including the version
suffix, e.g. `v2-2-0`) by fetching NCEI's HTML directory listing and
regex-matching it, rather than hardcoding a version string that would
break silently when NOAA republishes a file with a bumped version.

## SHARP (magnetic field indices)

Source: Stanford's JSOC, via the `drms` Python package, series
`hmi.sharp_cea_720s` (Cylindrical Equal Area projection, 12-minute
cadence). Chapter 3.3's stated cadence and Table 7.1's 18 keywords match
this series — cross-checked against Bobra et al. 2014, the SHARP
definition paper already in the thesis bibliography.

**Access requirement — a real JSOC policy, not a code limitation:**
exporting/querying data from JSOC requires a registered email address
(free, ~5 minutes: http://jsoc.stanford.edu/ajax/register_email.html).
`configs/sharp.yaml`'s `jsoc_email` field has no default; `load_sharp_config()`
raises immediately if it is unset, before any network call is attempted.

Because this delivery only ingests **keyword tables** (the 18 physical
indices + 6 bookkeeping keywords), not FITS image segments, the ingestion
uses `drms.Client.query(record_set, key=...)` directly — this does not
require the export+poll workflow that segment/image downloads need. The
`request.export_method`/`request.poll_*` settings in `configs/sharp.yaml`
are reserved for a possible future stage that downloads raw magnetogram
cutouts; they are unused by the current `sharp_ingestion.py`.

**RESEARCHER VERIFICATION REQUIRED before the first real run:**
1. Confirm the record-set query syntax in `sharp_ingestion.build_query_string`
   (`series[harpnum][time_range]`) still resolves — test with a short
   time window and 1–2 known `harpnum_filter` values first, not the full
   2010–2019 range.
2. JSOC applies undocumented-to-us row/size limits on very large queries.
   No automatic chunking is implemented; if a full-range query fails or
   silently truncates, split `configs/sharp.yaml`'s `time_range` into
   smaller windows and call `ingest_sharp` once per window.

SDO/HMI launched February 2010, so no real SHARP data exists before then
— `configs/sharp.yaml`'s `time_range.start` (2010-05-01) and
`configs/solar_ingest.yaml`'s GOES `start_date` are deliberately aligned
to this, even though GOES XRS itself (legacy GOES-13) has data back to
2008: there is no SHARP data to pair with 2008–2010 GOES rows for this
project's purposes.

## Provenance and failure handling

Every file/query fetch — success or failure — is recorded as an
`IngestedFileRecord` (`data_sources/provenance.py`) with the exact URL or
JSOC query string, HTTP status, byte size, SHA-256 checksum (for
successes), and UTC access timestamp. `ingest.py` writes one JSON
manifest per run per dataset to `data_cache/{goes,sharp}/manifests/` and
never overwrites a previous run's manifest. A dataset with zero
successfully ingested rows is reported with `status: "FAILED..."` in the
CLI's printed summary and its normalized CSV is not written — there is no
silent fallback to placeholder or previously cached data.

## Verification log (GOES)

On 2026-07-25, `goes_ingestion.py` was actually run against the live NOAA
archives (a 1-day, all-satellites smoke test, `2018-01-01`), not just
written and assumed correct. This caught and fixed three real defects that
a purely-transcribed config would have shipped silently:

1. `goes_r.base_url` pointed at the wrong host/path entirely
   (`www.ncei.noaa.gov/.../access/l2/data/...` 404s; the real host is
   `data.ngdc.noaa.gov`, and the satellite number is part of the URL path,
   not just the filename).
2. Legacy GOES flux variables are named `A_AVG`/`B_AVG` in the real files,
   not `A_FLUX`/`B_FLUX` as an earlier draft assumed.
3. The time coordinate is named differently per series (`time` for GOES-R,
   `time_tag` for legacy) — hardcoding `"time"` silently produced wrong
   results for legacy files before this was caught.

A fourth issue surfaced only intermittently across repeated runs: a
network stall could truncate a download mid-stream without `requests`
raising an error, leaving a corrupted file that `_download` recorded as
`"success"` and that later crashed the netCDF parser with an opaque
low-level HDF5 error (`NCbytes failure`) instead of a clear per-file
failure. `_download` now compares the downloaded byte count against the
response's `Content-Length` header and rejects (deletes, marks failed)
any mismatch before the file is ever handed to the parser. After this
fix, 3 consecutive full smoke-test runs completed without a crash;
individual file failures (e.g. GOES-13 having no XRS data for a date
after it stopped being actively used, GOES-17/18 having no data before
their respective launches) were correctly recorded as per-file manifest
failures rather than crashing the run.

`goes.yaml`'s quality-flag variable names (`xrsa_flag`/`xrsb_flag` for
GOES-R, `A_QUAL_FLAG`/`B_QUAL_FLAG` for legacy) were also confirmed
directly from real downloaded files' variable listings during this same
verification pass.

## Verification log (SHARP)

On 2026-07-25, `sharp_ingestion.py` was run against the live JSOC service
for the first time (`hmi.sharp_cea_720s`, 2017-09-04 .. 2017-09-10, all
HARPNUMs) — part of the first full real end-to-end pipeline run; see
`planning/REAL_PILOT_RUN_2017-09.md` for the complete picture. `build_query_string`'s
record-set syntax (`series[harp_spec][time_spec]`) resolved correctly on
the first attempt: 4,230 real keyword-table rows across 14 real HARPNUMs
were returned, including HARP 7115 (the real, independently-documented HARP
number for AR 12673, source of the 2017-09-06 X9.3 flare) — an
independent cross-check that the query is pulling genuinely correct,
correctly-attributed data, not just *some* data.

One thing this run did NOT confirm: whether `configs/sharp.yaml`'s
`jsoc_email` genuinely needs to be a JSOC-registered address. JSOC's
keyword-only `drms.Client.query()` (the only method this project's
`ingest_sharp` calls) appears not to authenticate against the supplied
email — the query succeeded regardless. The email requirement documented
above is a real JSOC policy for `.export()` (used for FITS/image segment
downloads, e.g. `request.export_method` in `configs/sharp.yaml`, which
this project's keyword-table-only ingestion never calls) — RESEARCHER
VERIFICATION REQUIRED if this project is ever extended to call `.export()`
directly.

`configs/sharp.yaml`'s RESEARCHER VERIFICATION REQUIRED note on
`invalid_quality_flag`'s `pass_value=0` assumption remains open: this run
rejected 190 of 4,230 real rows under that rule, which is at least
evidence the `QUALITY` column is present and populated with a plausible
bitmask in real data, but the VALUE semantics (which nonzero codes mean
what) still have not been independently checked against JSOC's own
documentation.

**Full-scale run status (updated 2026-07-26):** both datasets have now
been run across the entire configured 2010-05-01..2025-07-24 range — see
`planning/REAL_FULL_RANGE_RUN.md` for the real row counts, day-level
coverage, and the one real operational lesson learned from it: a
single-shot multi-year `ingest_goes()` call has no internal checkpointing
(the normalized CSV is only written after the *entire* call returns), so a
background process killed partway through loses all normalized output for
that run even though raw files were already fetched to disk. `_download`'s
existing-file skip made recovery cheap, but a future full-range GOES run
should still prefer chunking by year (as SHARP already does automatically
via `ingest_sharp_chunked`) so a partial failure doesn't require redoing
the whole multi-year call to get any output at all.

## What is NOT implemented in this delivery

- Preprocessing, feature engineering, causal filtering, labeling, chronological
  splitting, modeling, evaluation (Chapter 3 §3.4 onward / Chapter 4).
- Automatic chunking of oversized JSOC queries.
- Downloading raw SHARP FITS segments/images (keyword tables only).
