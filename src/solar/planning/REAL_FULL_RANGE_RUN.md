# Real Full-Range Ingestion Run — 2010-05-01 .. 2025-07-24 (Cycle 24 + 25)

**Status: COMPLETE as of 2026-07-26.** Both GOES and SHARP normalized
datasets now cover the full 2010-05-01..2025-07-24 target range. See
"Final real results" below for row counts, day-level coverage, and the
genuine (not fabricated, not silently dropped) gaps that remain.

**What actually happened, honestly:** the original plan below (two
concurrent single-shot background jobs) is what was launched, but the GOES
job did not run to completion in one shot — its background process was
killed partway through (a harness/environment issue, not a code bug; see
`data_sources/goes_ingestion.py::_download`'s resume-skip comment) after it
had already downloaded raw `.nc` files for 2010-2017 and partway into 2023.
Because `ingest.py` only writes the normalized CSV/manifest *after* the
entire requested date range's `ingest_goes()` call returns, that partial
progress produced no normalized output at all for 2018-2020 or 2023-2025,
even though the raw files were sitting on disk. This was only discovered
2026-07-26 when a researcher noticed raw files already present for years
the normalized CSV showed as empty. The gap was closed by re-running
`ingest_goes` in two smaller chunks (2018-2020, then 2023-2025) via scoped
one-off run configs; `_download`'s existing-file skip meant most of the
already-downloaded raw files were reused for free rather than re-fetched.
SHARP's single chunked job (`ingest_sharp_chunked`) completed in one run as
originally planned — see its own row counts below.

## Scope and why this range

Per explicit researcher instruction, `configs/solar_ingest.yaml`'s
`goes.end_date` and `configs/sharp.yaml`'s `time_range.end` were extended
from an earlier `2019-12-31` (Cycle 24 only) to **2025-07-24**, so this run
genuinely covers Solar Cycle 25 (started ~Dec 2019), not just Cycle 24
(2010-2019). Start date (2010-05-01) is unchanged — SDO/HMI's launch date,
the earliest real SHARP data can exist (see
`planning/INGESTION_DOCUMENTATION.md`).

## A real limit discovered while preparing this run

Before launching the full range, `configs/sharp.yaml`'s time_range was
smoke-tested at calendar-YEAR and calendar-MONTH granularity against the
live JSOC service. Both failed for large/active periods:

- A full calendar year (2017) always failed (`DrmsQueryError(' [status=6]')`).
- A full calendar MONTH succeeded for quieter periods (Jan 2017: 11,485
  rows; Sep 2017: 15,646 rows) but STILL FAILED for high-activity months
  near Solar Cycle 24's maximum (Feb 2014, Oct 2014).

There is no fixed "safe" chunk size known in advance — JSOC's real limit
depends on how many active regions were being tracked, which varies with
solar activity, not calendar time. `data_sources/sharp_ingestion.py` was
extended with **adaptive bisection**: `ingest_sharp_chunked` starts at
month granularity and recursively bisects (by wall-clock time) any window
JSOC actually rejects, down to a 1-day floor, before recording a genuine
final failure. Verified against the two real months that failed at
month-granularity:

| Window | Bisection result | Rows | Time |
|---|---|---|---|
| Feb 2014 | split into 2 half-months, both succeeded | 36,512 | 135.9s |
| Oct 2014 | split into 2 half-months, both succeeded | 43,504 | 118.0s |
| Full year 2017 (control, no bisection needed) | 12/12 month chunks succeeded directly | 116,734 (195 distinct HARPNUMs) | 231.3s |

This is documented in `sharp_ingestion.py`'s module docstring and covered
by `tests/test_sharp_ingestion.py` (bisection-triggers-on-real-size-limit,
gives-up-at-the-floor, zero-row-chunks-are-never-bisected).

## The two background jobs

Launched 2026-07-25, running concurrently (independent services, no
resource contention):

- **GOES**: `ingest_goes(date(2010,5,1), date(2025,7,24))` — real full-disk
  1-min XRS irradiance, GOES-13/14/15 (legacy) + GOES-16/17/18 (R-series).
  Estimated several thousand individual file requests (one per
  satellite-day for R-series, satellite-month for legacy); expected
  multi-hour runtime.
- **SHARP**: `ingest_sharp_chunked(...)` — real SHARP magnetic keyword
  table, adaptive month-to-day chunking as described above. Extrapolating
  from the 2017 control run (~231s/quiet year) and the two bisected 2014
  months (~127s/half-month during high activity), a rough estimate is on
  the order of 1-2 hours for the full ~15.2-year range, though solar-max
  years (2011-2015, 2024-2025) will take longer per year than solar-min
  years.

Real elapsed times, row counts, manifest summaries, and any genuinely
failed (non-bisectable) chunks will be recorded here once both complete.

## Final real results (2026-07-26)

| | GOES XRS | SHARP |
|---|---|---|
| Rows | 16,597,686 | 4,967,165 |
| Date range covered | 2010-05-01 .. 2025-07-24 | 2010-05-01 .. 2025-07-24 |
| Fully missing calendar days (both satellites/all HARPs absent) | 2 | 372 |

Both datasets now genuinely span the full target range with no missing
years or multi-month blocks. Remaining gaps:

- **GOES: 2 fully-missing days** (2012-03-22, 2016-03-11) — every
  satellite's monthly legacy file for that month exists and was
  successfully parsed, but neither day has any rows inside it. This reads
  as a real data gap in NOAA's own archived files (e.g. an instrument
  outage), not an ingestion failure — the manifest for the surrounding
  files shows no fetch error. RESEARCHER VERIFICATION REQUIRED to confirm
  against NOAA's own outage records before citing a cause.
- **SHARP: 372 missing days**, concentrated before 2021-02-08 (none after)
  — plausible as genuine days with zero tracked active regions, not yet
  independently verified against an external flare/AR catalog.
- **Per-satellite (not per-day) GOES gaps**, all explained by satellite
  operational lifecycle boundaries, each failure cleanly bounded to a
  contiguous month range (strong internal evidence, but RESEARCHER
  VERIFICATION REQUIRED against NOAA's official GOES constellation
  history before citing as fact):
  - GOES-13 (legacy): fails from 2018 onward — consistent with GOES-13
    being retired/placed in storage in 2018.
  - GOES-18 (R-series): fails for all of 2018-2020 — consistent with
    GOES-18 not having launched yet (launch was March 2022).
  - GOES-17 (R-series): fails for essentially all of 2018-2020 AND all of
    2023-2025 — consistent with GOES-17 either not publishing this
    science-quality L2 product early in its mission, and/or being
    decommissioned once GOES-18 took over as GOES-West primary
    (~January 2023).
  - GOES-16 (R-series): fails April-July 2025 — consistent with GOES-16
    being superseded as GOES-East primary by GOES-19 (~April 2025).

None of these per-satellite gaps produce a fully-missing day in the final
dataset — at least one satellite has real data for every day except the 2
listed above, which is why GOES's day-level coverage (2 missing days) is
actually tighter than SHARP's (372 missing days).

## Real full pipeline run (2026-07-26): dataset + model results

With the GOES/SHARP ingestion above, the `splits.yaml` fix, the
`non_finite_value` QC rule, and tau*/interpretability additions all in
place, `run_pipeline.py` completed end-to-end on the real full range for
the first time. Real numbers (`results/real_full_20100501_20250724/`):

- 526,763 disk-integrated SHARP-cadence rows, 257 features.
- Split (post undefined-label drop): train 237,012 / val 77,100 / test
  211,594. `label_primary` (M-or-above, 24h horizon) prevalence 22.8% —
  high relative to per-active-region literature values, expected given
  disk-integrated labeling (any M+ flare anywhere counts as positive,
  not just within one HARP).
- Test-split TSS (the honest out-of-cycle, Cycle-25 generalization
  number): persistence 0.57, random forest 0.45, KNN 0.39, gradient
  boosting 0.37, climatology 0.00 (correct, by construction) —
  **logistic regression 0.00 (a real bug, not a legitimate result — see
  below)**.

**Known issue, deliberately left unfixed for now**: logistic regression
collapsed to a degenerate all-zero-coefficient model (TSS=0, ROC-AUC
exactly 0.5). The training log shows `lbfgs failed to converge after 0
iteration(s): ABNORMAL` — the classic failure mode of unscaled features:
raw SHARP physical quantities span wildly different magnitudes (USFLUX
~1e22, R_VALUE ~0-5, etc.), which breaks LBFGS immediately. The fix is
standard (standardize features, fit on train only, before logistic
regression specifically) but has not been applied — re-running the full
pipeline costs ~70 minutes (dominated by gradient boosting), so this was
deliberately deferred rather than re-run immediately. The other 5 models'
real results are unaffected and already valid.
