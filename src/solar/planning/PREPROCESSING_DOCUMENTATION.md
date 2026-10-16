# Solar Preprocessing — Documentation

Status: **implemented, unit-tested (105/105 tests passing across the whole
`src/solar` package), and now also run end-to-end against real ingested
data.** Every module below is exercised against small, hand-constructed
DataFrames covering its documented edge cases (causality, gap handling,
aggregation methods) in `tests/`; on top of that, `alignment.py`,
`missing_data.py`, and `features.py` were all run against a real 7-day
GOES+SHARP ingestion (2017-09-04..10) on 2026-07-25 — see
`planning/REAL_PILOT_RUN_2017-09.md` for the full numbers, including a real
performance bug in `features.py` that only surfaced at real scale (234
columns) and was fixed as a direct result. **Not yet run at FULL scale**
(2010-2019, `configs/solar_ingest.yaml`/`configs/sharp.yaml`'s real
intended range) — the pilot run was 7 days, not ~9.5 years; see
`planning/INGESTION_DOCUMENTATION.md`'s verification log for what is and
isn't confirmed at full scale.

## Scope

Covers `preprocessing/alignment.py`, `preprocessing/missing_data.py`, and
`preprocessing/features.py`. Consumes the QC-accepted output of
`preprocessing/quality_control.py`, which itself consumes the normalized
output of `data_sources/{goes,sharp}_ingestion.py`. Labeling
(`labeling/flare_labels.py`) is a separate, downstream stage — see
`planning/LABELING_DOCUMENTATION.md` — deliberately kept out of this
package because it is the one place forward-looking (non-causal) logic is
allowed; every module documented here is strictly causal.

Pipeline order: `quality_control.py` → `alignment.py` → `missing_data.py`
→ `features.py` → (`labeling/flare_labels.py`, separately, on the same
aligned timeline).

## Why disk-integrated, not per-active-region (`alignment.py`)

This project has not ingested a real flare-event-to-active-region
attribution catalogue (e.g. NOAA SWPC's flare event list with NOAA AR
numbers), and GOES XRS itself is a full-disk, not per-region, flux
measurement. Building a per-active-region forecasting dataset without a
real attribution catalogue would require either ingesting that catalogue as
a new real data source (not done in this delivery) or fabricating an
AR-to-flare mapping — refused per this project's no-fabrication policy.
`alignment.py` therefore builds a **disk-integrated** series: at each SHARP
cadence timestamp, aggregate ACROSS all currently-tracked active regions and
combine with the full-disk GOES flux state. This is a legitimate, published
approach to flare forecasting ("will a flare of class ≥ K occur anywhere on
the disk"), just different from per-region forecasting. See
`configs/alignment.yaml`'s header for the full statement, including why
SHARP keywords are split sum- vs. max-aggregated (extensive vs.
intensive/complexity quantities) and why GOES multi-satellite aggregation
uses `max` rather than a primary/secondary satellite lookup.

**Bug found and fixed during this pass:** `attach_goes_trailing_state`'s
lower window bound used `np.searchsorted(..., side="right")`, which
excludes a GOES observation landing exactly on `window_start` — an off-by-one
that silently dropped one valid observation per window whenever a GOES
timestamp aligned exactly with `sharp_time - window`. Caught by
`tests/test_alignment.py::test_attach_goes_trailing_state_is_causal_never_uses_future_goes`,
which was failing before this fix (`n_obs` was 1, expected 2). Fixed to
`side="left"` so the window is correctly inclusive of both its start and
end, matching the docstring's stated semantics.

## Causal gap-filling (`missing_data.py`)

A missing SHARP-cadence step in the disk-integrated series is genuinely
ambiguous: it could mean "zero active regions on the visible disk" (a real
physical state) or "an active region was present but every one of its
records was rejected by QC" (a data-quality gap). This project has not
ingested an independent active-region count that would disambiguate the
two — see `configs/missing_data.yaml`'s header. **Engineering decision:**
last-observation-carried-forward (LOCF), bounded by
`max_trailing_fill_gap_steps` (default 3 steps = 36 minutes), was chosen
over zero-filling the extensive SHARP keywords, because zero-filling would
silently assert "no flare risk" during what might actually be a QC dropout
of a genuinely active region — the more dangerous of the two possible
errors for a flare-forecasting feature. A gap wider than the bound is never
fabricated: those rows are simply absent from the output, mirroring
`martian/preprocessing/missing_data.py`'s "do not interpolate a row into
existence when no defensible observation is available" convention.

Unlike the Martian version (which fills independently within each
`(cell_id, mars_year)` spatial group), this operates on a single global
disk-wide time series — there is no spatial grid in the disk-integrated
design — so the whole series is reindexed onto one regular
`cadence_minutes` grid and filled in one vectorized pass, no per-group
looping needed.

## Causal features (`features.py`)

Lag / trailing-window / first-difference / slope features, plus a
"persistence of elevated flux" run-length counter and pass-through QC
indicator columns (`is_observed`, `interpolation_flag`, `gap_length_steps`
from `missing_data.py`). All feature columns are prefixed `feat_`.

**Important correctness point:** lag/trailing lookups are keyed by EXACT
timestamp (`t - lag * cadence_minutes`), never by row-adjacency. Once
`missing_data.py` has dropped rows for gaps too long to fill, consecutive
ROWS in the DataFrame are no longer guaranteed to be consecutive TIME
STEPS — naive row-shifting would silently treat row-adjacency as
time-adjacency across such a gap and fabricate a lag that spans a real,
unfilled data gap. Tested explicitly in
`tests/test_features.py::test_lag_across_dropped_gap_is_nan_not_fabricated`.

**Performance rewrite (2026-07-26):** the original implementation enforced
the above via a per-row Python dict lookup (`lookup.get(t - lag*cadence)`)
inside a `for idx, t in enumerate(...)` loop, plus an `np.polyfit` call per
row per variable for the slope feature. This was fine at pilot scale (745
rows) but measured at **76ms/row** when benchmarked against the real
full-range scale (~666k rows over 2010-2025) — a projected **~14 hours**
for feature engineering alone, and as a single non-checkpointed Python call
with no incremental output, a single point of failure of exactly the kind
`planning/REAL_FULL_RANGE_RUN.md` already documented biting the GOES
ingestion run. Rewritten to be vectorized: reindex `filled_df` onto its OWN
regular cadence grid first (making any dropped-gap step an explicit NaN
row, so `.shift()`/`.rolling()` become safe — this is a vectorized form of
the same exact-timestamp lookup, not a different, looser one), then compute
lags/trailing-stats/diff via `.shift()`/`.rolling()`, the slope via a
closed-form OLS formula over masked sums across the `window` shifted
columns (mathematically identical to `np.polyfit(x, y, 1)[0]`, not an
approximation), and the persistence run-counter via a vectorized
reset-on-change group/cumcount idiom. Verified equivalent to the original
row-loop implementation across 8 randomized trials (150-400 rows each,
multiple variables, ~10% randomly dropped rows to exercise real gaps, ~5%
NaN injection) with every output column matching within `rtol=1e-9`, in
addition to the existing 8 hand-computed unit tests all still passing
unchanged. Real-scale benchmark after the rewrite: **9.56s for 666,120
rows** (~5,300x speedup) — see `planning/REAL_FULL_RANGE_RUN.md`'s sibling
doc for the real full pipeline run this unblocked.

No feature column is ever named with a `label_` prefix, and this module has
no access to any forecast-horizon or flare-class-threshold construct — a
structural guarantee against feature/label leakage, mirroring
`martian/preprocessing/features.py`'s equivalent guarantee (there, the
`target_` prefix reservation).

The persistence-counter's "elevated flux" threshold
(`configs/features.yaml`'s `persistence_flux_threshold_w_m2`, default the
C-class threshold) is deliberately NOT cross-loaded from
`configs/goes.yaml` or `configs/labeling.yaml` — kept local so this module
never has an implicit dependency on the separately-configured label
definition.

## What is NOT implemented in this delivery

- Running any of these modules against a real, full-scale ingested dataset
  (only unit-tested against small synthetic DataFrames so far).
- Per-active-region preprocessing (would require ingesting a real
  AR-to-flare attribution catalogue — see "Why disk-integrated" above).
- Modeling and evaluation — still future work, not part of this delivery.
  Chronological train/val/test splitting IS now implemented — see
  `planning/SPLITTING_DOCUMENTATION.md`.
