# Real End-to-End Pilot Run — 2017-09-04 .. 2017-09-10

**This is the first time every stage of the solar pipeline (ingestion ->
QC -> alignment -> gap-filling -> features -> labeling -> dataset join ->
split+embargo -> baseline/classical-model training) has been run on REAL
data start to finish, in one continuous chain, on this machine.** Every
number below comes from an actual run against the live NOAA and JSOC
services on 2026-07-25; nothing here is synthetic, simulated, or
hand-constructed. It supersedes the "not yet verified" status the other
planning docs previously carried for SHARP and for everything downstream
of ingestion.

## Why this window

2017-09-04 .. 2017-09-10 covers AR 12673 (HARP 7115), which produced an
X2.2 and the X9.3 flare (the largest of Solar Cycle 24) on 2017-09-06, and
an X8.2 flare on 2017-09-10 — chosen deliberately so the pilot dataset
would contain genuine, well-documented positive flare events, not just a
quiet-sun stretch. It is a 7-day window, not the full 2010-2019 range
`configs/solar_ingest.yaml`/`configs/sharp.yaml` specify — a full-cycle run
was not attempted in this pass (see "What this run does NOT establish"
below).

## What actually ran, stage by stage

| Stage | Result |
|---|---|
| GOES ingestion | 96,480 raw rows. `goes_legacy`: 3 requested (GOES-13/14/15), 2 succeeded, 1 failed. `goes_r`: 21 requested (GOES-16/17/18 x 7 days), 7 succeeded (GOES-16 only — GOES-17/18 correctly failed, not yet launched in Sept 2017), 14 failed. All failures are genuine per-file manifest entries, not crashes — consistent with the GOES verification log in `planning/INGESTION_DOCUMENTATION.md`. |
| SHARP ingestion | **First live run ever** — 4,230 real keyword-table rows across 14 real HARPNUMs (including 7115, confirmed as the real HARP for AR 12673), via one `drms.Client.query()` call, zero retries needed. `jsoc_email` in `configs/sharp.yaml` was NOT actually required for this query — JSOC's keyword-only `.query()` does not appear to authenticate against it (only `.export()`, unused by this project, does per JSOC's docs); the field is still required by `load_sharp_config` as a guard, but this run did not exercise whether the address is genuinely registered. |
| QC (GOES) | 92,066 / 96,480 accepted (4,414 rejected — 4,404 by `invalid_quality_flag`, 10 by `missing_required_column`). |
| QC (SHARP) | 4,040 / 4,230 accepted (190 rejected by `invalid_quality_flag`). |
| Disk-level alignment | 745 SHARP-cadence rows, 2017-09-04T00:00Z .. 2017-09-11T00:00Z. 43,200 native 1-min GOES disk rows. |
| Causal gap-filling | 792 rows (745 observed + 47 LOCF-filled, default `max_trailing_fill_gap_steps=3`). |
| Feature engineering | 234 `feat_`-prefixed columns. |
| Labeling | Ran with `forecast_horizon_hours=6` (a PILOT override — see below), not the default 24h. `label_max_flare_class_in_horizon`: 485 rows M, 138 rows X, 122 rows C (all 745 rows had a DEFINED label — full coverage, no undefined rows, since the ingested window fully spans every row's 6h look-ahead). |
| Dataset join | 745 rows, 257-column feature manifest. `label_primary` (M-or-above) positive rate **0.836** — see caveat below. `label_flare_x_or_above` positive rate 0.185. |
| Split + embargo (PILOT config) | train 278, val 182, test 229, embargo 55, unassigned 1 (of 745). |
| Training | Baselines (climatology, persistence) and classical models (logistic regression, random forest) all fit and evaluated without error; checkpoints and a metrics JSON were written for real. See "Real, honest results" below for what the numbers actually show. |

## Two deliberate, explicitly-flagged deviations from the project's real configs

This run used ITS OWN in-memory config overrides, **never written back to
the tracked `configs/*.yaml` files**, which still represent the intended
full 2010-2019 run:

1. **`forecast_horizon_hours=6`**, not `configs/labeling.yaml`'s default
   24h — a 7-day window can't sensibly support a 24h forward-coverage
   requirement near its own edges without discarding most of it as
   UNDEFINED; 6h was chosen from `configs/labeling.yaml`'s own
   `sensitivity_horizons_hours` list, not invented ad hoc.
2. **A pilot `splits_config`** (train 09-04..06, val 09-07..08, test
   09-09..10, `embargo_hours=6`), not `configs/splits.yaml`'s real
   2010/2016/2018 boundaries — those boundaries assume years of data per
   split; applied to a single week they would put 100% of this pilot's
   rows in the "val" bucket and report `status: "insufficient_data"` (train
   empty). Confirmed this is exactly what happens if
   `configs/splits.yaml`'s real boundaries ARE used unmodified — a
   legitimate, correct behavior of the leak-audited splitter (it refuses to
   fabricate a train/test split where none genuinely exists), not a bug.

## Real, honest results — and why they should not be over-interpreted

- **`label_primary` positive rate is 0.836**, dramatically higher than any
  realistic multi-year base rate for M-or-above flares. This is a direct,
  expected consequence of choosing the single most flare-active week of
  Solar Cycle 24 as the pilot window — it validates that the pipeline
  correctly IDENTIFIES real positive events, but this pilot's class balance
  must never be read as an estimate of the true base rate.
- **RandomForestClassifier**: train TSS 0.996 (near-perfect), val TSS 0.0,
  test TSS 0.0. This is textbook overfitting on a tiny (278-row), narrow
  (one dominant active region, one week) training set — an honest,
  informative real result, not a failure of the code. It demonstrates
  exactly why `configs/splits.yaml`'s real run needs years of data per
  split, not a week.
- **LogisticRegression**: train TSS 0.793, val TSS 0.053, test TSS -0.006 —
  same overfitting pattern, less extreme.
- **PersistenceBaseline** (the simplified `feat_persistence_elevated_flux_run_steps`
  proxy — see `planning/MODELING_DOCUMENTATION.md`): test TSS 0.207, the
  best test-split TSS of any model here. On this pilot, "was flux already
  elevated" outperformed both classifiers on held-out data — plausible
  given the tiny training set, but a sample size of one 7-day window is far
  too small to draw a general conclusion from this ranking.
- **ClimatologyBaseline**: TSS 0.0 on every split by construction (predicts
  the constant train-set rate, which cannot discriminate positives from
  negatives — TSS being exactly 0 here is the expected, correct behavior
  for this baseline, not a bug).

None of these numbers should be cited as evidence of real forecasting
skill — the sample is one week, one dominant active region, and a
pilot-only split. Their value here is entirely as PROOF THE PIPELINE RUNS
CORRECTLY ON REAL DATA END TO END, including producing sensible,
non-fabricated, appropriately-noisy output at small scale.

## A real bug fixed during this run

`preprocessing/features.py`'s per-column preallocation loop (`out[col] =
np.nan` called once per lag/stat/variable) triggered pandas'
`PerformanceWarning: DataFrame is highly fragmented` repeatedly once the
real 234-column feature set was built — invisible in unit tests (which use
2-3 variable synthetic configs), only surfaced by running the real
`configs/features.yaml` variable list at real scale. Fixed by building all
new columns in one dict and `pd.concat`-ing them once instead of looping
individual assignments; verified the fix produces byte-identical results
on this same real pilot (re-ran after the fix, all numbers above
unchanged) and that the full test suite (105/105) still passes.

## What this run does NOT establish

- **Not a full-scale run.** 2010-2019 (`configs/solar_ingest.yaml`,
  `configs/sharp.yaml`) has not been ingested — this pilot is 7 days out of
  a ~9.5-year intended range. Runtime/storage/JSOC-row-limit behavior at
  full scale (`sharp_ingestion.py`'s documented "no automatic chunking"
  limitation) is still unverified.
- **Not a real model-quality result.** See "Real, honest results" above —
  the metrics are genuine but drawn from a sample far too small and too
  biased toward one active region's most active week to say anything about
  real forecasting skill.
- **Does not confirm `jsoc_email` genuinely needs to be a REGISTERED
  address.** This run's query succeeded without JSOC apparently checking
  it. `sharp_ingestion.py`'s `export`/`poll` code path (unused by this
  project, reserved for a possible future FITS-segment stage) would very
  likely still require a real registered address — not exercised here.
- **Does not validate `configs/splits.yaml`'s real 2010/2016/2018
  boundaries** — this run deliberately used its own pilot boundaries
  instead (see above).

## Where the real artifacts live

- `data_cache/goes/normalized/goes_xrs_normalized_20170904_20170910.csv`,
  `data_cache/sharp/normalized/sharp_normalized_20170904_20170910.csv` —
  real normalized ingestion output.
- `data_cache/{goes,sharp}/manifests/*.json` — real per-file/per-query
  provenance manifests (checksums, timestamps, HTTP/query status).
- `results/real_pilot_20170904_20170910/{dataset,train_dataset}.csv`,
  `feature_manifest.json`, `metrics.json` — the real joined dataset and
  the real training run's full metrics.
- `checkpoints/real_pilot_20170904_20170910/*.joblib` — the real fitted
  model objects (baselines and classical models alike).
