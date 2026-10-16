# Solar Labeling Documentation — Flare-Occurrence Targets

Status: **implemented, unit-tested (7/7 tests in
`tests/test_flare_labels.py`, part of 105/105 passing across the whole
`src/solar` package), and now also run against real data.** On 2026-07-25,
`build_primary_label` was run on a real 7-day GOES+SHARP ingestion
(2017-09-04..10, covering the 2017-09-06 X9.3 flare) — 745/745 rows got a
DEFINED label (full look-ahead coverage), with a real, correctly-identified
class distribution (485 M, 138 X, 122 C peaks in a 6h pilot horizon). See
`planning/REAL_PILOT_RUN_2017-09.md` for the full run and its caveats (a
shorter, pilot-only 6h horizon was used, not the 24h default — a 7-day
window can't support 24h look-ahead near its own edges). **Not yet run at
full scale** — same caveat as `planning/PREPROCESSING_DOCUMENTATION.md`.

## Scope

Covers `labeling/flare_labels.py`. Consumes the two outputs of
`preprocessing/alignment.py::build_aligned_timeline`: the SHARP-cadence
`aligned` DataFrame (label rows are produced one per row of this) and the
fine-grained (1-minute), disk-aggregated `goes_disk_series` (the source of
truth for whether a flare occurred, never the SHARP-coarsened series).

## Why the fine-grained GOES series, not the aligned one

SHARP's native cadence is 12 minutes. An X-class flare's impulsive rise can
peak and decay within a few minutes — well inside a single 12-minute SHARP
step. If labeling used the already-SHARP-coarsened `aligned` DataFrame's
trailing-window GOES summary, a flare peaking between two SHARP timestamps
could be under-counted or missed entirely, depending on exactly where the
trailing windows fall. Using the fine-grained 1-minute series directly for
the forward-looking label search avoids this by construction.

## Why labels are allowed to look into the future (and features are not)

Every module in `preprocessing/` (`alignment.py`, `missing_data.py`,
`features.py`) is strictly causal: a value at time `t` never depends on any
observation after `t`. Labels are the one deliberate, documented exception
— a supervised-learning target for "will a flare occur in the next N
hours" is definitionally a statement about the future relative to the
feature row it is attached to. `flare_labels.py` is kept in its own
`labeling/` package, never imported by anything in `preprocessing/`, so
this directionality can never be accidentally inverted by a stray import.

## Coverage-gated UNDEFINED labels — not fabricated negatives

A label is only trustworthy if the forecast window `(t, t+horizon]` had
enough real GOES observations to be confident no flare was missed. If the
ingestion window happens to end partway through a row's horizon (e.g. the
last rows of a dataset near the ingestion cutoff date), the true answer for
"did a flare occur" is genuinely unknown, not "no." `build_labels` computes
`label_coverage_fraction` (observed 1-minute steps ÷ expected steps in the
horizon) and leaves `label_max_flare_class_in_horizon` as `None` and every
`label_flare_{cls}_or_above` as `NaN` (never `0.0`) whenever coverage falls
below `1 - max_allowed_missing_coverage_fraction` (default: below 80%
coverage). This mirrors `preprocessing/missing_data.py`'s "do not fabricate
a row that lacks a defensible observation" convention, applied to labels
instead of features. Tested explicitly in
`test_build_labels_undefined_when_coverage_too_low`.

A downstream dataset-builder (not yet implemented — see "What is NOT
implemented" below) must drop or explicitly handle rows with undefined
labels before training; they must never be treated as negatives.

## Label definition

For each aligned row at time `t`, over the open-closed window
`(t, t+forecast_horizon_hours]`:

- `label_max_flare_class_in_horizon`: the highest flare class (`X` > `M` >
  `C` > `B`) whose `configs/labeling.yaml` threshold the window's peak
  `xrsb_flux_w_m2` reaches, or `None` if it never reaches even the `B`
  threshold (or the row's coverage is insufficient — see above).
- `label_flare_{b,c,m,x}_or_above`: four binary (1.0/0.0/NaN) columns, one
  per class, `1.0` iff the max class reached is that class or higher.
- `label_primary` (via `build_primary_label`): a convenience alias for
  `label_flare_{primary_target_class}_or_above` (`configs/labeling.yaml`'s
  `primary_target_class`, default `M` — the most common single target in
  the published flare-forecasting literature this project follows, e.g.
  "will an M-class-or-larger flare occur in the next 24h").
- `label_coverage_fraction`, `label_horizon_hours`: bookkeeping columns
  carried on every row for auditability.

Thresholds (`configs/labeling.yaml`'s `flare_class_thresholds_w_m2`) are
duplicated from, not cross-loaded from, `configs/goes.yaml` — a deliberate
choice (see that config file's header) so the label definition stays
self-contained and any intentional future divergence between the two must
show up as a visible diff, not a silent shared-state change.

`run_horizon_sensitivity` sweeps `sensitivity_horizons_hours` (default
`[6, 12, 24, 48]`), reporting per-class positive rate and coverage at each
horizon — useful for judging how horizon choice trades off label density
against how far in advance a forecast is useful, before committing to one
horizon for modeling.

## What is NOT implemented in this delivery

- Running this module against a real, full-scale ingested dataset (only
  unit-tested against small synthetic DataFrames so far).
- The dataset-builder that joins `preprocessing/features.py`'s feature
  columns with this module's label columns IS now implemented
  (`modeling/datasets.py`) — see `planning/DATASETS_DOCUMENTATION.md`.
  Chronological splitting is also implemented (`splitting/`) — see
  `planning/SPLITTING_DOCUMENTATION.md`, including its `embargo_hours`
  mechanism for this module's forward-looking label horizons specifically.
  Nothing yet chains dataset-building -> splitting -> model training on
  real data.
- Model training and evaluation.
- Per-active-region labeling (would require the same real AR-to-flare
  attribution catalogue discussed in
  `planning/PREPROCESSING_DOCUMENTATION.md`).
