# Solar Splitting Documentation — Leak-Audited Chronological Train/Val/Test

Status: **implemented, unit-tested (117/117 tests passing across the whole
`src/solar` package), and `configs/splits.yaml`'s real boundaries are now
validated against the real, full-scale (2010-05-01..2025-07-24) ingested
GOES+SHARP dataset — see planning/REAL_FULL_RANGE_RUN.md.**

On 2026-07-25, `assign_split`/`apply_embargo` ran on a real 7-day
GOES+SHARP dataset (2017-09-04..10) using its own PILOT-ONLY split config
(not `configs/splits.yaml`'s real boundaries) — see
`planning/REAL_PILOT_RUN_2017-09.md`.

On 2026-07-26, once full-range ingestion was complete, `configs/splits.yaml`'s
real boundaries were checked against actual data coverage and found broken:
the original placeholder (train 2010-2015, val 2016-2017, test 2018-2019)
put essentially zero flare activity in test (0 X-class, 17 M-class-or-above
minutes across two full years — the deep Cycle 24 minimum) and silently
excluded 2020-2025 (`assign_split` marks it `"unassigned"`) even though that
excluded period alone contains more GOES rows and more M-class-or-above
activity (92,551 minutes) than the original train+val+test combined
(27,572 minutes) — despite the ingestion configs having been deliberately
extended to 2025-07-24 specifically to include Solar Cycle 25 (Chapter
3.3.1's test period). Boundaries were corrected to
train=2010-05-01..2016-12-31, val=2017-01-01..2019-12-31,
test=2020-01-01..2025-07-24 (test now genuinely = Solar Cycle 25 only, as
the thesis specifies); see `configs/splits.yaml`'s header comment for the
full rationale, including why val was widened to include 2017 (the
2017-09-06 X9.3 flare year) rather than left at the near-flare-free
2018-2019 tail of Cycle 24. Fixing this surfaced a real test-isolation gap:
several tests called `load_splits_config()` with no arguments, silently
depending on `configs/splits.yaml`'s real (evolving) values instead of a
self-contained synthetic config — `tests/test_chronological_split.py`,
`tests/test_datasets.py`, and `tests/test_train.py` now each define their
own local `_TEST_SPLITS_CONFIG`/`_synthetic_dataset` fixtures so they no
longer break every time the real research boundaries are tuned.

## Scope

Covers `splitting/chronological_split.py`. Consumes a DataFrame carrying a
`time_utc` column — typically the output of joining
`preprocessing/features.py`'s feature columns with
`labeling/flare_labels.py`'s label columns on that column (that join itself
is not yet implemented — see "What is NOT implemented" below).

## Why plain date cutoffs, not a (mars_year, sol)-style calendar

`martian/splitting/chronological_split.py` splits on Mars's cyclical
`(mars_year, sol)` calendar. GOES/SHARP have no such structure — just a
continuous UTC timeline — so `configs/splits.yaml` uses plain
`train_start`/`train_end`/`val_start`/`val_end`/`test_start`/`test_end`
date strings instead. `load_splits_config` validates at load time that the
three ranges are non-overlapping and strictly chronologically ordered
(train entirely before val entirely before test), raising `ValueError`
immediately if not — mirrors `preprocessing/quality_control.py`'s
load-time config validation convention (there, contiguous rule-id
checking).

**The chosen boundaries are placeholders**, not fit to a real dataset's
actual coverage (SHARP has not been ingested at scale in this environment —
see `planning/INGESTION_DOCUMENTATION.md`). They must be revisited once real
ingestion output exists and its true date range is known.

## The embargo mechanism — why it exists

`preprocessing/` is strictly causal, so features carry no forward-leakage
risk across a split boundary. Labels are different: `labeling/flare_labels.py`
deliberately looks forward over `(t, t+horizon]`. A row sitting near the END
of the train period can therefore have a label informed by GOES observations
that fall inside the val period — not a feature-input leak, but the
standard **purged / embargoed time-series cross-validation** concern
(well-established practice for horizon-based labels in financial and
scientific time series): a train-split row's target blurs the
independence the boundary is supposed to guarantee if it depends on
val-period data.

`find_rows_with_label_horizon_crossing_split_boundary` detects exactly
this (mirrors `martian/splitting/chronological_split.py`'s
`find_episodes_crossing_split_boundary` — same find/assert pairing:
`assert_no_label_horizon_crosses_split_boundary` raises
`LabelHorizonCrossesSplitBoundaryError` on the same input, for a caller
that wants a hard failure instead of a manual check).
`apply_embargo` is the actual fix: it reassigns every crossing row's split
to `"embargo"` (excluded from train/val/test entirely), after which the
assert is guaranteed to pass on the same data. `configs/splits.yaml`'s
`embargo_hours` defaults to 24, matching `configs/labeling.yaml`'s default
`forecast_horizon_hours` — if a non-default label horizon is used, that
horizon must be passed explicitly to these functions (they do not
cross-load `configs/labeling.yaml`, for the same "self-contained config"
reason documented in `configs/labeling.yaml`'s own header).

## Train-only standardization

`fit_standardization_stats`/`apply_standardization` are a direct,
unmodified port of `martian/splitting/chronological_split.py`'s equivalent
functions — mean/std computed only from whatever DataFrame is passed in
(the caller's responsibility to pass only the train split), applied to any
DataFrame via the returned stats dict. No behavior differs between the two
packages here, so no solar-specific rationale is needed beyond "already
proven correct in martian, reused as-is."

## What is NOT implemented in this delivery

- Running the full features->labels->dataset->split->model pipeline
  end-to-end against the real, full-scale (2010-2025) ingested dataset —
  `configs/splits.yaml`'s boundaries are now validated against real data
  coverage (see above), but the only real end-to-end modeling run so far is
  still the 7-day 2017-09 pilot (`planning/REAL_PILOT_RUN_2017-09.md`); the
  module itself is otherwise only unit-tested against small synthetic
  DataFrames.
- `modeling.datasets.assign_split_with_embargo` now chains this module's
  `assign_split`/`apply_embargo` directly onto `modeling/datasets.py`'s
  joined feature+label dataset (deriving the embargo horizon from the
  dataset's own `label_horizon_hours` column) — see
  `planning/DATASETS_DOCUMENTATION.md`. `modeling/train.py` now calls it
  end-to-end on synthetic data — see `planning/MODELING_DOCUMENTATION.md`.
  Nothing has run this on a REAL ingested dataset yet.
- Sequence models and an `evaluation/` package — see
  `planning/MODELING_DOCUMENTATION.md`'s "What is NOT implemented" section
  for the precise boundary of what `modeling/` currently covers.
