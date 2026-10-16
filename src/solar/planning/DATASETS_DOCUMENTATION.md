# Solar Modeling-Dataset Documentation — Feature/Label Join

Status: **implemented, unit-tested (11/11 tests in `tests/test_datasets.py`,
part of 105/105 passing across the whole `src/solar` package, including an
end-to-end test on synthetic data), and now also run on real data.** On
2026-07-25, `build_flare_forecast_dataset` and `assign_split_with_embargo`
both ran on the real output of a 7-day GOES+SHARP ingestion
(2017-09-04..10) — 745 real rows joined, 257-column real feature manifest,
`assign_split_with_embargo` correctly auto-derived its embargo horizon
(6h) from the real dataset's own `label_horizon_hours` column. See
`planning/REAL_PILOT_RUN_2017-09.md` for the full run. **Not yet run at
full scale** — same caveat as the other planning docs: the underlying
ingestion has not been run at full scale in this environment.

## Scope

Covers `modeling/datasets.py`, the sole bridge between
`preprocessing/features.py` (strictly causal features) and
`labeling/flare_labels.py` (forward-looking labels). Nothing in
`preprocessing/` imports `labeling/`, and nothing in `labeling/` imports
`preprocessing/features.py`'s feature machinery — this module is
deliberately the only place the two are combined, so the causal/
non-causal boundary the rest of the package maintains can't be blurred by
an unexpected import elsewhere.

## Why an inner join on `time_utc` is enough to enforce "forecast from a real observation"

`features_df` (`preprocessing/features.py`'s output) is built on the
TRAILING-FILLED series (`preprocessing/missing_data.py`'s output), so it
can contain rows for timestamps that were never genuinely observed
(LOCF-filled gaps) — see `planning/PREPROCESSING_DOCUMENTATION.md`.
`labeled_df` (`labeling/flare_labels.py`'s output) is built directly on the
PRE-FILL aligned series (`build_labels` takes `alignment.py`'s `aligned`
DataFrame, never the filled one — see `planning/LABELING_DOCUMENTATION.md`),
so it only ever contains genuinely observed timestamps.

An inner join of the two on `time_utc` therefore automatically restricts
the joined dataset to rows whose FORECAST ORIGIN is a real observation,
with no extra `is_observed` filter needed — mirroring
`martian/modeling/datasets.py`'s documented principle "you can only
forecast FROM a moment you actually measured, using whatever historical
context (real or gap-filled) is available." A filled-only row's rich
feature history (built from real trailing-filled context) is simply
discarded along with it, since there is no corresponding label to pair it
with. Tested explicitly in
`test_join_restricts_to_genuinely_observed_forecast_origins`.

## Column collision guard

`build_flare_forecast_dataset` raises `ValueError` if `features_df` already
contains any `label_`-prefixed column before the join — this should be
structurally impossible given `preprocessing/features.py`'s own
leakage-guarantee docstring, but the check makes that guarantee an
enforced invariant rather than an assumed one, and turns a silent
column-overwrite bug into an immediate, legible failure. It does NOT flag
the many legitimate raw-column overlaps between the two inputs (e.g.
`USFLUX`, `R_VALUE` both appear in `features_df` and in `labeled_df`,
since `labeled_df` is built by appending label columns onto a copy of the
aligned DataFrame) — only `label_`-prefixed columns from `labeled_df` are
ever pulled into the merge.

## What gets carried on the final dataset

- `feature_manifest` (returned explicitly, never re-derived by prefix
  guessing downstream): `feature_column_names(merged)` (`feat_`-prefixed
  engineered columns) plus the raw current-state SHARP/GOES columns listed
  in `configs/features.yaml`'s `variables` (legitimate, strictly causal
  "current state" features — same status as `martian/modeling/datasets.py`'s
  `CURRENT_STATE_RAW_COLUMNS`, just sourced from config here instead of
  hardcoded, to avoid duplicating that list by hand).
- `target_cols`: the four binary `label_flare_{b,c,m,x}_or_above` columns
  plus `label_primary`, whichever are present.
- ID/bookkeeping metadata (never a feature, never a target):
  `time_utc`, `is_observed`/`interpolation_flag`/`gap_length_steps`
  (`preprocessing/missing_data.py`'s QC indicators — always `True`/`0` for
  the surviving rows post-join, but carried for auditability), and
  `label_max_flare_class_in_horizon`/`label_coverage_fraction`/
  `label_horizon_hours` (`labeling/flare_labels.py`'s bookkeeping columns).

## Undefined labels are never silently trained against

Per `planning/LABELING_DOCUMENTATION.md`, a `label_*` column can be `NaN`
(insufficient forward coverage) — never fabricated as `0.0`.
`drop_rows_with_undefined_label(dataset, target_col)` is the explicit,
named operation a caller must run before training on a specific target;
there is no implicit auto-drop inside `build_flare_forecast_dataset`
itself, so a caller who forgets gets an obviously-wrong dataset (still
containing `NaN` targets) rather than a silently-shrunk one.
`summarize_targets` reports `n_total`/`n_defined`/`n_positive`/
`positive_rate` per target as a sanity check before training — e.g. a
target that ends up entirely undefined after a bad join is caught here as
`n_defined == 0`, `positive_rate == NaN`, rather than surfacing later as a
confusing training-time crash.

## Chaining into `splitting/` — `assign_split_with_embargo`

`assign_split_with_embargo(dataset, splits_config=None, *, horizon_hours=None)`
runs `splitting.chronological_split.assign_split` immediately followed by
`apply_embargo`, so a caller can never end up with an un-embargoed split by
forgetting the second call. Its one notable design choice: `horizon_hours`
defaults to being READ FROM the dataset's own `label_horizon_hours` column
(present because `build_flare_forecast_dataset` carries
`labeling/flare_labels.py`'s bookkeeping columns through), not from
`configs/splits.yaml`'s static `embargo_hours` default. This means the
embargo always matches whatever horizon actually produced this specific
dataset's labels, even if `configs/labeling.yaml`'s
`forecast_horizon_hours` is changed for a later run without also updating
`configs/splits.yaml`. Raises `ValueError` if that column is missing (not a
flare-forecast dataset) or carries more than one distinct value (the
embargo mechanism assumes one horizon per dataset — pass `horizon_hours`
explicitly to bypass this check for a mixed-horizon dataset, which is not
otherwise something this module constructs).

`modeling/train.py` now calls this once per dataset (not once per target
column) — see `planning/MODELING_DOCUMENTATION.md`.

## What is NOT implemented in this delivery

- Running this module against a real, full-scale ingested dataset (only
  unit-tested against small, hand-constructed DataFrames so far, including
  one that runs the real alignment/missing-data/features/labeling chain).
- `modeling/` beyond dataset construction and `assign_split_with_embargo` —
  baseline and classical models ARE now implemented
  (`planning/MODELING_DOCUMENTATION.md`); sequence models and an
  `evaluation/` package are not.
