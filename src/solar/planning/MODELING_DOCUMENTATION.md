# Solar Modeling Documentation — Baselines & Classical Models

Status: **implemented, unit-tested (126/126 across the whole `src/solar`
package), and now also run end-to-end on real data.** On 2026-07-25,
`run_task` fit and evaluated both baselines (climatology, persistence) and
both configured classical models (logistic regression, random forest) on a
real 745-row dataset built from a 7-day GOES+SHARP ingestion
(2017-09-04..10) — real checkpoints and a real metrics JSON were written.
See `planning/REAL_PILOT_RUN_2017-09.md` for the full numbers and an
honest read of them: the sample is far too small (278 train rows, one
dominant active region) to support any real forecasting-skill claim, and
the results show textbook overfitting (RandomForest: train TSS 0.996,
val/test TSS 0.0) — informative precisely because it demonstrates why
`configs/splits.yaml`'s real boundaries need years of data per split, not
a week.

**2026-07-26 additions** (Chapter 3.6.2/3.8 alignment, before the first
real full-scale run): `run_task` now (1) selects the decision threshold
tau* by maximizing a configurable metric (default TSS) on the VALIDATION
split only, per-model, falling back to `configs/task_flare_forecast.yaml`'s
static `decision_threshold` only when val is too small/single-class to
select from — every model's result records both the threshold used and
`decision_threshold_source` explaining which case applied; and (2) reports
coefficient (linear models) or `feature_importances_` (tree ensembles)
interpretability per classical model (`modeling/interpretability.py`),
`None` where the model exposes neither (e.g. KNN) rather than fabricating
one. See `modeling/metrics.py::select_threshold` and
`modeling/interpretability.py`. A real, full-scale (2010-2025) run is
documented in `planning/REAL_FULL_RANGE_RUN.md`.

## Scope

Covers `modeling/metrics.py`, `modeling/baselines.py`,
`modeling/classical_models.py`, `modeling/registry.py`, and
`modeling/train.py`. Consumes `modeling/datasets.py`'s output: a joined
feature+label dataset, a `feature_manifest` column list, and (via
`modeling.datasets.assign_split_with_embargo`) a leak-audited
train/val/test `split` column.

This is a **deliberately scoped-down** counterpart to
`martian/modeling/`. It covers the baseline and classical-model tiers only:

| martian/modeling/ | solar/modeling/ equivalent | status |
|---|---|---|
| `metrics.py` (classification half) | `metrics.py` | ported, unmodified formulas |
| `baselines.py` | `baselines.py` | `ClimatologyBaseline`/`PersistenceBaseline` ported; `SeasonalMeanBaseline` NOT ported (no analogue yet) |
| `classical_models.py` + `registry.py` | same | classifier-only (no regressor namespace — see `registry.py`'s header) |
| `train.py` | `train.py` | baselines + classical models only — no sequence (LSTM/GRU) tier, no `evaluation/`-package integration, no full `TrainingRecord` provenance schema |
| `sequence_models.py`, `sequence_feasibility.py` | — | not implemented (no torch dependency added yet) |
| `evaluation/` (confidence intervals, figures, failure analysis) | — | not implemented |
| `provenance.py` (full `TrainingRecord`) | — | not implemented; `train.py` writes a plain metrics JSON + joblib checkpoints instead |

## Why TSS/HSS are the right metrics here, not just a martian port

`modeling/metrics.py`'s `classification_metrics` is a direct, unmodified
port of `martian/modeling/metrics.py`'s classification half — but unlike
most of this project's martian-mirroring, this isn't just "reuse for
consistency": TSS (True Skill Statistic) and HSS (Heidke Skill Score) are
the DE FACTO standard verification metrics in the published
solar-flare-forecasting literature (Bloomfield et al. 2012 established TSS
as the standard; large forecast-method comparisons such as Barnes et al.
2016 rank primarily by TSS). Their presence here is independently
justified by the domain, not merely inherited.

## Baseline design — what's ported, what's not, and why

- **`ClimatologyBaseline(mode="train_mean")`**: the constant train-set
  positive rate. Only this one mode is registered — martian's `"column"`
  and `"zero"` modes are regression-target concepts (a climatology-provider
  column; a z-score's climatological value of exactly 0) with no analogue
  for a binary flare-occurrence target.
- **`PersistenceBaseline(source_column=...)`**: reads a configured column
  directly, clips to `[0, 1]`. `configs/task_flare_forecast.yaml` points it
  at `feat_persistence_elevated_flux_run_steps`
  (`preprocessing/features.py`'s run-length counter): clipping means 0 (no
  recent elevated-flux run) maps to probability 0, and ANY nonzero run maps
  to probability 1. This is a genuine, recognized flare-forecasting
  baseline family ("recent activity predicts near-term activity"), but this
  specific implementation is an explicitly-flagged SIMPLIFIED proxy, not a
  literature-calibrated persistence model — a real persistence baseline
  would more typically threshold the trailing flux level itself, not a
  run-length count. Revisit once real data is available to tune this
  properly.
- **`SeasonalMeanBaseline`**: NOT ported. Solar Cycle phase is the loose
  analogue of Mars's Ls-season, but no cycle-phase feature exists in this
  project yet — not fabricated here; add it to
  `preprocessing/features.py` first if this baseline is wanted later.

## `modeling/registry.py` — classifier-only, and why that's not a gap

Unlike `martian/modeling/registry.py` (which splits `REGRESSOR_FACTORIES`/
`CLASSIFIER_FACTORIES` behind a `task_type` parameter, because martian has
both regression and classification tasks), solar's registry has no such
split: `get_model(name, hyperparameters, seed)` only ever returns a
classifier, because **this delivery has exactly one modeling task**
(binary flare-occurrence forecasting via `configs/task_flare_forecast.yaml`).
This is a scope reflection, not a design compromise — if a regression task
is added later (e.g. forecasting peak flux magnitude directly, rather than
class-threshold occurrence), the registry should be split then, mirroring
martian's structure. Building that split speculatively now, for a task
that doesn't exist, was deliberately avoided.

## `modeling/train.py` — what it actually does

For each `target_columns` entry in the task config (default: just
`label_primary`; `configs/task_flare_forecast.yaml`'s header explains how
to also train each individual class threshold):

1. Restrict to metadata + `feature_manifest` + this one target column
   (sibling `label_flare_*_or_above`/`label_primary` columns are dropped —
   never accidentally along for the ride as a feature).
2. Drop rows where the target is UNDEFINED (`NaN` — coverage-gated, per
   `planning/LABELING_DOCUMENTATION.md`) — never trained as a fabricated
   negative.
3. Split into train/val/test using the `split` column
   `modeling.datasets.assign_split_with_embargo` already assigned on the
   full dataset (computed once, not per target column, since the embargo
   horizon is dataset-wide).
4. If train has fewer than 2 rows or test is empty: report
   `status: "insufficient_data"` and skip — never a crash, never a
   fabricated result.
5. Fit each configured baseline on the raw train dataframe; evaluate
   `classification_metrics` on train/val/test.
6. If train has only one class present for this target: skip classical
   models entirely (most sklearn classifiers cannot fit a single-class
   target) — reported explicitly via
   `classical_models_skipped_reason`, never silently dropped without a
   trace. Otherwise: median-impute (`SimpleImputer`, fit on TRAIN ONLY,
   same discipline as martian's `train.py`) and fit/evaluate each
   configured classical model the same way.
7. For each fitted model, predicted probabilities are computed on every
   non-empty split, then `modeling.metrics.select_threshold` sweeps
   `configs/task_flare_forecast.yaml`'s `threshold_selection.metric`
   (default TSS) over a grid on the VAL split's probabilities only — never
   train, never test (Chapter 3.6.2: "the test set plays no role in
   threshold selection") — to pick tau*. If val has fewer than 2 rows or
   only one class, tau* selection is impossible; `decision_threshold`
   (the static fallback) is used instead, and
   `decision_threshold_source` records which case happened. Every split's
   `classification_metrics` are then computed at that one threshold.
8. For classical models, `modeling.interpretability.extract_interpretability`
   records ranked coefficients (linear) or feature importances (tree
   ensembles) alongside the metrics — `None` for models exposing neither
   (KNN has no native per-feature attribution).
9. Every fitted model (baseline or classical) is checkpointed via
   `joblib.dump` to `configs/task_flare_forecast.yaml`'s `checkpoint_dir`,
   and the full metrics dict (now including `decision_threshold`,
   `decision_threshold_source`, `interpretability`) is written to
   `paths.results` as JSON.

No confidence intervals, no per-model comparison against a "primary"
baseline, no figures, no failure analysis, no SHAP/attribution tooling for
non-existent deep models — those all live in martian's separate
`evaluation/` package (or Chapter 3.6/3.8's deep-learning-specific
interpretability methods), neither of which has a solar counterpart yet.

## What is NOT implemented in this delivery

- Weighted BCE / focal loss (Chapter 3.7) — both are neural-network loss
  functions; `class_weight: balanced` on the sklearn classifiers
  (`configs/task_flare_forecast.yaml`) is the closest equivalent this
  classical-only pipeline has for cost-sensitive learning.
- Sequence models (LSTM/GRU) and the feasibility-report gating that decides
  whether they're worth attempting — no torch dependency in
  `requirements.txt` yet, added only if/when this is built.
- An `evaluation/` package (confidence intervals, required figure suite,
  failure analysis, baseline-comparison reporting) — mirror
  `martian/evaluation/` once real data exists to evaluate.
- A full `TrainingRecord`-style provenance schema
  (`martian/modeling/provenance.py`'s exact-parameters/seed/software-
  versions/convergence/runtime/hardware record per checkpoint) —
  `train.py` currently records only parameters and runtime_seconds inline
  in the results JSON.
- `SeasonalMeanBaseline` (needs a Solar Cycle phase feature first — see
  above).
- A regressor namespace in `modeling/registry.py` (not needed until a
  regression task exists).
