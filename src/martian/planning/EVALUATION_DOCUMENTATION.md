<!-- AUTO-MAINTAINED (by hand). Update when evaluation/*.py, modeling/train.py's
     evaluation wiring, or configs/evaluation.yaml change. -->

# Evaluation Documentation (per-model, per-horizon, per-split reporting)

**Status: implemented, tested, and actually run against real data.** Every
baseline and classical model, for every task (2 tasks now, `dust_regression`
and `storm_detection` — the former `temperature_regression` was removed
entirely, see `planning/MODELING_DOCUMENTATION.md`), every horizon, is
evaluated on every split it has (train/val/test) with confidence intervals,
the full required figure suite (test split), failure analysis across nine
dimensions (test split), and a comparison against the task's "appropriate
baseline." Sequence models get metrics + CIs on train/test but skip the full
figure/failure-analysis battery — see §5's scope note.

Called from `modeling/train.py::run_task`, immediately after each model is
fit, via `evaluation/evaluate_task.py::evaluate_and_report` — the
already-fitted model/imputer/train-val-test frames are reused directly
rather than reloaded from a checkpoint, which would otherwise require
persisting the fitted imputer separately (it currently is not).

## 1. Metrics, every split

`modeling/metrics.py`'s `regression_metrics`/`classification_metrics`
(already documented in `planning/MODELING_DOCUMENTATION.md`) are computed on
train, val, and test for every baseline and classical model — not just
test. `classification_metrics` additionally gets a `prevalence` field
(mean of `y_true`) attached per split. Classification reports carry TP/FP/
TN/FN (from `modeling/metrics.py`'s confusion counts), prevalence, TSS,
HSS, POD, FAR, precision, recall, F1, Brier score, log score, PR-AUC, and
ROC-AUC — everything this task's classification-report requirement lists.
Regression reports carry MAE, RMSE, MSE, R², bias, and correlation.

## 2. Confidence intervals (`evaluation/confidence_intervals.py`)

Bootstrap, 200 replicates (`configs/evaluation.yaml`), 90% CI by default,
**block-by-cell** rather than a plain i.i.d. row bootstrap: nearby cell-sols
are spatially/temporally correlated, so resampling whole cells (with
replacement) is a closer approximation to independent resampling than
resampling individual rows. This is an explicit ENGINEERING DECISION,
adapting an originally-planned single-time-series temporal block to our
data's actual shape — a (cell x sol) panel — by blocking on `cell_id`
instead; RESEARCHER VERIFICATION REQUIRED if a literal temporal block is
intended. A bootstrap replicate with only one class present contributes no
value to an AUC-based metric's CI (excluded, not treated as 0 or crashing)
— the same convention `modeling/metrics.py` already uses for the point
estimate.

## 3. Required figures, every one with provenance (`evaluation/figure_provenance.py`)

Every figure gets a **stable figure number** ("4.N", continuing the
existing Figure 4.3/4.7 sequence — starts at 4.8 — via a persistent
`results/figure_registry.json` keyed by task+model+horizon+split+kind, so
re-running the pipeline never renumbers a figure), a **descriptive
caption**, **units**, **sample size**, **data split**, **horizon**,
**data-source label**, and **script+config provenance** — stamped visibly
on the figure itself (so the PNG is self-describing even in isolation) AND
written to a companion `<name>.provenance.json`. Every figure also gets a
companion `<name>.csv` with the exact data it was drawn from — the
**generated source-data CSV** this task requires.

Classification figures (`evaluation/classification_figures.py`): confusion
matrix (TP/FP/TN/FN as both an annotated heatmap and its source table),
reliability diagram **with explicit bin counts** (an empty bin reports
`count=0, observed_frequency=NaN` — never fabricated), precision-recall
curve, ROC curve.

Regression figures (`evaluation/regression_figures.py`): observed-vs-
predicted (with a 1:1 reference line), and residual analysis (residuals
vs. predicted, plus a residual histogram with mean/std in the title).

A real figure set was produced across both tasks, all horizons, all
baseline+classical models in the completed real run — each with a
provenance JSON correctly stamped `data_source: "REAL (...)"` (see
`CHAPTER_4_PARAMETER_VALUES.md` caveat 6 for the labeling bug this required
fixing).

## 4. Baseline comparison (`evaluation/baseline_comparison.py`)

Every classical (and sequence) model's TEST metrics are compared against
its task's **"appropriate baseline"** (`configs/evaluation.yaml`'s
`primary_baseline_by_task`): **persistence** for `dust_regression` (the
conventional forecast-skill reference for a continuous target) and
**climatology** for `storm_detection` (the conventional no-skill reference
for a binary event). The comparison table includes a `relative_skill`
column and a `model_beats_baseline` boolean, with metric direction handled
correctly (MAE/RMSE/MSE/Brier/log_score/FAR are lower-is-better; everything
else is higher-is-better) — proven by
`tests/test_baseline_comparison.py::test_lower_is_better_metric_skill_sign`
and its higher-is-better counterpart. Baselines themselves are not compared
against a baseline (`baseline_metrics=None` when evaluating a baseline
model) — that comparison would be circular.

## 5. Failure analysis (`evaluation/failure_analysis.py`) — nine dimensions

Season, latitude band (30° bins), local time (6-hour bins), retrieval
density (`observation_count`, configurable bin edges), interpolation
status, uncertainty (`uncertainty_summary`, quartile bins), event category,
label persistence, and spatial extent — run on the TEST split for every
baseline and classical model. **Forecast horizon** is satisfied by this
whole module running once per horizon (1/3/5 sol), not as an eleventh
binning column within a single horizon's report.

Two scope decisions, made explicit rather than silently assumed:

- **Interpolation status is degenerate by construction for both tasks** —
  every evaluated row is, by design
  (`modeling/datasets.py::_restrict_to_observed_base_rows`), a genuinely
  observed current sol. The breakdown still runs and reports a single
  `"observed"` category (`tests/test_failure_analysis.py::test_breakdown_by_interpolation_status_degenerate_when_all_observed`)
  rather than being silently skipped — an honest null result, not a
  fabricated one.
- **Event category / label persistence / spatial extent are `storm_detection`
  only**, and are operationalised as the CURRENT sol's own episode
  membership (via `episode_id`, carried through from
  `labeling/anomaly_labels.py`), not the future episode(s) that may drive a
  positive horizon target — linking every positive-target row to whichever
  future episode(s) fall inside its horizon window would be a materially
  larger undertaking for a marginal interpretive gain. Rows with no active
  current-sol episode fall in a `"quiet"` category rather than being
  dropped.

**Real, informative finding from the completed real run** (`random_forest_classifier`,
storm_detection, h=5,
`tables/storm_detection/failure_analysis_random_forest_classifier_h5_event_category.csv`):
TSS is **0.212 for cell-sols currently inside a "local" episode vs. 0.076 for
"quiet" cell-sols** — the classifier genuinely does better when there is
active storm structure to detect, exactly the kind of signal this
failure-analysis requirement exists to surface (the "regional"/"unclassified"
categories show high recall but poor precision — few rows, and the model
appears to over-predict positive there). Season breakdown shows a real,
substantial spread: TSS 0.257 in Northern Autumn (the largest season by row
count, 43,495 rows) vs. 0.180 in Northern Summer (7,111 rows) vs. essentially
no signal in Northern Spring (only 15 rows in this real run — real coverage
sparsity in that season, not a modeling artefact). Real seasonal skill
variation is itself a genuine, reportable finding.

## 6. Two real bugs caught only by running the real pipeline

1. **`failure_analysis.py`'s breakdown functions originally assumed their
   required column (`latitude`, `local_true_solar_time`, etc.) always
   existed**, crashing with a `KeyError` the first time `evaluate_and_report`
   ran against a synthetic test fixture lacking one. Fixed by making every
   breakdown function check for its column first and return an explicit
   empty (`_NOT_APPLICABLE`) table otherwise — the same "honest null result,
   never a crash, never a fabrication" discipline used everywhere else in
   this pipeline. Regression-tested
   (`tests/test_failure_analysis.py::test_breakdown_by_season_missing_column_returns_empty`).
2. **Figures were never closed** (`evaluation/figure_provenance.py`'s
   `save_figure_with_provenance` called `fig.savefig` but never
   `plt.close(fig)`), triggering matplotlib's "More than 20 figures have
   been opened" `RuntimeWarning` and leaking memory across a run that
   creates 151 figures. Fixed by closing every figure immediately after
   saving it. Regression-tested
   (`tests/test_figure_provenance.py::test_save_figure_with_provenance_closes_the_figure`).

## 7. Test coverage

`tests/test_confidence_intervals.py` (6) — CI contains the point estimate,
narrows with more data, block bootstrap behaviour, single-class-replicate
exclusion from AUC CIs, and config-driven `n_replicates`.

`tests/test_figure_provenance.py` (6) — stable figure numbering across
calls, incrementing for new keys, continuing from an existing registry, the
PNG/CSV/JSON triple, and the figure-close fix.

`tests/test_classification_figures.py` (5), `tests/test_regression_figures.py`
(3) — hand-computed confusion-matrix counts, reliability-diagram bin counts
summing to n with empty bins reporting NaN, PR/ROC curve source columns,
and hand-computed residuals.

`tests/test_failure_analysis.py` (14) — every breakdown dimension,
including the missing-column defensive path, the retrieval-density bin
hand-computation, the degenerate interpolation-status case, and the
event-category/label-persistence/spatial-extent "quiet" fallback.

`tests/test_baseline_comparison.py` (6) — skill-sign correctness for both
metric directions, FAR treated as lower-is-better, and NaN handling.

`tests/test_train.py` — `run_task`'s evaluation wiring: every model's
evaluation record carries per-split CIs and a `baseline_comparison` table,
and the full figure set is written per model (not just one "primary model"
figure per horizon, as the pre-evaluation version did).

Full suite (all of `src/martian`): 316 passed as of the last real-data run
(2026-08-05).
