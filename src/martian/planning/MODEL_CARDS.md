<!-- AUTO-MAINTAINED (by hand). Update when a configs/task_*.yaml model
     hierarchy, hyperparameter set, or target definition changes. -->

# Model Cards

One card per task. **Two tasks now** — the former Task B
(`temperature_regression`) was removed entirely (no real temperature
climatology exists, see `CHAPTER_4_PARAMETER_VALUES.md` caveat 5), not just
left out of this document. Both remaining tasks are FULLY SEPARATED — own
target, own dataset, own feature manifest, own checkpoints, own seed, own
results, own figures (`planning/MODELING_DOCUMENTATION.md` carries the full
cross-task-leakage argument; `mars/leakage_checks.py` re-verifies it at
runtime on every `mars.run_all` run). Every model in every task hierarchy
is trained and evaluated at every configured horizon; nothing here is
cherry-picked. Real result summaries: `CHAPTER_4_PARAMETER_VALUES.md`
Section 8.

---

## `dust_regression` (`configs/task_dust_regression.yaml`)

**Type:** regression. **Seed:** 42. **Horizons:** 1, 3, 5 sols.

**Target:** `target_dust_optical_depth_h{h}(c,s) = dust_optical_depth(c, s+h)`
— future AGGREGATED dust optical depth, read from a genuinely observed
(never trailing-filled) future cell-sol only; a row with no exact observed
match at `s+h` is dropped, never fabricated.

**Metrics:** MAE, RMSE, MSE, R², bias, correlation (+ bootstrap 90% CI on
each, block-by-cell — `configs/evaluation.yaml`).

| Tier | Model | Hyperparameters |
|---|---|---|
| Baseline | `climatology` | `column: dust_optical_depth_mu` (real Montabone-built climatological mean) |
| Baseline | `persistence` | `source_column: dust_optical_depth` (last observed value carried forward) |
| Baseline | `seasonal_mean` | (Ls-binned historical mean) |
| Classical | `random_forest_regressor` | `n_estimators=200, max_depth=8, min_samples_leaf=3` |
| Classical | `gradient_boosting_regressor` | `n_estimators=200, max_depth=3, learning_rate=0.05` |
| Sequence | `lstm` | `hidden_size=16, num_layers=1, dropout=0.0`; window 10 primary (3/5/10 sensitivity) |
| Sequence | `gru` | same architecture shape as `lstm` |

Sequence training: 30 epochs, `lr=0.001`, early-stopping patience 5, with a
sequence-feasibility report (`sequence_feasibility_report.csv`) recording
how many windows were actually constructable per horizon. **Real run: all
three window lengths (3/5/10) were feasible**, unlike earlier simulated-data
runs where window=10 was infeasible at every cell — see
`planning/MODELING_DOCUMENTATION.md` §3.

**Real result** (best model overall: `gradient_boosting_regressor`, h=5 sol,
R²=0.122, MAE=0.105 — the only positive R² across the whole task):
`CHAPTER_4_PARAMETER_VALUES.md` Section 8.

**Known scope note:** the causal Savitzky-Golay-filtered dust series
(pipeline Step 6) is generated and persisted but is NOT one of this task's
input features — see `planning/PIPELINE_DOCUMENTATION.md`'s scope-boundary
section.

---

## `storm_detection` (`configs/task_storm_detection.yaml`)

**Type:** binary classification. **Seed:** 44. **Horizons:** 1, 3, 5 sols.
**Decision threshold:** 0.5.

**Target:** `target_storm_horizon_h{h}(c,s)` = reused directly from
`labeling/anomaly_labels.py`'s `horizon_{h}sol` column (single source of
truth, never recomputed independently here) — "does a qualifying
`storm_episode` occur at this cell within the next h sols." This is EVENT
DETECTION (does a storm occur), never event classification (which
taxonomy class, `labeling/storm_taxonomy.py`, descriptive only).

**Metrics:** TSS, HSS, POD, FAR, precision, recall, F1, Brier score, log
score, PR-AUC, ROC-AUC (+ bootstrap 90% CI, block-by-cell). **FAR is the
meteorological false-alarm RATE** (FP / (FP+TN)), not "false alarm ratio"
(FP / (FP+TP)) — **RESEARCHER VERIFICATION REQUIRED** if the thesis text
uses "FAR" to mean the ratio instead.

| Tier | Model | Hyperparameters |
|---|---|---|
| Baseline | `climatology` | `mode: train_mean` (constant base rate) |
| Baseline | `persistence` | `source_column: atmospheric_anomaly` |
| Baseline | `seasonal_mean` | (Ls-binned historical event rate) |
| Classical | `logistic_regression` | `max_iter=500, class_weight=balanced` |
| Classical | `random_forest_classifier` | `n_estimators=200, max_depth=8, min_samples_leaf=3, class_weight=balanced` |
| Classical | `gradient_boosting_classifier` | `n_estimators=200, max_depth=3, learning_rate=0.05` (no `class_weight` support — degrades under severe class imbalance, an expected consequence not a defect) |
| Classical | `knn_classifier` | `n_neighbors=15` (no `random_state` — deterministic given data/hyperparameters; seed threaded through unused, for interface uniformity only) |
| Sequence | `lstm` / `gru` | same architecture/training as `dust_regression` |

**Real result** (best model overall: `random_forest_classifier`, h=5 sol,
TSS=0.244, ROC-AUC=0.769 — clearly beats the `persistence` baseline's
TSS=0.203; GRU sequence models also beat baselines at longer horizons):
`CHAPTER_4_PARAMETER_VALUES.md` Section 8. Real per-episode-category and
per-season skill breakdown: `planning/EVALUATION_DOCUMENTATION.md` §5.

---

## Common across both cards

- **Model-hierarchy rationale:** every task always trains baselines FIRST,
  then classical models, then sequence models, and every one of them is
  evaluated (never just the "best" one) — this is what lets
  `evaluation.evaluate_task`'s baseline-comparison table be a genuine
  comparison rather than a cherry-picked one.
- **No model here uses future information**: every input feature is
  causal by construction (`planning/FEATURE_DICTIONARY.md`), every target
  is read from a genuinely observed future cell-sol (regression) or a
  label built without look-ahead (classification), and chronological
  train/val/test splits plus episode-boundary exclusion
  (`splitting/chronological_split.py`) prevent any temporal leakage. See
  `mars/leakage_checks.py` for the runtime re-verification of this on
  every `mars.run_all` run — **passed** in the completed real run.
- **Uncertainty**: every headline metric ships with a bootstrap 90% CI,
  resampled at the CELL level (not the row level) so spatial
  autocorrelation within a cell's own sol-series does not inflate
  apparent precision — `configs/evaluation.yaml`.
