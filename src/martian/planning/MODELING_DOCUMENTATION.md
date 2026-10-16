<!-- AUTO-MAINTAINED (by hand). Update when preprocessing/features.py,
     splitting/chronological_split.py, modeling/*.py, or
     configs/features.yaml, configs/splits.yaml, configs/task_*.yaml change. -->

# Modeling Documentation (two fully separated tasks, full model hierarchy)

**Status: implemented, tested, and actually run against real data.** Builds on the
aggregated cell-sol dataframe and the anomaly/storm-episode labels; adds shared
causal feature engineering, a real leak-audited chronological split, and **two**
independently-configured, independently-seeded modeling tasks — the former Task B
(`temperature_regression`) was removed entirely (no real temperature climatology
exists, see `CHAPTER_4_PARAMETER_VALUES.md` caveat 5), not deferred or left broken.
Each remaining task runs the FULL model hierarchy: baselines (climatology,
persistence, seasonal mean) → classical models (logistic regression, random forest,
gradient boosting, KNN where appropriate) → sequence models (LSTM, GRU), the last
gated by an always-run, always-reported feasibility check.

**Every trained model, of every kind, gets its own checkpoint and its own
provenance record** (`modeling/provenance.py`): exact parameters, random
seed, software versions, training history, feature order, train/val sample
counts, convergence status, runtime, and hardware info. See §7.

## 1. The two tasks

| | Task 1 | Task 2 |
|---|---|---|
| Name | `dust_regression` | `storm_detection` |
| Type | Regression | Binary classification |
| Target | `dust_optical_depth(c, s+h)` | `horizon_{h}sol(c,s)` (any storm_episode in (s, s+h]) |
| Target source | Genuinely observed future cell-sol only | Reused directly from `labeling/anomaly_labels.py` (single source of truth) |
| Horizons | 1, 3, 5 sols | 1, 3, 5 sols |
| Seed | 42 | 44 |
| Config | `configs/task_dust_regression.yaml` | `configs/task_storm_detection.yaml` |

There is no seed 43 / `temperature_regression` config anywhere in this repository.
Each task has its own dataset, feature manifest, checkpoints (task-scoped
subdirectory — see `planning/PIPELINE_DOCUMENTATION.md`'s output-structure note),
results, figures, and seed (its own config's `seed` field — never a shared global
default).

## 2. Model hierarchy (every task runs all of it) — exact hyperparameters in `CHAPTER_4_PARAMETER_VALUES.md` Section 9

**Baselines** (`modeling/baselines.py`, config-driven per task):
- `climatology` — `column` mode for `dust_regression` (predicts
  `dust_optical_depth_mu`, the real MCD climatology's own value), `train_mean` for
  `storm_detection` (the constant base rate).
- `persistence` — predicts the current sol's own value/state as the future
  value/state (source column set per task: `dust_optical_depth`, `atmospheric_anomaly`).
- `seasonal_mean` — per-Ls-season mean of the target, fit on TRAIN ONLY.

All three are dataframe-aware (`uses_raw_dataframe = True`), not fit on the
imputed numeric feature matrix — they read specific named columns directly.

**Classical models** (`modeling/classical_models.py`, `modeling/registry.py`):
`logistic_regression` (`storm_detection` only), `random_forest_regressor`/
`random_forest_classifier`, `gradient_boosting_regressor`/
`gradient_boosting_classifier`, `knn_classifier`. KNN is registered for
**classification only**; requesting `knn_classifier` (or any classifier) for a
regression task raises `TaskModelMismatchError` (`tests/test_registry.py`) rather
than silently substituting a different family.

**Sequence models** (`modeling/sequence_models.py`): LSTM and GRU
(`torch.nn.LSTM`/`GRU` + a linear head), trained on windows built by
`build_sequence_dataset` from the SAME trailing-filled feature series as the
tabular path. A window is dropped only for a genuine sol gap (never
bridgeable); NaN feature values within an otherwise-valid window are
imputed downstream (median, fit on train only, entirely-NaN columns
fall back to 0 rather than being dropped — see §8 for why that distinction
matters). All hyperparameters and training settings come from each task's own
config, never hardcoded.

## 3. Sequence-model feasibility (`modeling/sequence_feasibility.py`) — ALWAYS run first

Before any LSTM/GRU is trained, `run_feasibility_report`:
1. computes every consecutive-valid-sol run length, PER GRID CELL, in the
   trailing-filled series;
2. compares each run length against every configured window length (primary
   + sensitivity, deduplicated);
3. reports mean, median, p25/p75/p90/p95, and maximum — written to
   `metrics/<task_name>/sequence_feasibility_report.csv` for every task,
   every run, whether or not the result is favourable.

**Primary window: 10 sols. Sensitivity windows: 3, 5, 10 sols.**

**Real result, from the completed real run**: windows 3, 5, and 10 were ALL
feasible for both tasks at real data density (unlike the earlier simulated-data
runs, where window=10 was infeasible for every cell) — reported per-window
training-window counts range from ~13,000 (dust_regression, window=10) to
~55,000 (storm_detection, window=3). See `CHAPTER_4_PARAMETER_VALUES.md` Section 8's
model performance table for the actual trained results at each window.

## 4. Shared causal feature engineering (`preprocessing/features.py`, `configs/features.yaml`)

One shared builder produces 32 `feat_`-prefixed columns per cell-sol (real run
count) from `dust_optical_depth`/`_z` only (dust-only — no
`surface_temperature_k` feature family, see `planning/FEATURE_DICTIONARY.md`):
current state, lags 1-3 sols, trailing-5-sol mean/std/max/min, first
difference, slope, a persistence-run counter, spatial-neighbour mean/max/
exceedance-count (8 neighbours, SAME sol only), and Ls/LTST harmonics.
Every feature at (cell, sol=s) reads only rows with sol<=s for the SAME
cell, or sol==s for a SPATIAL neighbour — proven by a future-perturbation
test. Built from the trailing-filled series (denser than raw observations,
still strictly causal). `preprocessing.features.FEATURE_PREFIX = "feat_"`
and the module never produces a `target_`-prefixed column.

## 5. Real vs. fabricated: the target-construction rule, and cross-task leakage prevention

`dust_regression` requires an EXACT future-sol match in a genuinely observed
cell-sol (never trailing-filled); a row without one is dropped, never
interpolated. `storm_detection`'s target is copied unchanged from
`labeling/anomaly_labels.py`'s `horizon_{h}sol`. Both tasks restrict
their BASE (current-sol) rows to genuine observations too. Cross-task
leakage is prevented structurally: distinct, non-overlapping target-column
prefixes (`TASK_TARGET_PREFIXES`), the shared feature builder never emitting
a `target_`-prefixed column, and a dedicated test proving a fabricated
(trailing-filled-only) future value is never used as a target
(`tests/test_datasets_no_leakage.py`).

## 6. Chronological split (`splitting/chronological_split.py`)

Leak-audited: asserts no storm episode spans a split boundary, provides
train-only standardisation-statistic fitting. **2 genuine boundary-crossing
episodes were found in the real run** and handled by excluding their member
cell-sols from both tasks rather than crashing or silently accepting the leak
— see `find_episodes_crossing_split_boundary`. Real split of the full
labeled/featured dataset: 81,355 train / 50,621 test / 10,697 unassigned
(`CHAPTER_4_PARAMETER_VALUES.md` Section 8).

## 7. Provenance (`modeling/provenance.py`) — every model, every kind, uniformly

`TrainingRecord` (written as `results/task_<name>/<model>_h<horizon>_provenance.json`
for EVERY baseline, classical, and sequence model) captures: `exact_parameters`
(`model.get_params()` for classical/baseline models; hyperparameters +
window length for sequence models), `random_seed`, `software_versions`
(python/numpy/pandas/sklearn/scipy/torch + CUDA availability, captured at
call time via `platform`/`sys`/module `__version__`), `training_history`
(GradientBoosting's `train_score_` per estimator; LogisticRegression's
`n_iter_`; LSTM/GRU's per-epoch train/val loss; an honest empty list with
`convergence_status="n/a (non-iterative model)"` for RandomForest/KNN/every
baseline — never fabricated), `checkpoint_path`, `feature_order` (empty list
for baselines, which don't consume the tabular feature manifest), `n_train`/
`n_val`, `convergence_status`, `runtime_seconds` (wall-clock, via a `Timer`
context manager), and `hardware_info` (processor/machine/cpu_count/GPU
availability).

## 8. Two real bugs caught by running the full pipeline end-to-end (not by unit tests alone)

Both were caught only because the pipeline was actually run against real data after
the unit tests passed — a reminder that synthetic-fixture tests prove logic
correctness but not necessarily real-data behaviour. Both fixes remain in place and
apply equally to the current real-only pipeline:

1. **`build_sequence_dataset` originally dropped a window if ANY feature
   value was NaN anywhere in it.** Given how often the sparse spatial-
   neighbour-stat features are NaN, this could silently produce zero usable
   sequence windows even when the feasibility report showed a window length as
   genuinely feasible. Fixed by only dropping a window for a genuine sol GAP
   (never bridgeable) and deferring NaN handling to imputation, exactly like the
   tabular path.
2. **That imputation, once added via `sklearn.impute.SimpleImputer`, then
   crashed with a reshape `ValueError`.** `SimpleImputer` silently DROPS a
   column that is entirely NaN across all of train (logged only as a
   `UserWarning`, "Skipping features..." — confirmed still occurring in the real
   run, e.g. for `feat_dust_optical_depth_z_neighbour_mean`/`_max` in some
   task/horizon splits, see `planning/FEATURE_DICTIONARY.md`) rather than
   filling it — which changes the output's column count and breaks the
   `(n_samples, window, n_features)` reshape used for sequence data. Fixed with a
   manual, shape-preserving median-with-zero-fallback imputer
   (`_impute_sequence_train_only` in `modeling/train.py`) that never drops
   a column.

Both fixes are covered by regression tests
(`tests/test_sequence_models.py::test_build_sequence_dataset_keeps_windows_with_nan_feature_values`,
and `tests/test_train.py::test_run_task_sequence_models_feasible_window`).

## 9. Real results (completed run, `outputs/real/run_20260805T062557386507Z_seed42`)

Full per-model, per-horizon metrics with 90% bootstrap CIs:
`CHAPTER_4_PARAMETER_VALUES.md` Section 8, and
`outputs/real/<run_id>/reports/final_report.md` /
`outputs/real/<run_id>/metrics/<task_name>/*.json` directly. Summary:

**`dust_regression`**: real but modest skill. `gradient_boosting_regressor` is the
strongest model, with the only positive R² in the whole task (0.122 at h=5 sol).
At shorter horizons no model clears R²=0 by much; MAE improves monotonically with
horizon for every model (the trend the old simulated-data doc predicted for a
smoothly-varying seasonal signal does hold qualitatively in real data, though the
magnitude of skill is smaller).

**`storm_detection`**: real, growing skill with horizon. `random_forest_classifier`
and GRU sequence models both clearly beat the `persistence`/`climatology` baselines
at the 3- and 5-sol horizons (best: TSS 0.244, ROC-AUC 0.769 at h=5 sol,
`random_forest_classifier`). At h=1 sol, `persistence` is still competitive with the
classical/sequence models — consistent with a genuinely hard short-horizon
forecasting problem, not a modeling defect.

Both results are reported as real, CI-supported, genuinely modest findings — neither
inflated nor apologized for.

## 10. Test coverage

`tests/test_features.py`, `tests/test_chronological_split.py`,
`tests/test_metrics.py`, `tests/test_datasets_no_leakage.py` (now parametrized over
2 tasks, not 3) — see earlier sections.

`tests/test_baselines.py` — persistence/seasonal-mean/climatology
correctness (including the unseen-season global-mean fallback and both remaining
climatology modes), the factory, and a config-driven-not-hardcoded proof.

`tests/test_provenance.py` — software/hardware info keys, the `Timer`
context manager, and a full `TrainingRecord` JSON roundtrip.

`tests/test_sequence_feasibility.py` — hand-computed consecutive-run
lengths (including the mars_year-boundary case), percentile/max summary
statistics, and per-window feasibility verdicts.

`tests/test_sequence_models.py` — windowed-dataset construction (full
window available, anchors dropped for insufficient history, a gap never
bridged, NaN feature values RETAINED rather than dropped — see §8), LSTM/GRU
training producing a real loss history and a valid convergence status,
early stopping actually triggering under tight patience, and seed-determinism
of the trained model's predictions.

`tests/test_registry.py` — task-namespace enforcement across all five
classical model families, including that KNN is registered for
classification only.

`tests/test_train.py` — end-to-end runs through the full hierarchy
(baselines + classical models create their own checkpoints/provenance;
a dense synthetic dataset exercises the sequence-model branch end-to-end,
including one genuinely feasible and one genuinely infeasible window
length in the same run), per-task seed reaching the fitted model, provenance
record field completeness, and graceful `"insufficient_data"` handling.

Full suite (all of `src/martian`): 316 passed as of the last real-data run
(2026-08-05).
