<!-- AUTO-MAINTAINED (by hand). Update when mars/*.py or the 15-step sequence
     in mars/run_all.py::run_pipeline changes. -->

# Pipeline Documentation — `mars.run_all`, the one reproducible entry point

**Status: implemented, tested, and actually run against real data.** One command runs
the entire Chapter 4 pipeline end to end:

```
python -m mars.run_all --config configs/run.yaml
```

Every step below calls an already-tested library function from
`data_sources/`, `preprocessing/`, `labeling/`, `splitting/`, `modeling/`,
or `evaluation/` — `mars/` adds orchestration, run isolation, and
reproducibility bookkeeping, not new scientific logic. See
`planning/PREPROCESSING_DOCUMENTATION.md`, `planning/LABELING_DOCUMENTATION.md`,
`planning/MODELING_DOCUMENTATION.md`, and `planning/EVALUATION_DOCUMENTATION.md`
for what each underlying step actually does and why; this document maps
those steps onto the required 15-step run and the required output
structure.

This pipeline is real-data-only: there is no simulated fallback anywhere in
`src/martian` (see `CHAPTER_4_PARAMETER_VALUES.md` caveats 0.5/0.6 for the
full history of that pivot). `data_source: "real"` (`mars/run_all.py`'s
`DATA_SOURCE` constant) is the only path.

## The 15 steps, and what implements each one

| # | Step | Implementation |
|---|---|---|
| 1 | Validate inputs | `mars/validate_inputs.py::validate_inputs` — collects every problem found, not just the first |
| 2 | Write a run manifest | `mars/run_manifest.py::RunManifest` — written at start, rewritten after every step |
| 3 | QC and gridding | `RealEMIRSObservationProvider.load()`, `preprocessing.quality_control.run_quality_control`, `preprocessing.grid.assign_cells` |
| 4 | Aggregate cell-sols | `preprocessing.aggregation.aggregate_by_cell_sol` |
| 5 | Treat missing data causally | `preprocessing.missing_data.apply_trailing_fill` |
| 6 | Apply causal filtering | `preprocessing.causal_filters.apply_causal_savgol` — output persisted to `data/filled_with_causal_savgol.csv`; not currently threaded into the ML feature set (a documented scope boundary, see below) |
| 7 | Calculate climatological deviations | `labeling.anomaly_labels.compute_z_scores` / `compute_exceedance`, against `RealMCDClimatologyProvider` (on the trailing-filled series, for feature density) |
| 8 | Build labels and episodes | `labeling.anomaly_labels.run_full_labeling_pipeline`, `labeling.storm_taxonomy.classify_episode_catalogue`; boundary-crossing episodes found and excluded here (see `planning/LABELING_DOCUMENTATION.md`) |
| 9 | Validate labels against the real documented dust-storm catalogue (Table 4.6, Section 4.4.1) | `labeling.catalogue_validation.run_catalogue_validation`, against `data_sources.real_documented_events.load_real_documented_events` — **optional**: recorded as `{"status": "NOT PERFORMED"}` in `pipeline_summary`, not a run failure, if the real catalogue file isn't present (see `planning/LABELING_DOCUMENTATION.md` §7) |
| 10 | Build causal features | `preprocessing.features.build_features` |
| 11 | Construct chronological splits | `splitting.chronological_split.assign_split` + `write_split_manifest`; per-task datasets (`modeling.datasets.build_dust_regression_dataset`/`build_storm_detection_dataset`) are also built here, needed before leakage checks |
| 12 | Run leakage tests | `mars/leakage_checks.py::run_leakage_checks` — RUNTIME checks against this run's actual per-task datasets (episode-boundary + cross-task target-column leakage), distinct from (and complementary to) the static pytest leakage suite |
| 13 | Train baselines/models, evaluate every task/horizon, compute uncertainty intervals, generate figures/tables | `modeling.train.run_task` (baselines → classical → sequence, per task, per horizon), which calls `evaluation.evaluate_task.evaluate_and_report` immediately after each model is fit |
| 14 | Generate a final report | `mars/final_report.py::generate_final_report` — `reports/final_report.md`, every number read directly from the run's own artifacts, `[DATA SOURCE: ...]` banner derived from the manifest's own `data_source` field (not a separate constant — see below) |
| 15 | Write checksums for every output | `mars/checksums.py::write_checksums_manifest` — SHA-256 of every file under the run directory, written to `reports/checksums.json` |

**Step 9 was reinstated 2026-08-05** after a real, independently-sourced MY36
dust-storm catalogue was located (Guha et al. 2024's EMM/EXI database) —
previously Section 4.4.1 was descoped entirely because no real catalogue was
in hand. It is deliberately optional (wrapped in a `try/except FileNotFoundError`)
rather than a hard pipeline dependency, since it validates labels rather than
producing them — the rest of the pipeline runs identically whether or not the
catalogue file is present.

**Step 13 fuses training, evaluation, uncertainty, and figure generation into one
per-model pass**, not four sequential sweeps over all models:
`evaluate_and_report` runs immediately after each model is fit, reusing the
already-fitted model/imputer directly rather than reloading a checkpoint
(which would require separately persisting the fitted imputer). This remains
the dominant-cost step of the whole pipeline (see Performance note below).

## Output structure

```
outputs/real/<run_id>/
  manifests/   run_manifest.json, QC log
  data/        aggregated cell-sols, trailing-filled series, causal-SG-filtered series
  labels/      labeled cell-sols, episode catalogue, catalogue-validation matched
               pairs/documented-events/episode-status (if Step 9 ran)
  features/    per-task dataset.csv + feature_manifest.json
  splits/      split_manifest.json
  models/<task_name>/    per-task checkpoint subdirectories (baseline .joblib, classical .joblib, sequence .pt)
  metrics/<task_name>/   per-task metrics.json, predictions.csv, sequence feasibility report, per-model provenance JSON, per-model evaluation JSON
  figures/<task_name>/   per-task figure subdirectories (PNG + source CSV + provenance JSON per figure)
  tables/                QC summary, Table 4.6 (independent-catalogue validation, if Step 9 ran) (run-level)
  tables/<task_name>/    failure-analysis breakdowns, baseline-comparison tables (per task)
  reports/     final_report.md, checksums.json
```

`<run_id>` is `run_<UTC-microsecond-timestamp>_seed<seed>` — microsecond-resolution
timestamps mean two runs never collide, and a run never overwrites a
previous run's outputs (`mars/run_paths.py::generate_run_id`). Task-scoped
`<task_name>/` subdirectories under `models/`/`metrics/`/`figures/`/`tables/`
exist specifically because `dust_regression` and `storm_detection` share
baseline model names (`climatology`, `persistence`, `seasonal_mean`) — without
per-task scoping, one task's files would silently overwrite the other's.

## Scope boundary: causal-filtered dust is not yet a model feature

Step 6 (causal Savitzky-Golay filtering of `dust_optical_depth`) is run and
its output persisted as its own pipeline artifact, exactly as this task's
step list requires. It is NOT currently included as an input column to
`preprocessing.features.build_features` — that module's "current state"
family already includes the raw and z-scored `dust_optical_depth`, and
adding a third (SG-smoothed) variant was judged out of scope (a genuine
feature-engineering decision, not an oversight — noted here explicitly
rather than silently folded in without being asked).

## Performance note (real numbers, from an actual full-scale real run)

A full run against the real EMIRS dataset (7,445 downloaded FITS files,
`configs/run.yaml`) took **~1h23m end to end** in an earlier run before Step 9
(catalogue validation) existed (`run_20260805T062557386507Z_seed42`). Per-step
timings from that run: Step 3 (QC and gridding) ~17.5 min — dominated by
parsing thousands of individual real FITS files via `astropy.io.fits`, not by
any downstream computation; the labeling/feature/split/leakage steps combined
~7 min; **the train/evaluate/uncertainty/figures step ~58 min**, the dominant
cost, scaling with the number of tasks × models × horizons trained in that
single pass (2 tasks, ~9 baseline/classical/sequence models each, 3 horizons
each); the report/checksums steps a few seconds. Step 9 (catalogue validation,
added afterward) is cheap by comparison — a few hundred documented events
against typically a few thousand qualifying episodes, well under a second.

**A real, since-fixed bug that could make Step 3 fail entirely**: real l3atm/l3emiss
files contain genuine fill/sentinel rows (unparseable `utc`, `latitude=longitude=0`
— confirmed up to ~37% of rows in some files). Left in, every such row shares an
identical merge key, and pandas' `merge()` treats `NaT` as equal to `NaT` for join
matching, so `RealEMIRSObservationProvider`'s l3atm/l3emiss inner merge produced a
many-to-many cartesian product for that one key — a 120 GiB allocation failure at
full scale. Fixed by dropping unparseable-`utc` rows at load time
(`data_sources/real_emirs_fits.py::load_real_emirs_directory`) — see
`CHAPTER_4_PARAMETER_VALUES.md` caveat 6 for the full writeup, including a second,
related bug (mixed `utc` timestamp formats silently producing missing values).

If a full run raises `bad allocation` (`std::bad_alloc`) instead, that is a
genuine host memory-headroom issue during the train/evaluate step (not the
Step-3 bug above) — close other heavy applications before launching a full
run if free memory is tight.
