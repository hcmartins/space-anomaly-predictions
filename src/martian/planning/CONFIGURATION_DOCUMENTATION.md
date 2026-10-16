<!-- AUTO-MAINTAINED (by hand). Add a row whenever a new configs/*.yaml file
     is added; keep the one-line purpose current if a file's role changes. -->

# Configuration Documentation

Every scientific parameter, threshold, hyperparameter, and file path in
this pipeline lives in a `configs/*.yaml` file — never hardcoded inside a
`.py` module (the one standing rule that has held across every phase of
this build). This document is the index: what each config file governs,
who reads it, and where its detailed field-by-field explanation lives (each
config file also carries its own header comment explaining every field in
place — this index is for finding the right file quickly, not a
duplicate of that).

**Real-only pipeline, one run config**: there is no `configs/mars/{simulated,real}.yaml`
pair any more — `configs/run.yaml` is the only top-level run config, and there is no
`data_source` selection logic left in the code to choose between (see
`CHAPTER_4_PARAMETER_VALUES.md` caveat 0.5 for the pivot history).

| Config file | Governs | Read by | Detailed docs |
|---|---|---|---|
| `configs/global.yaml` | Default seed, output/figures roots, provenance tag | reference | `planning/SEED_STRATEGY.md` |
| `configs/run.yaml` | The only top-level run config: seed, `mars_year_range`, `output_root`, `configs_dir`, task list | `mars/run_all.py` | `planning/PIPELINE_DOCUMENTATION.md` |
| `configs/grid.yaml` | 5°×5° primary grid (36×72×2592 cells), 2.5°/10° sensitivity grids, longitude wraparound, numerical justification | `preprocessing/grid.py` | `planning/PREPROCESSING_DOCUMENTATION.md` §1 |
| `configs/real_emirs_ingestion.yaml` | Real EMIRS FITS directory location, product list, file extensions | `data_sources/real_emirs_fits.py` | `planning/SCHEMA_DOCUMENTATION.md` |
| `configs/real_emirs_schema.yaml` | Real EMIRS L3 (l3atm+l3emiss) field names/units/products/verification status, the 19-level vertical grid, and `pipeline_column_mapping` (real field name → pipeline canonical name) | `data_sources/real_emirs_provider.py`, `data_sources/real_emirs_fits.py` | `planning/SCHEMA_DOCUMENTATION.md` |
| `configs/mcd.yaml` | Real MCD provider config: precomputed-climatology netCDF path, variables covered, scenario/interpolation (CONFIRMED real values), Montabone-build-specific settings | `data_sources/real_mcd_provider.py` | `planning/SCHEMA_DOCUMENTATION.md` |
| `configs/storm_taxonomy.yaml` | Local/regional/planet-encircling area+duration thresholds (descriptive only, never an ML target) | `labeling/storm_taxonomy.py` | `planning/LABELING_DOCUMENTATION.md` §3 |
| `configs/quality_control.yaml` | 8-rule first-failing-rule QC hierarchy, priority order, thresholds. Rule 7 (viewing-geometry) is disabled: no real field exists for it | `preprocessing/quality_control.py` | `planning/PREPROCESSING_DOCUMENTATION.md` §2 |
| `configs/aggregation.yaml` | Cell-sol aggregation: uncertainty-weighted vs. simple-mean variable lists, min-obs/min-coverage primary+sensitivity | `preprocessing/aggregation.py` | `planning/PREPROCESSING_DOCUMENTATION.md` §3 |
| `configs/missing_data.yaml` | Trailing-fill max gap (primary 2 sols, sensitivity 0/1/2/3), fill-eligible variable list, reserved label patterns | `preprocessing/missing_data.py` | `planning/PREPROCESSING_DOCUMENTATION.md` §6 |
| `configs/causal_filter.yaml` | Causal Savitzky-Golay: variable, window=7, polyorder=2, trailing coefficient position | `preprocessing/causal_filters.py` | `planning/PREPROCESSING_DOCUMENTATION.md` §7 |
| `configs/anomaly_definition.yaml` | z-score epsilon, dust exceedance sidedness, tau primary+sensitivity, persistence/area gate (primary k=2/m=3 as of 2026-08-05), horizons. Dust-only — no temperature exceedance term | `labeling/anomaly_labels.py` | `planning/LABELING_DOCUMENTATION.md` §1-2 |
| `configs/real_documented_events.yaml` | Real, independently-sourced MY36 documented dust-storm catalogue (Guha et al. 2024, EMM/EXI): file path, citation, licence, Mars-Year coverage | `data_sources/real_documented_events.py` | `planning/LABELING_DOCUMENTATION.md` §5 |
| `configs/catalogue_validation.yaml` | Independent-catalogue matching rule (temporal tolerance, spatial region radius) for Table 4.6 — reinstated 2026-08-05 against real data on both sides | `labeling/catalogue_validation.py` | `planning/LABELING_DOCUMENTATION.md` §5 |
| `configs/splits.yaml` | Chronological train/val/test sol boundaries (real MY36/MY37) | `splitting/chronological_split.py` | `planning/MODELING_DOCUMENTATION.md` §5 |
| `configs/features.yaml` | Feature-family toggles, lag depths, trailing-window length. Dust-only (`dust_optical_depth`/`_z`) | `preprocessing/features.py` | `planning/MODELING_DOCUMENTATION.md` §2, `planning/FEATURE_DICTIONARY.md` |
| `configs/evaluation.yaml` | Bootstrap CI settings (block-by-cell, 200 replicates, 90%), reliability-diagram bins, failure-analysis bin edges, primary baseline per task | `evaluation/*.py` | `planning/EVALUATION_DOCUMENTATION.md` |
| `configs/task_dust_regression.yaml` | Task 1: target definition, horizons, seed, full baseline/classical/sequence model hierarchy, output paths | `modeling/train.py` | `planning/MODELING_DOCUMENTATION.md` §1 |
| `configs/task_storm_detection.yaml` | Task 2: same, plus the FAR-rate-vs-ratio definition note | `modeling/train.py` | `planning/MODELING_DOCUMENTATION.md` §1 |

## Removed configs (pivot history, kept for context — files no longer exist)

`configs/emirs_variables.yaml` (the old simulated EMIRS schema), `configs/mcd.yaml`'s
simulated-provider fields, `configs/event_provider.yaml`, `configs/documented_events.yaml`,
`configs/task_temperature_regression.yaml`, `configs/mars/simulated.yaml`,
`configs/mars/real.yaml`, and `configs/emirs_simulation_calibration.yaml` were all
deleted when the pipeline pivoted to real-data-only (see
`CHAPTER_4_PARAMETER_VALUES.md` caveat 0.5). If you find a reference to any of these
elsewhere in `planning/`, it describes history, not current state.
**`configs/catalogue_validation.yaml` was deleted in that same pivot but has since
been reinstated** (2026-08-05, see `configs/real_documented_events.yaml` above) —
it is back in the main table, not removed.

## Conventions used consistently across every config file

- **Chapter-grounding tags**: every scientific parameter is tagged either
  as grounded in the supplied Chapter 4 text (with a citation) or
  `RESEARCHER VERIFICATION REQUIRED` (an engineering addition or
  assumption, never silently presented as verified). Real EMIRS schema
  fields additionally use `CONFIRMED_BY_REAL_DATA_PRODUCT_GUIDE` /
  `CONFIRMED_AGAINST_REAL_DOWNLOADED_DATA` where a field's real name/meaning
  has been independently verified against actual downloaded data (stronger
  than a documentation citation alone — see `configs/real_emirs_schema.yaml`'s
  `qls`/`qlt` entries for why this distinction matters).
- **Primary vs. sensitivity**: every threshold that has a sensitivity sweep
  requirement states its primary value AND its full sensitivity list in the
  same file, never split across files.
- **No config imports another config's raw values by re-declaring them** —
  e.g. `configs/storm_taxonomy.yaml` is the ONLY place local/regional/
  planet-encircling thresholds are defined; anything else that needs them
  (the taxonomy classifier) loads this file, never hardcodes a copy.
