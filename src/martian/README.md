# Martian Atmospheric Forecasting — Chapter 4 Pipeline

Corrected implementation of Chapter 4 ("Martian Atmospheric Forecasting")
for Herbert Ayoola Caxton-Martins' thesis, University of Bradford. This
directory (`src/martian/`) is a self-contained, importable `martian`
package covering the full pipeline: provider architecture → preprocessing
→ labeling → chronological splitting → modeling → evaluation →
orchestration.

**Status: real data only.** There is no simulated-data fallback anywhere in
this codebase — the examiner requires real data, so the pipeline was
deliberately stripped down to two real providers and two forecasting tasks
rather than maintaining a synthetic parallel path:

- **EMIRS** (`data_sources/real_emirs_provider.py`): real, tested code,
  real downloaded FITS files (7,445 l3atm/l3emiss files, Feb 2021-Jan 2025,
  via `download_emirs_sdc.py`, in `data_cache/emirs/raw`), and a real
  column-name mapping onto the pipeline's canonical schema
  (`configs/real_emirs_schema.yaml`'s `pipeline_column_mapping`) —
  including `solar_longitude_ls`/`local_true_solar_time`, which turned out
  to be real, native per-footprint l3atm columns (`qls`/`qlt`), not
  something this pipeline needs to compute. Verified working end-to-end
  against real data at small scale.
- **MCD dust climatology** (`data_sources/real_mcd_provider.py`): real and
  working. Built from real, already-downloaded Montabone et al. gridded
  column-dust-optical-depth data for Mars Years 34-36
  (`build_mcd_climatology_from_montabone.py`,
  `data_cache/mcd/mcd_climatology_5deg.nc`). Covers `dust_optical_depth`
  only — no real climatology source exists for `surface_temperature_k`, so
  temperature-anomaly forecasting (the former Task 2) remains descoped
  rather than simulated.
- **Independent-catalogue label validation** (Chapter 4 Section 4.4.1,
  Table 4.6, `labeling/catalogue_validation.py`): reinstated 2026-08-05
  against a real source — Guha et al. (2024)'s EMM/EXI Mars Year 36
  dust-storm database (`data_sources/real_documented_events.py`,
  `data_cache/emm_exi_dust_storms/`, CC BY 4.0). The two well-known Mars
  dust-storm catalogues (MDAD, MDSSD) both end before real EMIRS coverage
  begins (MY36) and cannot validate this pipeline's real labels; this one
  covers exactly MY36, from the same mission. MY36-only — MY37 episodes/data
  have no documented-catalogue counterpart and are excluded from matching,
  never scored as misses. Runs as pipeline Step 9, optional (fails
  gracefully to "NOT PERFORMED" if the catalogue file isn't present, never
  fabricated).

**A 120 GiB allocation failure at Step 3 (QC/gridding) has been root-caused
and fixed.** Real l3atm/l3emiss files contain genuine fill/sentinel rows
(empty `utc` string, never-geolocated footprints at `latitude=longitude=0`)
— confirmed up to ~37% of rows in some files. Left in, every such row
collapses to the identical `(NaT, 0, 0)` merge key, and pandas' `merge()`
treats `NaT` as equal to `NaT` for join matching, so
`RealEMIRSObservationProvider`'s l3atm/l3emiss inner merge produced a
many-to-many cartesian product for that one key — reproduced directly
(2 NaT rows x 3 NaT rows -> 6 cartesian rows), and confirmed as the real
mechanism behind the 120 GiB failure. Fixed by dropping unparseable-`utc`
rows at load time (`data_sources/real_emirs_fits.py::load_real_emirs_directory`)
— a legitimate filter regardless (a row with no timestamp can never get a
`mars_year`/`sol` and is unusable downstream anyway), not a workaround.
Verified against a 100-real-file sample: load -> QC -> grid-assignment now
completes in seconds with no crash.

## Quick start

```bash
# 1. Create the environment (either one)
conda env create -f environment.yml && conda activate martian-ch4
# or:
pip install -r requirements.txt

# 2. Run the full pipeline (fails at Step 3 until real EMIRS FITS files exist)
python -m mars.run_all --config configs/run.yaml

# 3. Run the test suite
python -m pytest -q
```

All commands assume the current directory is `src/martian/` (this
directory) — `tests/conftest.py` and the `mars`/`preprocessing`/`labeling`/
etc. top-level imports both depend on it being on `sys.path`, which happens
automatically for both `pytest` and `python -m mars.run_all` when invoked
from here.

## The one reproducible entry point

```bash
python -m mars.run_all --config configs/run.yaml
```

Runs all 15 pipeline steps — input validation, run-manifest writing,
QC/gridding, cell-sol aggregation, causal missing-data treatment, causal
filtering, climatological deviations, label/episode construction,
independent-catalogue label validation against a real documented dust-storm
catalogue (Table 4.6 — optional, "NOT PERFORMED" if the catalogue file isn't
present, never fabricated), causal feature engineering, chronological
splitting, runtime leakage tests, model training (baselines → classical →
sequence) for both tasks (`dust_regression`, `storm_detection`), evaluation,
bootstrap uncertainty intervals, figures/tables, a final report, and a
checksum manifest for every output file — into a fresh, never-overwritten
run directory. Full step-by-step mapping: `planning/PIPELINE_DOCUMENTATION.md`.

### Output structure

```
outputs/real/<run_id>/
  manifests/   run manifest, QC log
  data/        aggregated / trailing-filled / causal-SG-filtered cell-sol series
  labels/      labeled cell-sols, episode catalogue
  features/    per-task dataset.csv + feature_manifest.json
  splits/      chronological split manifest
  models/<task_name>/    per-task model checkpoints
  metrics/<task_name>/   per-task metrics, predictions, per-model evaluation/provenance JSON
  figures/<task_name>/   per-task figures (PNG + source CSV + provenance JSON)
  tables/, tables/<task_name>/   QC summary, failure-analysis and baseline-comparison tables
  reports/     final_report.md, checksums.json (SHA-256 of every file in the run)
```

`<run_id>` is `run_<UTC-microsecond-timestamp>_seed<seed>` — every run gets
its own directory; nothing is ever overwritten by a later run.

## Documentation index

| Document | Covers |
|---|---|
| `CHAPTER_4_PARAMETER_VALUES.md` | Real, resolved values for every chapter [INSERT]/[CONFIRM] marker, and what remains blocked pending real EMIRS data |
| `planning/PIPELINE_DOCUMENTATION.md` | Step-by-step pipeline mapping (some detail predates the real-only simplification) |
| `planning/CONFIGURATION_DOCUMENTATION.md` | Every `configs/*.yaml` file: what it governs, who reads it |
| `planning/DATA_DICTIONARY.md` | Every persisted data/label column |
| `planning/FEATURE_DICTIONARY.md` | Every engineered `feat_*` feature column |
| `planning/MODEL_CARDS.md` | Every model in every task's baseline/classical/sequence hierarchy |
| `planning/SEED_STRATEGY.md` | Every seed in the pipeline, where it's set, and where it's recorded |
| `planning/PREPROCESSING_DOCUMENTATION.md` | Grid, QC, aggregation, missing-data, causal filtering |
| `planning/LABELING_DOCUMENTATION.md` | Anomaly/episode definitions, taxonomy |
| `planning/MODELING_DOCUMENTATION.md` | Task separation, datasets, model hierarchy, chronological splits |
| `planning/EVALUATION_DOCUMENTATION.md` | Metrics, confidence intervals, figures, failure analysis, baseline comparison |

## Tests

```bash
python -m pytest -q                       # full suite
python -m pytest tests/test_mars_*.py -q  # orchestration layer only
```

Test categories: unit tests per module (`tests/test_<module>.py`),
integration tests for the `mars.run_all` orchestration wiring
(`tests/test_mars_integration.py` — see its module docstring for why it
doesn't re-run the full multi-minute real pipeline on every invocation),
and static leakage tests (`tests/test_datasets_no_leakage.py`,
`tests/test_chronological_split.py`) complementing the pipeline's own
runtime leakage checks (`mars/leakage_checks.py`, Step 11).

## Known limitations

- ~~`mars.run_all` fails at Step 3 (QC/gridding) on the full real dataset
  with a 120 GiB array-allocation error~~ — **root-caused and fixed**, see
  the top of this README (real fill/sentinel rows with `utc=NaT` caused a
  many-to-many cartesian merge). Verified on a 100-file real sample; not
  yet re-verified against the full 7,445-file dataset (a full run takes
  ~30+ minutes just to reach this step).
- ~~Real EMIRS's column names have not been mapped onto the pipeline's
  canonical names~~ — **done** (`configs/real_emirs_schema.yaml`'s
  `pipeline_column_mapping`, applied by `RealEMIRSObservationProvider.load()`).
  Includes a real-column fix worth knowing about: `utc` strings mix a
  `.mmm` fractional-second suffix with plain `HH:MM:SS` within the same
  file — `load_real_emirs_directory` now parses with `format="mixed"`;
  without it, pandas silently turned up to ~80% of rows in some real files
  into `NaT` (confirmed against real downloaded data, 2026-08-04).
- **No real temperature climatology.** Temperature-anomaly forecasting
  (the former Task 2) remains descoped from this implementation, not
  simulated — see `CHAPTER_4_PARAMETER_VALUES.md`. The independent-catalogue
  validation step (Section 4.4.1) is no longer in this state — see the
  Status section above.
- A full-scale run's peak memory during the fused train/evaluate/figures
  step scales with (tasks × models × horizons trained in one pass); see
  `planning/PIPELINE_DOCUMENTATION.md`'s Performance note if you hit a
  `bad allocation` on a memory-constrained machine.
- Several schema fields are marked `RESEARCHER VERIFICATION REQUIRED` in
  `planning/DATA_DICTIONARY.md` where an engineering assumption fills a gap
  the supplied Chapter 4 text did not specify — never silently presented as
  verified fact.
