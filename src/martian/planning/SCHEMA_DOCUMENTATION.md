# Schema and Validation Documentation

**Status: real-only, two provider interfaces, both with a real, working, actually-run
implementation.** `EventProvider`/`DocumentedDustEventProvider` (ground-truth event
injection / independent-catalogue validation) no longer exist — they were simulation-only
concepts by definition, removed entirely in the real-only pivot (see
`CHAPTER_4_PARAMETER_VALUES.md` caveat 0.5 and Section 7). This document covers the
provider architecture as it exists now: `EMIRSObservationProvider`,
`MCDClimatologyProvider`, their real implementations, the provenance contract, the
(now largely unused, but kept as tested reusable utilities) generic schema-validation
helpers, and the immutable manifest writer.

## 1. Provider interfaces (`data_sources/base.py`)

| Interface | Concrete implementation | Purpose |
|---|---|---|
| `EMIRSObservationProvider` | `RealEMIRSObservationProvider` (`data_sources/real_emirs_provider.py`) | Physical observation records + provenance. Real, downloaded FITS files (7,445 files), real column-name mapping onto the pipeline's canonical schema (`configs/real_emirs_schema.yaml`'s `pipeline_column_mapping`) |
| `MCDClimatologyProvider` | `RealMCDClimatologyProvider` (`data_sources/real_mcd_provider.py`) | Climatological (mu, sigma) reference for anomaly z-scores (Eq. 4.1). Built from real, already-downloaded Montabone et al. gridded dust-optical-depth data (MY34-36); dust-only, no real temperature climatology source exists |

Every concrete provider's `.load()`/`.query()` ends by calling
`self._enforce_provenance_contract(df, caller=...)`, which raises
`ProvenanceContractError` if any of the 9 required provenance columns is absent — a
structural guarantee, not a convention that could be silently skipped.

## 2. The provenance contract

Every loaded/queried record carries these columns (see
`data_sources/provenance.REQUIRED_PROVENANCE_COLUMNS`):

| Column | Source | Notes |
|---|---|---|
| `data_source` | `ProviderIdentity.data_source` | `"REAL"` for both concrete providers |
| `product_id` | `ProviderIdentity.product_id` | e.g. `REAL-EMIRS-LEVEL-3-ATMOSPHERIC-AND-EMISSIVITY-PRODUCTS` |
| `processing_level` | `ProviderIdentity.processing_level` | `"L3"` |
| `version` | `ProviderIdentity.version` | Real EMIRS: no single fixed version — real downloaded files span ~30 distinct format-version strings (the SDC serves each file's own latest, `latest=true`); Real MCD: read from the loaded netCDF's own embedded attrs |
| `access_date` | `ProviderIdentity.access_date` (property) | ISO date, computed at call time |
| `original_filename` | Per-row, from the real source FITS file (`source_fits_file_l3atm`) or netCDF filename | Real filenames, e.g. `emm_emr_l3atm_20210220t112122_0008_f_v02-20.fits` |
| `checksum` | `compute_row_checksum()` — SHA-256 truncated to 16 hex chars | Computed over the row's own physical-variable content |
| `retrieval_timestamp` | Wall-clock UTC time at load/query time | Distinct from `utc_timestamp`, the real observation epoch |
| `quality_flag` | Per-row physical variable | Real EMIRS has no native per-row quality flag (unlike an earlier speculative schema) — `RealEMIRSObservationProvider` sets `quality_flag=0` for every row; `chi2temp`/`chi2a` (real goodness-of-fit fields) are available for the real QC rules downstream instead |

**Uncertainty** is a per-variable `"<variable>_uncertainty"` companion column, not one
fixed column every provider must emit — present for `dust_optical_depth`,
`water_ice_optical_depth`, `surface_temperature_k` (all real, mapped from real
`dustuncert`/`iceuncert`/`kinetic_temp_uncertainty`), and explicitly absent for
`latitude`, `longitude`, timestamps, `quality_flag`, `surface_pressure_pa`, and the
collapsed atmospheric-temperature mean — since no real per-measurement uncertainty
exists for those fields (see `CHAPTER_4_PARAMETER_VALUES.md` Section 1's vertical-grid
reconciliation note for why the 19-level profile isn't exploded into per-level columns).

## 3. Generic schema-validation utilities (`data_sources/schema.py`) — kept, not currently wired in

`FieldSpec`, `validate_schema()`, and `SchemaValidationError` remain as real, tested,
reusable generic utilities (a caller supplies its own field specs and a dataframe;
`validate_schema` checks missing columns / invalid units / duplicate observations /
inconsistent versions in one pass, reporting every violation found, not just the
first — `tests/test_schema_validation.py`). **They are not currently called by either
real provider** — `RealEMIRSObservationProvider`/`RealMCDClimatologyProvider` do their
own, more specific validation (real column presence via `apply_pipeline_column_mapping`'s
fail-fast `KeyError`, the provenance contract above) rather than going through this
generic mechanism. `load_emirs_schema()` (the old simulated-schema-specific loader) was
deleted along with `configs/emirs_variables.yaml` in the real-only pivot.

## 4. Immutable manifests (`data_sources/manifest.py`)

`write_immutable_manifest(manifest_dir, run_id, payload)`:
- refuses to overwrite an existing `run_id`'s manifest (`ManifestAlreadyExistsError`);
- always requires a fresh `run_id` (`new_run_id()` uses a UTC timestamp with
  microsecond resolution) for a new write;
- sets the written file read-only (`os.chmod(path, stat.S_IREAD)`) as a
  filesystem-level guard.

**Stated limitation:** this is a best-effort guard against accidental in-place
edits by the same tooling, not a cryptographic tamper-evidence mechanism — a
sufficiently privileged process could still delete and recreate a file under the
same name. This is documented in the module docstring rather than overclaimed.

Note this is distinct from `mars/run_manifest.py::RunManifest`, the per-run
step-timing/status/leakage-check manifest written by `mars.run_all` — this
module's immutable-manifest mechanism is a lower-level, general-purpose utility.

## 5. `data_source` column on processed datasets

`provenance.stamp_processed_dataset(df, data_source_label)` returns a *copy* of
`df` with a `data_source` column set to the given label, leaving the input frame
unmodified. This is distinct from the raw per-record provenance columns attached at load
time (§2): a processed row (post-gridding, post-feature-engineering) may aggregate
several raw records, so its `data_source` label describes the *processing stage's*
provenance.

## 6. Visible data-source labels on figures/tables/reports

Three helpers in `data_sources/provenance.py`, each tested:

- `stamp_figure(fig, data_source_label)` — writes a visible caption onto a
  matplotlib `Figure` before it is saved.
- `stamp_table_markdown(df, data_source_label, title=None)` — renders a
  dataframe as Markdown with a mandatory `*Data source: ...*` caption line
  (hand-rolled table renderer, no `tabulate` dependency).
- `stamp_report(body_text, data_source_label)` — wraps a text report with a
  `[DATA SOURCE: ...]` banner at both the top and the bottom.

**A real bug lived here until 2026-08-05**: `mars/final_report.py`,
`evaluation/evaluate_task.py`, and (unused, dead code) `modeling/train.py` each
carried their own separately-maintained `DATA_SOURCE_LABEL = "SIMULATED (...)"`
module constant that was never updated when the pipeline went real-only — every
figure and the final-report banner from the first real run said "SIMULATED"
despite the run being genuinely real, directly contradicting the correct
`Data source: real` line in the same report. Fixed by deriving the report's label
from the run manifest's own `data_source` field, and threading a
`data_source_label` parameter from `mars/run_all.py`'s own already-correct
`DATA_SOURCE_LABEL` down through `run_task()` → `evaluate_and_report()` → every
figure's `FigureProvenance` — see `CHAPTER_4_PARAMETER_VALUES.md` caveat 6.
**Lesson for any future provider/report addition**: derive a data-source label
from the run's actual manifest/config at call time, never hardcode a
module-level constant that has to be remembered and updated by hand.

## 7. Storm taxonomy consistency

`configs/storm_taxonomy.yaml` is the single definition of `local`/`regional`/
`planet_encircling` (area-in-cells and duration-in-sols thresholds), consumed by
`labeling/storm_taxonomy.py::classify_episode_catalogue`, run as pipeline Step 8. No
other module redefines these thresholds. Real classification breakdown from the
completed run: `CHAPTER_4_PARAMETER_VALUES.md` Section 1.
