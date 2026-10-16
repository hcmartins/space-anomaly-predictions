<!-- AUTO-MAINTAINED (by hand). Add a row whenever a new persisted data
     column is introduced anywhere in the pipeline. -->

# Data Dictionary

Every column that appears in a persisted cell-sol dataframe, grouped by the
pipeline stage that introduces it. "Chapter-grounded" columns cite the
supplied Chapter 4 text; everything else is flagged `RESEARCHER
VERIFICATION REQUIRED` at its point of introduction. This dictionary
describes the AGGREGATED cell-sol table (`data/aggregated_cell_sol.csv` in
a `mars.run_all` run) and the tables derived from it — not the raw
per-retrieval table, which `planning/SCHEMA_DOCUMENTATION.md` covers at the
provider level and `configs/real_emirs_schema.yaml` documents in full
(real field names before `pipeline_column_mapping` renames them).

**Real-only, dust-only.** `surface_temperature_k`, `temperature_anomaly`, and the
14-level `atm_temp_k_level_*` columns described in earlier versions of this
document no longer exist anywhere in the pipeline output — see
`CHAPTER_4_PARAMETER_VALUES.md` caveat 0.5/Section 1 for why.

## Identity / provenance (every row)

| Column | Type | Meaning |
|---|---|---|
| `cell_id` | int | `lat_band * n_lon_bands + lon_band`, deterministic per grid resolution (`preprocessing/grid.py`) |
| `mars_year` | int | Real Mars Year (36 or 37 in the primary run), from a documented linear approximation on real `utc` (`data_sources/real_emirs_fits.py::utc_to_mars_year_sol`) |
| `sol` | int | Sol-of-year |
| `latitude`, `longitude` | float | Cell-sol simple-mean of contributing real retrievals' coordinates (degrees; longitude 0-360°E) |
| `data_source` | str | Visible provenance label, stamped on every processed table — `"REAL (Martian Chapter 4 mars.run_all pipeline)"` |

## Physical variables (aggregated, uncertainty-weighted where a companion uncertainty column exists)

| Column | Unit | Real source (`configs/real_emirs_schema.yaml`'s `pipeline_column_mapping`) |
|---|---|---|
| `dust_optical_depth` (+ `_uncertainty`) | dimensionless | `taudust`/`dustuncert` (real l3atm) |
| `water_ice_optical_depth` (+ `_uncertainty`) | dimensionless | `tauice`/`iceuncert` (real l3atm) |
| `surface_pressure_pa` | Pa (unit unconfirmed — RESEARCHER VERIFICATION REQUIRED) | `pres0` (real l3atm) |
| `solar_longitude_ls` | degrees (Ls) | `qls` (real l3atm — the EMM EMIRS Data Product Guide documents this as `l_sub_s`; every real downloaded file actually uses `qls`, verified empirically) |
| `local_true_solar_time` | hours | `qlt` (real l3atm — Guide documents `ltime`; real files use `qlt`, verified against l3emiss's own `ltime` column matching row-for-row) |

`surface_temperature_k` (from real `kinetic_temp`, l3emiss) and its `_uncertainty`
companion (`kinetic_temp_uncertainty`) ARE mapped and present on the raw per-retrieval
table for QC purposes (`preprocessing/quality_control.py`'s
`physically_impossible_temperature`/`excessive_retrieval_uncertainty` rules read
them), but are **not** in `configs/aggregation.yaml`'s variable lists — no real
temperature climatology exists, so temperature is not aggregated, not a feature, and
not a label input (see `CHAPTER_4_PARAMETER_VALUES.md` caveat 5). Likewise
`atmospheric_temperature_mean_k` (collapsed mean of the real 19-level `temp` array,
`data_sources/real_emirs_fits.py::collapse_temp_profile`) exists on the raw table but
is not aggregated — no real per-level uncertainty exists to weight it by (Section 1's
vertical-grid reconciliation note).

`emission_angle_deg`/`surface_emissivity` do not exist anywhere in the real pipeline —
no real EMIRS field corresponds to either (confirmed against real downloaded data);
QC rule 7 (viewing-geometry) is disabled and `surface_emissivity` was removed from
`configs/aggregation.yaml`'s variable lists rather than left to crash on a missing
column.

## Aggregation bookkeeping (`preprocessing/aggregation.py`)

| Column | Meaning |
|---|---|
| `observation_count` | Number of real retrievals contributing to this cell-sol |
| `pressure_level_coverage_frac` | Always `NaN` for real data (no `atm_temp_k_level_*` columns exist to compute coverage over) — `compute_low_support_flag` treats NaN as "not applicable," not a failure |
| `uncertainty_summary` | Mean of the combined (inverse-variance) uncertainties across weighted variables |
| `low_support_flag` | True if `observation_count` falls below the configured minimum (`pressure_level_coverage_frac`'s NaN never contributes to this for real data) — never silently dropped, flagged instead |
| `interpolation_flag` | Always `False` at this stage (see `preprocessing/missing_data.py` below for where this becomes real) |

## Missing-data / causal-fill bookkeeping (`preprocessing/missing_data.py`, present on the trailing-filled table only)

| Column | Meaning |
|---|---|
| `is_observed` | `True` for a genuine retrieval-derived row, `False` for a trailing-filled (carried-forward) row |
| `gap_length_sols` | Sols since the last real observation (0 for a genuine observation) |
| `interpolation_flag` | Now meaningful: `True` only for a filled row |

## Climatological deviation (`labeling/climatology_deviation.py`)

| Column | Meaning |
|---|---|
| `dust_optical_depth_mu`, `_sigma` | Real MCD-analogue climatology (Montabone et al. MY34-36) mean/std at this cell-sol's (lat, lon, Ls, LTST) |
| `dust_optical_depth_z` | `(x - mu) / max(sigma, epsilon)` — Eq. 4.1 |

No `surface_temperature_k_mu`/`_sigma`/`_z` — dust-only, no real temperature
climatology (caveat 5).

## Labels (`labeling/anomaly_labels.py`) — see `planning/LABELING_DOCUMENTATION.md` §1 for exact definitions

| Column | Meaning |
|---|---|
| `dust_anomaly` | `z_dust > tau` (one-sided) |
| `atmospheric_anomaly` | `= dust_anomaly` (no OR term — dust-only) |
| `storm_episode` | `atmospheric_anomaly` gated by the persistence+area threshold |
| `episode_id` | Which spatiotemporal connected component this cell-sol belongs to (NaN if none) |
| `horizon_1sol`, `_3sol`, `_5sol` | "Any `storm_episode` in the next h sols, same cell" |

There is no `temperature_anomaly` column anywhere in this pipeline.

## Episode catalogue (`labels/episode_catalogue.csv`)

| Column | Meaning |
|---|---|
| `episode_id` | Unique per connected component |
| `onset_sol`, `end_sol`, `duration_sols` | Episode's sol extent |
| `n_member_cell_sols`, `n_distinct_cells` | Episode size measures |
| `peak_area_cells` | Maximum simultaneous cell count at any one sol — the gating spatial-extent metric |
| `centroid_lat`, `centroid_lon` | Mean of member cells' coordinates |
| `longitude_sectors_spanned` | Of 3 × 120° sectors — used by the planet-encircling taxonomy test |
| `qualifies` | Whether this episode passed the persistence+area gate (only qualifying episodes' rows get `storm_episode=True`) |
| `taxonomy_class` | `local`/`regional`/`planet_encircling`/`unclassified` — descriptive only, never an ML target. Real breakdown: `CHAPTER_4_PARAMETER_VALUES.md` Section 1 |

## Per-task target columns (`modeling/datasets.py`) — see `planning/MODELING_DOCUMENTATION.md` §5

| Column | Meaning |
|---|---|
| `target_dust_optical_depth_h{1,3,5}` | `dust_regression` target: `dust_optical_depth` at sol `s+h`, genuinely observed only |
| `target_storm_horizon_h{1,3,5}` | `storm_detection` target: copy of `horizon_{h}sol` |

There is no `target_temperature_z_h{1,3,5}` column — the temperature-regression task
was descoped entirely, not just left untargeted. Feature columns (32 `feat_`-prefixed
engineered columns in the real run) are documented separately in
`planning/FEATURE_DICTIONARY.md`.
