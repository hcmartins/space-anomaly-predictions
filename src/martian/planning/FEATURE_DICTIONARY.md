<!-- AUTO-MAINTAINED (by hand). Update when configs/features.yaml's variable
     lists change, or when preprocessing/features.py gains/removes a
     feature family. -->

# Feature Dictionary

Every `feat_`-prefixed column produced by `preprocessing.features.build_features`,
which BOTH modeling tasks (`dust_regression`, `storm_detection`) consume from the same
shared builder (each task then selects its own subset into its own feature manifest —
see `planning/MODELING_DOCUMENTATION.md` §2). Governed by `configs/features.yaml`.
32 concrete `feat_` columns were produced in the real run
(`outputs/real/<run_id>/reports/final_report.md`'s `n_engineered_feature_columns`).

**Dust-only** — there is no `surface_temperature_k`/`_z` feature family. No real
climatology source exists for surface temperature, so it was removed from
`configs/features.yaml`'s `variables`/`z_score_variables` entirely, not just left
unused (see `CHAPTER_4_PARAMETER_VALUES.md` caveat 5).

**Causality guarantee** (structural, not just documentation): every feature
at `(cell, sol=s)` is computed from either (a) the SAME cell's rows with
`sol <= s`, or (b) a SPATIAL NEIGHBOUR's row at `sol == s` (a neighbouring
cell's value at the SAME sol is legitimately "already known" at prediction
time — it is never a future sol, and never a future cell-sol). No feature
column is ever named with a `target_` prefix, which is reserved exclusively
for per-task targets built later in `modeling/datasets.py` — this makes
feature/target confusion structurally impossible, not just avoided by
convention. Proven by `tests/test_features.py`.

Two base variable families feed every feature below:
- **`variables`**: `dust_optical_depth` (raw physical value)
- **`z_score_variables`**: `dust_optical_depth_z` (climatological deviation, from Step 7)

## 1. Lags — `feat_{var}_lag{1,2,3}`

Same cell's value exactly `{1,2,3}` sols before `s` (NaN if that sol has no
row at all, e.g. beyond the trailing-fill max-gap). `lags_sols: [1, 2, 3]`.

## 2. Trailing-window statistics — `feat_{var}_trailing5_{mean,std,max,min}`

Over the same cell's available (non-NaN) values in the trailing 5-sol
window `[s-5, s-1]`. `std` is `0.0` (not NaN) when only one value is
available in the window. `trailing_window_sols: 5`.

## 3. First difference and slope — `feat_{var}_diff1`, `feat_{var}_slope5`

`diff1` = current value minus the value at `s-1` (NaN if either is missing).
`slope5` = least-squares linear-fit slope of the trailing-window values
against their sol offsets (only computed when >= 2 valid points exist in
the window).

## 4. Persistence-run counter — `feat_persistence_run_sols`

Consecutive prior sols (same cell, walking backward from `s-1`) with
`atmospheric_anomaly == True`, stopping at the first non-anomalous or
missing prior sol. Captures "how long has this cell already been anomalous"
without looking forward.

## 5. Spatial-neighbour statistics — `feat_{var}_neighbour_{mean,max}`, `feat_neighbour_exceedance_count`

Computed across the current cell's grid neighbours (`preprocessing.grid.neighbours`,
including longitude wraparound) AT THE SAME SOL `s` — legitimate "currently
known" information, never a future sol. `neighbour_exceedance_count` counts
how many neighbouring cells have `atmospheric_anomaly == True` at sol `s`.

## 6. Seasonal/diurnal harmonics — `feat_ls_sin`, `feat_ls_cos`, `feat_ltst_sin`, `feat_ltst_cos`

`sin`/`cos` of `solar_longitude_ls` and `local_true_solar_time` (converted
to radians; LTST scaled from a 24-hour clock to 360°) — lets tree/linear
models use a continuous, wraparound-safe encoding of season and time-of-day
instead of a raw degree/hour value that discontinuously jumps at 360°/24h.

## 7. QC / interpolation indicators — `feat_is_observed`, `feat_interpolation_flag`, `feat_low_support_flag`, `feat_observation_count`

Surfaced directly from the trailing-filled series' own bookkeeping columns
(see `planning/DATA_DICTIONARY.md`) so a model can distinguish a genuine
retrieval-derived row from a carried-forward one, and gauge how much raw
support (`observation_count`) backed that cell-sol.

## Sparse-history feature columns (no non-missing values in some runs)

At real EMIRS's sparse, non-uniform spatiotemporal coverage (Section "Coverage
density" in `CHAPTER_4_PARAMETER_VALUES.md`, ~4.7% of possible cell-sols
populated), some engineered columns can end up with NO observed value across
the entire dataset — confirmed in the real run:
`feat_dust_optical_depth_z_neighbour_mean`/`feat_dust_optical_depth_z_neighbour_max`
(a cell simply never has a populated neighbour at any shared sol for at least
one task/horizon split). `SimpleImputer` then legitimately drops that column
for that model/horizon rather than imputing a fabricated constant, and logs a
`UserWarning` naming the dropped columns. This is expected, non-silent
behaviour given real coverage sparsity, not a bug.

## Feature-family toggles

Every family above is individually toggleable in `configs/features.yaml`
(`enable_lags`, `enable_trailing_stats`, `enable_first_difference_and_slope`,
`enable_persistence_counter`, `enable_spatial_neighbour_stats`,
`enable_ls_ltst_harmonics`, `enable_qc_indicators`) so a sensitivity run can
isolate one family's marginal contribution without editing
`preprocessing/features.py` itself.
