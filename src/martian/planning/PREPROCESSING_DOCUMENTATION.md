<!-- AUTO-MAINTAINED (by hand). Update when preprocessing/*.py or configs/grid.yaml,
     configs/quality_control.yaml, configs/aggregation.yaml change. -->

# Preprocessing Pipeline Documentation

**Status: implemented, tested, and actually run against real data.** This layer is
data-source-agnostic by construction (nothing in `preprocessing/` reads or cares
whether the retrievals it's given came from a real or simulated provider) — it
consumes whatever `RealEMIRSObservationProvider` produces (documented in
`planning/SCHEMA_DOCUMENTATION.md`), never generates raw retrievals itself.

## 1. Grid geometry (`preprocessing/grid.py`, `configs/grid.yaml`)

| Grid | Resolution | Lat bands | Lon bands | Nominal cells |
|---|---|---|---|---|
| Primary | 5.0° | 36 | 72 | 2,592 |
| Sensitivity | 2.5° | 72 | 144 | 10,368 |
| Sensitivity | 10.0° | 18 | 36 | 648 |

- Longitude: 0-360°E, **with wraparound** (`np.mod(lon, 360)`); band 359.9°
  and band 0.05° are adjacent.
- Latitude: -90 to 90°, **no wraparound** — the poles are real edges of the
  grid, not a torus.
- Adjacency: 8 neighbours for interior-row cells; **5 neighbours** for cells
  in the topmost or bottommost latitude band, because there is no row beyond
  the pole to include. This is deliberate, not a bug — see
  `tests/test_grid.py::test_neighbours_pole_cell_has_five`.
- Polar cells are retained (never merged or excluded); `configs/grid.yaml`
  records the numerical justification (Mars radius/circumference, cell size
  vs. documented EMIRS footprint scale).

## 2. Quality control (`preprocessing/quality_control.py`, `configs/quality_control.yaml`)

Applied as a **first-failing-rule hierarchy** in strict id-ascending order —
a retrieval is rejected under the first enabled rule it fails, never
double-counted:

| id | Rule | Source | Enabled? |
|---|---|---|---|
| 1 | `missing_required_variable` | VERIFIED_PRODUCT_DOCUMENTATION | Yes |
| 2 | `invalid_quality_flag` | VERIFIED_PRODUCT_DOCUMENTATION | Yes |
| 3 | `impossible_coordinate_or_timestamp` | RESEARCHER VERIFICATION REQUIRED | Yes |
| 4 | `negative_optical_depth` | VERIFIED_PRODUCT_DOCUMENTATION | Yes |
| 5 | `physically_impossible_temperature` | VERIFIED_PRODUCT_DOCUMENTATION | Yes |
| 6 | `excessive_retrieval_uncertainty` | VERIFIED_PRODUCT_DOCUMENTATION (threshold: RESEARCHER VERIFICATION REQUIRED) | Yes |
| 7 | `unacceptable_viewing_geometry` | VERIFIED_PRODUCT_DOCUMENTATION (threshold: RESEARCHER VERIFICATION REQUIRED) | **No — disabled for real data** |
| 8 | `duplicate_retrieval` | RESEARCHER VERIFICATION REQUIRED | Yes |

Rules 1, 2, 4, 5, 6, 7 correspond to the six-rule QC hierarchy named in
enhanced Ch.4 §4.3 ("missing variable, invalid quality flag, negative
optical depth, physically impossible temperature, excessive uncertainty and
viewing-geometry exceedance"). Rules 3 and 8 are engineering additions not
itemised in that text. **Every numeric threshold** is RESEARCHER VERIFICATION
REQUIRED regardless of whether the rule's existence is chapter-grounded.

**Rule 7 is disabled because no real EMIRS field corresponds to viewing
geometry** — confirmed against real downloaded data, unlike an earlier
speculative schema that assumed one existed. `_check_unacceptable_viewing_geometry`
is defensive against the missing column (returns "no failures," not a crash)
so the rule stays inert even if re-enabled without the field being present.

**Real QC breakdown, from the completed real run** (1,385,897 input retrievals,
326,323 accepted, 23.5%): `impossible_coordinate_or_timestamp` rejected 531,368;
`physically_impossible_temperature` rejected 254,506 (of 257,704 flagged);
`excessive_retrieval_uncertainty` rejected 273,700 (of 559,256 flagged); every
other rule rejected 0. Full breakdown and discussion:
`CHAPTER_4_PARAMETER_VALUES.md` Section 3.

## 3. Aggregation (`preprocessing/aggregation.py`, `configs/aggregation.yaml`)

QC-accepted retrievals are aggregated to one row per `(cell_id, mars_year,
sol)`. Variables with a documented uncertainty companion (`dust_optical_depth`,
`water_ice_optical_depth`) use an inverse-variance weighted mean:

```
weighted_mean = sum(v_i / u_i^2) / sum(1 / u_i^2)
combined_uncertainty = sqrt(1 / sum(1 / u_i^2))
```

(`u_i` floored at `epsilon = 1e-6` to avoid division by zero). `surface_pressure_pa`,
`latitude`, `longitude`, `solar_longitude_ls`, `local_true_solar_time` use a simple
arithmetic mean. There is no `surface_temperature_k`/`atm_temp_k_level_*` aggregation
— dust-only, real climatology limitation (`CHAPTER_4_PARAMETER_VALUES.md` caveat 5),
and no real per-level atmospheric-temperature uncertainty exists to weight by even if
temperature were in scope (Section 1's vertical-grid reconciliation note).

Primary settings: `min_obs_per_cell_sol=1`, `min_pressure_level_coverage_frac=0.5`.
Every populated cell-sol is retained regardless of these thresholds; they
only control the `low_support_flag` column (via `compute_low_support_flag()`),
never row exclusion. **For real data, `pressure_level_coverage_frac` is always
NaN** (no `atm_temp_k_level_*` columns exist to compute coverage over) —
treated as "not applicable" rather than a failure, so `low_support_flag` in
practice is driven entirely by the observation-count threshold for real runs.
`observation_count` and `uncertainty_summary` are always retained.
`interpolation_flag` is always `False` at this stage (becomes meaningful at
the missing-data stage, §6).

## 4. Sensitivity analysis (`preprocessing/sensitivity.py`) — implemented, tested, NOT executed against real data

Three one-factor-at-a-time sweeps, orchestrated by `run_all_sensitivity_analyses`:

- **Grid resolution** (2.5°, 5°, 10°) — re-runs cell assignment + aggregation
  per grid, since `cell_id` itself depends on resolution.
- **Minimum observations per cell-sol** (1, 3, 5) — cheap post-hoc
  recomputation of `low_support_flag` from the already-aggregated primary-grid
  dataframe; does not re-aggregate raw data.
- **Minimum pressure-level coverage** (0.5, 0.7, 0.9) — same cheap
  recomputation approach; not meaningful for real data currently (§3 — always NaN).

This module is unit-tested (`tests/test_sensitivity.py`) but **not wired into
`mars.run_all`'s 15-step sequence** — no sensitivity-sweep output exists under any
real run directory. If a Chapter 4 sensitivity table is needed, it would need to be
run as a separate, explicit invocation of this module against the real aggregated
data — not yet done.

## 5. Reporting (`preprocessing/reporting.py`)

Outputs carry a visible data-source stamp (`stamp_figure` / `stamp_table_markdown` /
`stamp_report` from `data_sources/provenance.py`): QC summary report (Markdown,
written by `mars.run_all` Step 3 to `tables/qc_summary.md`), spatial coverage
heatmap, cell-sol observation-count histogram, seasonal (Ls-binned) and local-time
(LTST-binned) coverage figures + tables. These reporting helpers are available but
not all are currently invoked by `mars.run_all`'s standard 15-step run — the QC
summary is; the coverage/histogram figures are ad-hoc utilities.

## 6. Missing-data handling (`preprocessing/missing_data.py`, `configs/missing_data.yaml`)

Processes each `(cell_id, mars_year)` sol-series **strictly chronologically**
(sol-ascending). A missing sol is filled **only** by carrying the most
recent **prior** observation forward (trailing / last-observation-carried-
forward) for a configured list of forecasting-input columns — never
bidirectionally, and never for a label column (enforced by a
reserved-name-pattern guard raised at config-load time, not just by
convention).

- Primary: `max_trailing_fill_gap_sols = 2`. Sensitivity sweep: `0, 1, 2, 3`
  (not executed against real data, §4).
- A sol whose gap since the last real observation exceeds the threshold is
  **dropped entirely**, never fabricated with a placeholder value. A sol
  *before* the first real observation in a series is never fillable
  regardless of threshold (no prior value exists to carry).
- Every retained row carries `is_observed`, `gap_length_sols` (0 for a
  genuine observation), and `interpolation_flag` (True only for a filled
  row), and `low_support_flag` is recomputed post-fill (`observation_count`
  is forced to 0, never fabricated, for filled rows).
- **Real run output**: `apply_trailing_fill` produced 272,681 rows from
  142,849 aggregated (observed) cell-sols — see
  `CHAPTER_4_PARAMETER_VALUES.md` Section 6.

## 7. Causal Savitzky–Golay filter (`preprocessing/causal_filters.py`, `configs/causal_filter.yaml`)

Variable: `dust_optical_depth`; window length 7; polynomial order 2;
trailing coefficient position — implemented via
`scipy.signal.savgol_coeffs(window_length, polyorder, pos=window_length-1, use="dot")`,
which places the estimated point at the **last** (most recent) sample of the
window, so the filter never reads a future observation. A structurally
separate function, `apply_centered_savgol_for_descriptive_comparison_only`
(deliberately verbose name), implements the ordinary centred variant
(`pos=None`, mid-window) for descriptive figures only; its output column
carries the suffix `_sg_centered_DESCRIPTIVE_ONLY`, and
`assert_feature_safe(columns)` raises `DescriptiveOnlyColumnError` if any
column matching that suffix is ever passed as a model feature or label.

Both variants require an unbroken, sol-contiguous window (checked via the
actual `sol` values, not row position) with no internal `NaN`; a window that
crosses a gap produces `NaN`, never a value computed from non-adjacent sols.
This is run as pipeline Step 6, but its output (`data/filled_with_causal_savgol.csv`)
is not currently threaded into the ML feature set — a documented scope boundary,
see `planning/PIPELINE_DOCUMENTATION.md`.

## 8. Test coverage

`tests/test_grid.py`, `tests/test_quality_control.py`,
`tests/test_aggregation.py`, `tests/test_sensitivity.py`,
`tests/test_reporting.py` — cell boundaries (poles, exact 5°/2.5°/10°
divisions), longitude wraparound (359.9°↔0.05°, 360°≡0°, negative
longitude), polar 5-neighbour vs. interior 8-neighbour adjacency,
first-failing-rule priority attribution (one deliberately-dirty fixture per
rule, plus a disabled-rule case, a rule-7-inert-when-column-absent case, and
a malformed-config contiguity check), hand-computed inverse-variance
weighting, and sensitivity retained-fraction arithmetic against hand-built
fixtures.

`tests/test_missing_data.py` — gap within/at/beyond threshold,
pre-first-observation non-fillability, `max_gap=0` no-op, sensitivity
monotonicity, reserved-label-pattern rejection, label-never-interpolated,
per-cell independence, and `low_support_flag` recomputation.

`tests/test_causal_filters.py` — proves, concretely: (1) a
perturbation to the last (future-most) input value never changes any earlier
causal output; (2) appending a later-mars-year "test" partition never
changes causal outputs already computed on a "train+val" partition (mirrors
`configs/splits.yaml`'s boundaries); (3) a `label` column passed through the
causal filter is returned byte-identical, and no `label*_sg_causal` column
is ever created; (4) causal and centred outputs live in disjoint,
distinctly-suffixed columns and differ numerically on non-polynomial data,
and `assert_feature_safe` rejects only the centred one; (5) loading a config
with `window_length=5` instead of the default 7 changes the observed
leading-`NaN` run from 6 to 4, proving the parameter is read at runtime, not
hardcoded; plus a direct test that a window spanning a sol-gap produces `NaN`.

Full suite (all of `src/martian`, not just this layer): 316 passed as of the
last real-data run (2026-08-05).
