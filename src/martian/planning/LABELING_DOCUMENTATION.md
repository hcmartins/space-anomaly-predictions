<!-- AUTO-MAINTAINED (by hand). Update when labeling/*.py or
     configs/anomaly_definition.yaml, configs/storm_taxonomy.yaml,
     configs/splits.yaml change. -->

# Labeling Documentation (anomaly/storm-episode labels)

**Status: implemented, tested, and actually run against real data.** Consumes the
aggregated cell-sol dataframe from `preprocessing/`; produces the label families
below plus specification/reporting artifacts. **Dust-only** (no temperature term
— caveat 5, `CHAPTER_4_PARAMETER_VALUES.md`). **Independent-catalogue validation
(Section 4.4.1 / Table 4.6) was reinstated 2026-08-05** against a real,
independently-sourced MY36 documented dust-storm catalogue (§7 below) — an
earlier version of this document described it as removed entirely; that is no
longer true.

**PRIMARY REGIME RECONCILED 2026-08-05**: the persistence/area primary switched
from k=1/m=1 (the permissive controlled-simulation placeholder) to k=2/m=3 (the
examiner-suggested stricter regime, always intended to become primary once real
data was available — see `configs/anomaly_definition.yaml`'s own history
comment). §2 and §4 below describe the current k=2/m=3 primary; some numbers in
§4 are being regenerated against a fresh real run under this regime and the new
catalogue-validation step together (2026-08-05) — check the run timestamp cited
there before quoting a number in the thesis.

## 1. Formulas (Eq. 4.1-4.4 equivalent)

```
z_v(c,s)              = (x_v(c,s) - mu_v(c,s)) / max(sigma_v(c,s), epsilon)      [Eq 4.1]
dust_anomaly(c,s)     = z_dust(c,s) > tau                       (one-sided, high only)
atmospheric_anomaly   = dust_anomaly                                              [Eq 4.2]
storm_episode(c,s)    = atmospheric_anomaly cell-sols grouped into spatiotemporal
                        episodes (8-neighbour spatial connectivity + longitude
                        wraparound + same-cell/spatial-neighbour temporal linking
                        across |delta_sol|<=1, within one mars_year); a cell-sol's
                        label is True iff its episode's duration_sols >= P
                        AND peak_area_cells >= A_min                              [Eq 4.3]
horizon_{h}sol(c,s)   = any(storm_episode(c,s') for s < s' <= s+h), same cell,
                        same mars_year -- "any event within the future horizon"    [Eq 4.4]
```

There is no `temperature_anomaly` term and no OR — `atmospheric_anomaly` is
literally `= dust_anomaly` (`labeling/anomaly_labels.py::compute_exceedance`).
`mu_v`/`sigma_v` come from `RealMCDClimatologyProvider.query()` — real, built from
Montabone et al. MY34-36 data (see `planning/SCHEMA_DOCUMENTATION.md`). Implemented
in `labeling/climatology_deviation.py` (z-scores), `labeling/anomaly_labels.py`
(exceedance, gate, horizon), and `labeling/connected_components.py` (episode
construction).

## 2. Primary vs. sensitivity settings

`configs/anomaly_definition.yaml`'s primary regime uses `tau_primary=2.0`,
**persistence_sols=2, min_area_cells=3** — the examiner-suggested stricter gate,
promoted from sensitivity to primary 2026-08-05 (see the file header note
above). Under this regime `storm_episode` is a genuine SUBSET of
`atmospheric_anomaly` (not identical to it, unlike the earlier k=1/m=1
placeholder) — a qualifying episode must actually persist >=2 sols and reach
>=3 cells of peak extent, so the gate does real work. `tau`/persistence/area
sensitivity sweeps (`sensitivity:` block in the config, k=1/m=1 now the
permissive end of that sweep rather than the old primary) computed by
`labeling/anomaly_labels.py::run_anomaly_sensitivity`. **Not executed against
real data** as part of `mars.run_all`'s standard 15-step run (same status as
`preprocessing/sensitivity.py` — see `planning/PREPROCESSING_DOCUMENTATION.md` §4).

## 3. Spatiotemporal episode construction (`labeling/connected_components.py`)

Two exceedance cell-sols within the SAME mars_year are connected iff they are
same-sol 8-neighbours (longitude wraparound via `preprocessing.grid.neighbours`)
OR adjacent-sol (`|delta_sol|==1`) AND either the same cell or 8-neighbours. This is
the standard 26-connectivity analogue for a 3D (lat, lon, time) volume — it correctly
merges a storm that **drifts** to a neighbouring cell between consecutive sols into
one episode, not just a storm that stays in exactly the same cell. `mars_year`
equality is enforced on every edge so sol indices can never create a false
cross-year connection. Implemented via a plain Union-Find over exceedance rows only,
so it stays fast even considering up to 26 candidate neighbours per node.

`labeling/storm_taxonomy.py::classify_episode_catalogue` attaches a
**descriptive-only** local/regional/planet_encircling classification to each
candidate episode (never the ML target — `configs/storm_taxonomy.yaml`'s own
`classification_target: "reporting_only"`).

## 4. Real empirical findings

**PENDING REGENERATION (2026-08-05)**: the numbers below are from the
k=1/m=1-primary run (`run_20260805T062557386507Z_seed42`) and are being
superseded by a fresh run under the current k=2/m=3 primary + Step 9
catalogue validation together — do not quote these specific counts in the
thesis until this note is removed and replaced with the new run's numbers
(check `CHAPTER_4_PARAMETER_VALUES.md` for the authoritative, up-to-date
values, which are regenerated together with this document).

1,385,897 real retrievals → 326,323 QC-accepted → 142,849 aggregated cell-sols.

| Label | Count | Rate |
|---|---|---|
| `dust_anomaly` = `atmospheric_anomaly` | 18,776 | 13.14% |
| `storm_episode` (OLD primary, P=1/A_min=1) | 18,776 | 13.14% (identical, as predicted in §2's old regime) |

4,436 candidate episodes were identified via connected-component analysis (2
excluded for crossing a chronological split boundary, `mars.run_all` Step 8). Real
taxonomy classification of the 4,434 remaining episodes (OLD k=1/m=1 run): **4,330
local (97.7%), 73 regional (1.6%), 0 planet_encircling, 31 unclassified (0.7%)** —
genuine spatiotemporal spread (median episode duration 1 sol / area 1 cell, but up
to 29 sols duration and 309 cells peak area), unlike the old simulated pipeline
where the maximally-permissive gate produced 100% `local` classification by
construction (a sampling-model artefact, not a real finding — see history note
below). Under the new k=2/m=3 primary, `storm_episode` is expected to be a
materially smaller, stricter subset of `atmospheric_anomaly` than shown here.

Per-horizon target prevalence (test split, OLD k=1/m=1 run): horizon-1-sol ≈1.3%,
horizon-3-sol ≈4.0%, horizon-5-sol ≈5.7% — real class imbalance, growing with
horizon length (more chances for a qualifying storm to occur within a longer
window).

**History note (no longer current)**: an earlier version of this document, describing
the now-removed simulated pipeline, reported 100% `local`-only classification and a
`temperature_anomaly` prevalence of ~33% attributed to a simulated-climatology
mismatch. Both were artefacts of the deleted synthetic generator's sampling model,
not real findings — see `planning/PREPROCESSING_DOCUMENTATION.md` for the general
pattern (sparse/uniform independent sampling rarely produces multi-sol/multi-cell
episodes).

## 5. Independent-catalogue validation (Section 4.4.1 / Table 4.6) — reinstated 2026-08-05

`labeling/catalogue_validation.py` matches this pipeline's own detected,
QUALIFYING `storm_episode`s (never raw `atmospheric_anomaly`) against a real,
independently-sourced documented dust-storm catalogue
(`data_sources/real_documented_events.py`) — real data on BOTH sides now, no
synthetic ground-truth/independent-catalogue pair as in the pre-real-only
design. Runs as `mars.run_all` Step 9, right after episode construction
(Step 8), and is deliberately OPTIONAL: if the real catalogue file isn't
present, it's recorded as `{"status": "NOT PERFORMED"}` in the run's
`pipeline_summary`, never a run failure and never a fabricated result.

**The real catalogue**: Guha, B. K., et al. (2024), "Seasonal and Diurnal
Variations of Dust Storms in Martian Year 36 Based on the EMM-EXI Database,"
J. Geophys. Res. Planets, 129, e2023JE008156 — dataset itself at
doi:10.5281/zenodo.10275974 (Zenodo, CC BY 4.0). Chosen after checking the two
well-known Mars dust-storm catalogues (MDAD, Battalio & Wang 2021, MY24-32;
MDSSD, MY24-33) and confirming both end before real EMIRS coverage begins
(2021-02-20, MY36) — neither can validate this pipeline's real labels. The
Guha et al. catalogue covers exactly MY36, from the same mission (Emirates
Mars Mission) as the real EMIRS data, and reports exactly the fields needed:
storm onset/end time (via each detection's own EXI image filename, parsed the
same way as real EMIRS filenames), centroid lat/lon, and areal extent.
Downloaded 2026-08-05: 248 raw per-image detections collapsing to 98 distinct
storms (`data_sources/real_documented_events.py::load_real_documented_events`
groups by the catalogue's own `Storm no.` column) — both counts independently
confirmed against the source paper's own reported totals.

**MY36 only.** The real catalogue has no MY37 coverage. Both detected episodes
and documented events are restricted to `mars_year == 36` before matching
(`run_catalogue_validation`) — real MY37 episodes are neither penalised nor
credited, not silently treated as a miss.

**Matching rule** (`configs/catalogue_validation.yaml`, RESEARCHER
VERIFICATION REQUIRED IN FULL — carried over unchanged from the pre-real-data
design, not independently confirmed against this real catalogue's own
registration accuracy): temporal — the documented event's `[onset_sol, end_sol]`
window, expanded by `temporal_tolerance_sols=2` on each side, must overlap a
qualifying episode's own `[onset_sol, end_sol]` window (interval overlap, not
"onset within 2 sols"). Spatial — the documented event's centroid cell is
BFS-expanded by `spatial_region_radius_cells=1` grid-step
(`preprocessing.grid.neighbours`) into a candidate region; a match requires at
least one of the episode's member cells (recovered from `labeled_df` via
`episode_id`) to fall in that region.

**Three things this module is careful never to conflate** (unchanged discipline
from the pre-real-only design): (1) event detection (does a detected episode
match a documented event at all) vs. (2) event classification — this
pipeline's own cell-count-based `storm_taxonomy` classification is never
compared against the real catalogue's own `peak_size_km2` (physical units,
no footprint-to-cell-count conversion has been built) — Table 4.6 validates
detection only; (3) atmospheric-anomaly detection — explicitly out of scope,
since the catalogue records discrete events, not pointwise exceedances.

Outputs: `tables/table_4_6_independent_catalogue_validation.{csv,md}`
(run-level, via `labeling/catalogue_validation.py::write_table_4_6`),
`labels/catalogue_matched_pairs.csv`, `labels/catalogue_documented_events.csv`,
`labels/catalogue_episode_status.csv` (per-episode hit/`false_positive_candidate`
status — an unmatched episode is never asserted a confirmed false alarm, since
the documented catalogue itself is not claimed to be exhaustive).

## 6. Reporting-only split/season tagging — scope note

`labeling/reporting.py::assign_split`/`assign_mars_season`/`prevalence_by_*`/
`episode_map`/`episode_timeline` are real, tested utilities for descriptive
prevalence-by-split/season/horizon/taxonomy tables and figures, but **are not
currently invoked by `mars.run_all`'s standard 15-step run** — they are available
library functions, not part of the reproducible pipeline output. `assign_split`
here is deliberately NOT the leak-audited `splitting/chronological_split.py` module
used by `mars.run_all` itself (no scaler-leakage check, no "episode never crosses a
split boundary" assertion) — treat any output from this module's split tagging as
descriptive only. Season tagging (`assign_mars_season`) uses the standard
areocentric-longitude (Ls) quadrant convention (0-90/90-180/180-270/270-360 =
Northern Spring/Summer/Autumn/Winter) — a definitional convention, not a data claim.

## 7. Test coverage

`tests/test_anomaly_labels.py` (using a deterministic fixed-mu/sigma MCD stub for
exact hand-computed z-scores) — z-score formula, dust one-sidedness, the dust-only
`atmospheric_anomaly = dust_anomaly` identity, the persistence gate, the area gate,
horizon "true within window / false beyond" and "never looks at the past,"
interpolated-row exclusion, and runtime config loading.

`tests/test_connected_components.py` — 8-neighbour same-sol merging, non-adjacent-cell
separation, longitude wraparound merging, same-cell temporal linking across
consecutive sols, non-consecutive-sol non-merging, spatiotemporal drift merging (the
key test that this is genuinely spatiotemporal, not just spatial-OR-temporal), the
mars_year boundary (same cell/sol number but different year never merges), pole-row
neighbour-count correctness, empty-input handling, and hand-computed episode
duration/peak-area summarisation.

`tests/test_storm_taxonomy.py`,
`tests/test_labeling_reporting.py` — taxonomy classification per band (including the
"matches no band" and "planet_encircling requires all three conditions" cases),
runtime config loading, split-boundary assignment,
season-quadrant assignment, prevalence arithmetic, and figure data-source stamping.

`tests/test_real_documented_events.py` — the real Guha et al. catalogue loader:
collapsing multiple per-image detections into one row per `Storm no.`,
West-to-0-360-East longitude conversion, citation metadata attachment, a clear
error when the catalogue file is missing, and a regression test for a real bug
(the source file's `Image` column header has a trailing space).

`tests/test_catalogue_validation.py` — hit when both temporal and spatial
criteria are satisfied, miss when spatially disjoint, the temporal-tolerance
boundary (exact vs. one sol beyond), `false_positive_candidate` labeling for
an unmatched qualifying episode, non-qualifying episodes never matched,
restriction to the documented catalogue's own Mars Year, and
`build_table_4_6` reading directly from the validation summary.

Full suite (all of `src/martian`): 330 passed as of 2026-08-05 (up from 316 —
+14 for the reinstated catalogue-validation feature).
