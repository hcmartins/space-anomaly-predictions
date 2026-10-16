# Chapter 4 [INSERT]/[CONFIRM]/[VERIFY] Values — Real, From This Repository

Every value below is read directly from real, working, tested code and real, executed
pipeline runs in this repository (`src/martian`), never invented to make the chapter read
cleanly. Where the actual implementation does not do what the chapter's prose assumes, that
is stated plainly rather than glossed over — consistent with this project's no-fabrication
convention. Source file/line references are given so every value can be re-verified
independently.

**2026-08-13 (completion pass): all four critical problems fully resolved, including the
full pipeline retrain — see Section 13.** Following the earlier "implementation pass"
below (split redesign, `modeling/train.py` leakage fix, sentinel-row ingestion fix,
window-completeness infrastructure — all confirmed and code-complete), this pass: ran the
real 18-combination missing-data sensitivity sweep on train/val evidence only (design
confirmed with the researcher first: gradient-boosting proxy model, val TSS/MAE selection
criterion — 4.6 hours of real compute) and applied its selection
(`max_trailing_fill_gap_sols=1`, `min_window_completeness_frac=0.6`) to the primary
config; then ran the complete corrected pipeline end-to-end
(`run_20260813T195013457439Z_seed42`, ~75 min, all leakage checks passed) and regenerated
Figures 4.5-4.8 against it. **Headline result**: sequence-model storm-detection TSS drops
substantially at every horizon relative to the superseded 5 August run (e.g. GRU w=10 at
h=5: 0.258 → 0.154) — the expected signature of a genuine test-leakage fix, not noise;
`persistence` and classical ROC-AUC are essentially unchanged, a useful internal
consistency check. Test-split positive counts (440/1,186/1,637) and Table 4.6b are
essentially unchanged, confirming MY37 itself was barely touched by the other three
fixes. Full before/after comparison, and the explicit MY37-integrity evidence chain:
Section 13. Full suite 378/378 passing throughout.

**2026-08-13 (implementation pass): real code fixes for the four critical problems,
stopping short of the final retrain by design.** In response to the examiner's four
critical-problems list, implemented and verified: (1) a redesigned MY36 train/val split
(`configs/splits.yaml`, boundary sol 432, zero episode crossings, real positive counts at
every horizon) plus the actual `modeling/train.py` code fix so LSTM/GRU early stopping
uses it — caveat 9 / Section 11 item 5; (2) the sentinel-row ingestion fix
(`data_sources/real_emirs_fits.py`, `chi2temp`/`chi2a`-based, the most reliable available
metadata signal), re-run against the real archive with revised counts reported — Section
11 item 6, and found the real issue affects 22.2% of raw retrievals, ~68x the earlier
0.33% aggregated-footprint estimate; (3) the missing-data sensitivity comparison's missing
third axis (window-completeness fraction) implemented in `preprocessing/features.py` —
Section 11 item 7. All changes covered by new tests; full suite 378/378 passing. The
missing-data sensitivity sweep execution and the full pipeline retrain were deliberately
deferred past this entry — see the completion pass above for both.

**2026-08-13: reconciled this document against the chapter's own 2026-08-13 draft (which
now incorporates Table 4.6b and Figures 4.5-4.8 inline, and states several issues as
critical/unresolved in its own prose) plus an examiner pass flagging four critical
problems. No code was changed in this pass — this was a documentation-accuracy pass
only.** Findings: (1) caveat 9 (LSTM/GRU early-stopped on the test split) is upgraded
from "a real, if narrow, form of test-set influence" to CRITICAL, matching the chapter's
own new language ("invalidates the claim of a fully clean held-out evaluation") — genuinely
NOT fixed yet, see Section 11 item 5 for the exact remediation the chapter itself now
specifies. (2) The sentinel-coordinate (0°,0°) rows are confirmed excluded from
representative-cell figures ONLY, not from the pipeline — and the chapter's own Algorithm
4.1 pseudocode ("drop rows with unparseable utc or latitude=longitude=0") was checked
against the real ingestion code and found to overstate it: only the unparseable-UTC half
is actually implemented (`data_sources/real_emirs_fits.py` line ~161), there is no
lat=lon=0 filter anywhere — see Section 11 item 6. (3) Two new required-work items
confirmed never executed: a missing-data/gap-fill sensitivity comparison (the sweep
configs exist but `run_all.py` never calls them, and no sensitivity output exists in the
canonical run — Section 11 item 7) and formal preprocessing-validation acceptance
thresholds (none exist — item 8). (4) Figures 4.3 and 4.4 are confirmed still schematic
per the chapter's own captions, distinct from the already-real Figures 4.5-4.8 (Section 11
item 9). (5) Added: Table 4.8's real CDOD-regression test-N per horizon
(13,682/16,590/22,227, Section 2) and a terminology-reconciliation note for the chapter's
"dust-anomaly-episode classification" renaming of "storm detection" (Section 2 — code
identifiers unchanged). The strengthened-bootstrap and Monte-Carlo items (Section 11,
items 1-2) are unchanged from the prior pass. None of items 5-9 were implemented in this
pass — they are documented as open, matching the chapter's own current honest framing,
not silently fixed.

**2026-08-12 (third pass): generated real Figures 4.5-4.8 (open item 3), resolving it —
2 of the original 5 outstanding items now remain (strengthened bootstrap re-run, CDOD
label-flip Monte Carlo).** New `evaluation/preprocessing_figures.py` (26 tests, 369
total) plus `generate_figure_4_5_raw_data.py` / `generate_chapter4_preprocessing_figures.py`,
run for real against the canonical run. Found and fixed two real bugs along the way (a
NaN-poisoned pooled ACF that hid real structure — a genuine lag-9 spike — behind
near-total NaN; a Mars-Year-mixing bug in Figure 4.5's raw-retrieval extraction), and
surfaced two new real-data findings not previously documented: a geolocation-sentinel
cell (1296, likely full-disk/calibration frames defaulting to lat=lon=0, 0.33% of
aggregated cell-sols) that had to be excluded from representative-cell selection; and a
heavy-tailed climatological z-score distribution (1,323/142,464 rows with |z|>50, max
z≈2026, from real but small sigma_clim in some climatology bins) that does not affect any
label but required switching Figure 4.7's composite/semivariogram to robust
(median/IQR-based) statistics. Full writeup: Section 12.

**2026-08-12 (second pass): implemented and ran the blind dev/test tolerance
re-validation (open item 4), resolving it — 3 of the original 5 outstanding items now
remain.** New `labeling/catalogue_tolerance_tuning.py` +
`configs/catalogue_tolerance_tuning.yaml` + `mars/run_all.py` Step 9b + 20 new tests (343
total, up from 331). Per researcher decision, tolerance search uses F1 of episode
detection as the selection criterion, on a fixed-seed 49/49 random dev/test split of the
98 documented MY36 events (`seed=42`); the DEV-selected tolerance is then applied once to
the held-out TEST half. Run for real against the canonical run's own saved labels — see
Section 7's "Table 4.6b" subsection for the full result and its honest limitations (the
selected tolerance sits at the edge of the searched grid, and F1 does not plateau even
well outside it — real evidence that the gap between this pipeline's detector and the
Guha et al. catalogue is a detection-methodology mismatch, not primarily a
registration-tolerance problem). Section 11's open-items list is updated accordingly.

**2026-08-12 (first pass): resolved the two items answerable from existing code/run
outputs; confirmed the remaining three as genuinely requiring new execution, not
documentation lookup.**
Investigated against the canonical run (`run_20260805T115705602822Z_seed42`) and
`modeling/sequence_models.py` / `modeling/train.py`: (1) exact real per-horizon positive
counts for storm detection, and full reconciliation of the 142,673-vs-142,849 row residual
(Section 8) — **the chapter draft's placeholder prevalence numbers (0.71%/1.77%/2.47%,
≈360/896/1,251) do not match the real run and must be corrected**, see Section 8; this
investigation also surfaced a real, previously undocumented ~200-sol data gap and an
early-stopping methodology issue, both flagged below (new caveat 9). (2) LSTM/GRU
optimiser, loss function, and batch size, confirmed exactly from source (Section 9).
The remaining three items — strengthened bootstrap re-run, CDOD-perturbation label-flip
analysis, and Figures 4.5-4.8 (a fourth, blind dev/test catalogue-tolerance
re-validation, was resolved in the same-day second pass below) — were confirmed **not
yet implemented anywhere in this repository** — see Section 11. These are not values
sitting unread in a log file; they require new code and a new pipeline/figure run,
exactly as flagged when they were proposed.

**2026-08-05 (second pass): reconciled the label regime and reinstated Table 4.6.**
Two things changed since the first real run: (1) the persistence/area primary regime was
switched from k=1/m=1 (a controlled-simulation-era placeholder) to k=2/m=3 (the
examiner-suggested stricter gate, always intended to become primary once real data was
available — `configs/anomaly_definition.yaml`'s own history comment); (2) Section 4.4.1's
independent-catalogue validation (Table 4.6), previously descoped for lack of a real
catalogue, was reinstated against a real, freely-licensed MY36 EMM/EXI dust-storm database
(Guha et al. 2024) — see caveats 2, 5, 6, and Section 7. The pipeline was re-run in full
under both changes together (`run_20260805T115705602822Z_seed42`); every number below is
from that run unless stated otherwise.

**2026-08-05 (first pass): rewritten after the pipeline was run end-to-end against real
EMIRS data for the first time.** 7,445 real EMIRS L3 FITS files were downloaded
(`download_emirs_sdc.py`, EMM Science Data Center REST API), a real column-name mapping
onto the pipeline's schema was built, and two real, previously-unknown data-quality bugs
were found and fixed (caveat 7 — a third was found and fixed in the second pass).

---

## 0. Scope caveats to read before filling anything in

1. **Two data sources, both real and both actually run now.** EMIRS
   (`data_sources/real_emirs_provider.py`, `data_sources/real_emirs_fits.py`) is real, and
   real FITS files have been downloaded and loaded: 7,445 files (`l3atm` + `l3emiss`) in
   `data_cache/emirs/raw`, covering **2021-02-20 to 2025-01-14** (EMIRS's real science
   start through the latest data the SDC had released as of the 2026-08-04 download — no
   data exists in the SDC beyond that; February–November 2025 metadata queries returned zero
   files). MCD's dust climatology (`data_sources/real_mcd_provider.py`) is built from real,
   already-downloaded Montabone et al. gridded column-dust-optical-depth (CDOD) climatology
   data for Mars Years 34–36 (`data_cache/griddedcdod_MY{34,35,36}_v2-5.nc`) via
   `build_mcd_climatology_from_montabone.py`. It covers `dust_optical_depth` only — no real
   climatology source exists for `surface_temperature_k` (see caveat 5).
2. **The canonical run is now `outputs/real/run_20260813T195013457439Z_seed42/`**
   (seed=42, 2026-08-13) — run under the corrected split, the sentinel-row ingestion fix,
   and the sensitivity-selected missing-data configuration, with the LSTM/GRU test-set
   leakage fixed in code. See Section 13 for the full writeup of what changed and why.
   **`run_20260805T115705602822Z_seed42` (2026-08-05) is now superseded, not deleted** —
   its own model-performance numbers should not be cited going forward (they are
   downstream of the leakage bug and the sentinel-row contamination), but its intermediate
   files remain on disk and this document still cites specific values from it explicitly
   where the comparison itself is the point (Section 13). Two earlier runs were already
   superseded before this pass: `run_20260805T062557386507Z_seed42` (correct labeling, old
   k=1/m=1 regime, no Table 4.6) and `run_20260804T195819221819Z_seed42` (same k=1/m=1
   numbers, plus a since-fixed metadata bug that stamped every figure/report `"SIMULATED"`
   despite being real).
3. **The real EMIRS-to-pipeline column mapping is built and working**
   (`configs/real_emirs_schema.yaml`'s `pipeline_column_mapping`, applied by
   `RealEMIRSObservationProvider.load()`). Real field names (`taudust`, `kinetic_temp`,
   `utc`, `qls`, `qlt`, ...) are renamed onto the pipeline's canonical names
   (`dust_optical_depth`, `surface_temperature_k`, `utc_timestamp`,
   `solar_longitude_ls`, `local_true_solar_time`) at load time. This was the last
   structural gap before the pipeline could run against real data; it is now closed.
4. **MCD scenario and interpolation method are confirmed real values (2026-08-03,
   researcher-supplied guidance from the LMD/IPSL team).** `configs/mcd.yaml` states, as
   CONFIRMED: `scenario: "climatology (average solar/EUV conditions)"` and
   `interpolation: "Linear interpolation in (Ls, local time, latitude, longitude) from the
   MCD's native grid"` — correcting an earlier chapter draft that described "built-in
   high-resolution grid interpolation." `mcd_version` is recorded as the **target** build,
   `"MCD v6.1"` — not a version confirmed against a running installation, since the
   `aymeric-spiga/mcd-python` wrapper this would require has not been compiled. **The dust
   climatology actually built and used here comes from a different, real source**
   (Montabone et al., caveat 1) — its own netCDF attrs correctly cite Montabone et al., not
   "MCD v6.1/Millour et al."; do not conflate the two when citing a source in the chapter.
5. **Task 2 (temperature-anomaly forecasting) is descoped, not unresolved; Section 4.4.1
   (independent-catalogue label validation) has been REINSTATED, not descoped.** No real
   climatology source exists for `surface_temperature_k` (temperature is no longer part of
   the anomaly-label formula, Eq. 4.2, at all — see Section 1) — Task 2 remains removed from
   the codebase entirely, and this chapter should describe two forecasting tasks
   (dust-opacity regression, storm detection), not three. Section 4.4.1 / Table 4.6, by
   contrast, is real and working again as of 2026-08-05 — see caveat 2 and Section 7.
6. **The primary persistence/area regime was reconciled 2026-08-05.** k=1/m=1 (the
   controlled-simulation-era placeholder, under which `storm_episode` was definitionally
   identical to raw `atmospheric_anomaly`) has been replaced by k=2/m=3 (the
   examiner-suggested stricter gate) as primary — see `configs/anomaly_definition.yaml`'s
   own history comment and `planning/LABELING_DOCUMENTATION.md`. k=1/m=1 is now the
   permissive end of the sensitivity sweep instead. This chapter should describe k=2/m=3 as
   the primary regime throughout — every `storm_episode`/prevalence/model-result number in
   this document already reflects it.
7. **Three real, previously-unknown data-quality bugs were found and fixed while bringing
   real data online (2026-08-04/05), all worth a Chapter 4/7 mention as genuine real-data
   engineering findings:**
   - Real `utc` strings mix a `.mmm` fractional-second suffix with plain `HH:MM:SS` within
     the same file; a naive single-format parse silently turned up to ~80% of rows in some
     files into missing timestamps. Fixed (`format="mixed"` parsing).
   - Real l3atm/l3emiss files contain genuine fill/sentinel rows (unparseable `utc`,
     `latitude=longitude=0`) — up to ~37% of rows in some files. Left in, every such row
     shared an identical merge key, and the l3atm/l3emiss inner join produced a
     many-to-many cartesian product for that one key, attempting to allocate a 120 GiB
     array. Fixed by dropping unparseable-timestamp rows at load time (a legitimate filter
     regardless: such rows can never get a `mars_year`/`sol` and are unusable downstream).
   - The real documented-catalogue source (Section 7) contains a storm that genuinely
     straddles a Mars-Year boundary (real detections at MY36 sol 665-666 and MY37 sol 1,
     four real days apart). Naively taking min/max sol across all of that storm's
     detections read the post-boundary sol≈1 rows as an EARLIER onset than the
     pre-boundary sol≈665 rows, producing a spurious ~665-sol span for one event and
     inflating Table 4.6's hit/precision counts in the run's first pass. Fixed by
     restricting each event's onset/end to the detections on the same side of the boundary
     as its own assigned Mars Year.
8. **No CRPS, and no probabilistic/ensemble scoring of any kind, exists anywhere in this
   repository.** Only point-estimate regression metrics (MAE, RMSE, MSE, R², bias,
   correlation) and point-estimate classification metrics are implemented.
9. **CRITICAL, STILL UNRESOLVED as of 2026-08-13 — the `val` split is genuinely empty (0
   rows) in the real run, the LSTM/GRU early-stopping criterion is monitored against the
   TEST split as a result, and this constitutes real test-set leakage, not a narrow or
   merely disclosable qualification.** `splits/split_manifest.json` for the canonical run:
   `val.n_rows = 0`. Root cause, confirmed against `data/aggregated_cell_sol.csv`: MY36
   aggregated cell-sols run sol 12-464 (nothing sol 465-668, the configured val range) and
   MY37 aggregated cell-sols run sol 188-668 (nothing sol 1-187) — a genuine ~204-sol gap in
   *aggregated, QC-accepted* coverage spanning the end of MY36 into MY37 (whether this
   reflects a real raw-retrieval sampling gap or a QC-rejection concentration in that window
   has not been checked — real, but not yet root-caused). Consequently
   `modeling/train.py`'s call to `train_sequence_model(X_train_seq, y_train_seq,
   X_test_seq, y_test_seq, ...)` (line ~317) passes the TEST split, not a held-out
   validation split, as `X_val`/`y_val` — and `train_sequence_model`
   (`modeling/sequence_models.py`) uses exactly that argument for its early-stopping
   patience check. This means, for every sequence model in this run, training DURATION
   (number of epochs, via early stopping) was selected using the test split's own loss.
   The decision threshold and hyperparameter VALUES genuinely were not tuned on test (no
   search occurred, Section 9) — but sequence-model training length was, and the chapter's
   own §4.7 (2026-08-13 draft) now states this plainly: it "invalidates the claim of a
   fully clean held-out evaluation for the LSTM and GRU," and "disclosure does not repair
   the experiment" (examiner's wording, concurred with). **This has NOT been fixed in this
   repository as of this entry** — no non-empty validation split, no retraining, and no
   re-evaluation have been performed. Every MY37 LSTM/GRU metric currently in Section 8's
   table, and every figure/CI derived from it, is downstream of this leakage. The fix
   requires code + a re-run, not a documentation change — see Section 11, item 5, for the
   exact remediation steps and current status.

---

## 1. Section 4.2.1 — Formal Definition of the Martian Anomaly Label (Eqs. 4.1–4.4)

### Climatological expectation (Eq. 4.1) — μ_clim, σ_clim

> "...taken here as [**INSERT/CONFIRM**: the MCD day-to-day RMS variability field, or the
> standard deviation of training-period EMIRS residuals about the MCD mean, state which
> the code implements]."

**Neither, precisely — but for a real, principled reason.**
`data_sources/real_mcd_provider.py::RealMCDClimatologyProvider` interpolates a precomputed
grid built by `build_mcd_climatology_from_montabone.py`: μ_clim is the NaN-aware pooled
MEAN, and σ_clim the NaN-aware pooled STANDARD DEVIATION (ddof=1), of real
`dust_optical_depth` (CDOD normalised to 610 Pa) observations across all three downloaded
Mars Years (34-36) falling in the same 30°-wide solar-longitude bin, per native grid cell —
this combines real within-season (day-to-day) and real year-to-year (interannual)
variability into a single statistic. This is a documented ENGINEERING DECISION
(`build_mcd_climatology_from_montabone.py`'s own module docstring), grounded in real
satellite data rather than any parametric guess. `epsilon = 1.0e-6` is the variance floor
(`configs/anomaly_definition.yaml`). **Temperature is no longer part of Eq. 4.1 at all** —
no real climatology source exists for `surface_temperature_k` (see caveat 5).

`solar_longitude_ls` and `local_true_solar_time` — the two coordinates (with lat/lon)
`RealMCDClimatologyProvider.query()` needs — are **real, native per-footprint EMIRS
columns**, not computed: confirmed against real downloaded data that l3atm's `qls`/`qlt`
fields (the EMM EMIRS Data Product Guide's Appendix A documents these as `l_sub_s`/`ltime`;
every real downloaded file actually uses `qls`/`qlt` instead) hold exactly these
quantities — verified by cross-checking against l3emiss's independently-named `ltime`
column (exact row-for-row match) and by `qls`'s ~6.4° value on a 2021-02-20 file being
physically correct for a date just past EMIRS's documented vernal-equinox science start.

### Exceedance condition (Eq. 4.3)

> "[**CONFIRM**: whether the implementation uses the two-sided condition |δ|>τ as written,
> or the one-sided condition δ>τ; for dust opacity a one-sided condition is physically
> natural...]"

**One-sided, high only** — `z_dust > τ` (never triggered by anomalously low dust).
Source: `labeling/anomaly_labels.py`, `configs/anomaly_definition.yaml`
(`dust_exceedance: "one_sided_high"`).

### Which variable(s) the label is computed on

**Dust opacity alone.** `atmospheric_anomaly(c,s) = dust_anomaly(c,s)` — there is no OR
term; this pipeline was descoped to dust-only because no real climatology source exists
for `surface_temperature_k` (caveat 5). Source: `labeling/anomaly_labels.py::compute_exceedance`.

### Cell–sol aggregation (mean vs. median)

**Neither a plain mean nor a median: an inverse-variance (uncertainty-weighted) mean**
for the physical variables that matter, computed by `_weighted_mean_and_combined_uncertainty`
(`preprocessing/aggregation.py`, `configs/aggregation.yaml`). Only geometric/bookkeeping
fields (surface pressure, lat/lon, Ls, LTST) use a simple arithmetic mean. Median is never
used anywhere in the aggregation code.

### Pressure levels 𝒫

**19 levels, 0.25 to 4.75 pressure-scale-heights above the surface, in 0.25 steps** — the
real EMIRS L3 vertical grid (`configs/real_emirs_schema.yaml`), not the pressure-in-Pa grid
an earlier chapter draft assumed. Real l3atm files store this as a 19-value array per
retrieval (`temp`), collapsed at load time to its mean
(`data_sources/real_emirs_fits.py::collapse_temp_profile`, matching the real prior
dissertation's own documented method) into a single `atmospheric_temperature_mean_k`
column. **This profile is not currently consumed by either remaining task** (dust
regression uses `dust_optical_depth`; storm detection's label no longer references
temperature at all, caveat 5) — the real per-level values exist in principle but there is
no real per-level uncertainty field to weight them by, so they are not exploded into
per-pressure-level features (would require fabricating an uncertainty that doesn't exist
in the real product).

### τ, k, m (Eq. 4.4)

Source: `configs/anomaly_definition.yaml`.

| Parameter | Primary value | Sensitivity sweep |
|---|---|---|
| τ | **2.0** | [1.5, 2.0, 2.5] |
| k (persistence, sols) | **2** | [1, 2, 3] (`sensitivity` block) |
| m (min cells) | **3** | [1, 3, 5] (`sensitivity` block) |

**k=2/m=3 is the examiner-suggested stricter gate**, promoted from sensitivity to primary
2026-08-05 (caveat 6) — a real, working filter: of 18,776 `atmospheric_anomaly` cell-sols
and 4,436 candidate spatiotemporal episodes, only **203 episodes (containing 10,962
cell-sols) actually qualify** as `storm_episode` under k=2/m=3 — `storm_episode` is a
genuine, non-trivial SUBSET of `atmospheric_anomaly` now, not identical to it as it was
under the old k=1/m=1 placeholder.

### Connected-component adjacency

**8-neighbour, with longitude wraparound (no latitude/pole wraparound — polar-row cells
have only 5 neighbours), plus temporal linking**: two exceedance cell–sols are connected
if they are same-sol 8-neighbours, OR `|Δsol| = 1` AND (same cell OR 8-neighbours) — a
genuinely spatiotemporal connected-component construction, restricted to not cross a
`mars_year` boundary. Source: `labeling/connected_components.py`,
`configs/anomaly_definition.yaml` (`connectivity: 8`, `longitude_wraparound: true`). In
the real run: 4,436 candidate episodes were found, of which 2 were excluded for crossing a
chronological split boundary (`EpisodeCrossesSplitBoundaryError` machinery working as
designed, not triggered as an error since these 2 are legitimately dropped, not raised on).

### Storm taxonomy (descriptive only, never an ML target)

```
local:             area ≤ 20 cells         AND duration ≤ 3 sols
regional:          21 ≤ area ≤ 200 cells   AND duration ≤ 20 sols
planet_encircling: area ≥ 201 cells        AND duration ≥ 20 sols AND longitude_sectors_spanned ≥ 3 (of 3×120°)
```

Thresholds anchored loosely to Wang & Richardson (2015) / Zurek & Martin (1993) and the
5° grid geometry — flagged `RESEARCHER VERIFICATION REQUIRED` against a literature
area/duration table (`configs/storm_taxonomy.yaml`). Source: `labeling/storm_taxonomy.py`.
**Real classification breakdown of the 203 QUALIFYING `storm_episode`s** (k=2/m=3 primary,
`outputs/real/<run_id>/labels/episode_catalogue.csv`):

| Class | Count | % |
|---|---|---|
| local | 134 | 66.0% |
| regional | 44 | 21.7% |
| planet_encircling | 0 | 0.0% |
| unclassified (falls in none of the three bands) | 25 | 12.3% |

(For reference, across all 4,434 raw candidate episodes — most of which do NOT qualify
under k=2/m=3 — the breakdown is 4,330 local / 73 regional / 0 planet_encircling / 31
unclassified; the stricter gate concentrates on genuinely larger/longer episodes, hence
the much higher regional share among qualifiers.) Real size/duration spread among
qualifiers: median duration 3 sols / varies in area, up to 29 sols duration and 309 cells
peak area — real spatiotemporal dust-storm structure, not a gate artefact.

---

## 2. Section 4.2.2 — Separation of the Forecasting Tasks (Table 4.2)

**Two tasks, not three.** Temperature-anomaly forecasting (the former Task 2) has been
removed — no real climatology source exists for `surface_temperature_k` (caveat 5). Both
tasks have now actually been trained and evaluated against real data (Section 8).

**Terminology reconciliation, 2026-08-13**: the chapter's 2026-08-13 draft renames "Task
2: Storm detection" to "Task 2: Dust-anomaly-episode classification" throughout its prose
(Sections 4.1, 4.2, Table 4.1, Table 4.2), and Table 4.1's event-rarity row now reads
"Dust-anomaly-episode target prevalence" rather than "storm-episode." **This is a prose
renaming only — no code, config, file, or output-directory name changed.** The codebase
still uses `storm_detection`/`storm_episode`/`task_storm_detection.yaml` throughout
(confirmed unchanged). This document keeps the code identifiers below since they are
what every source reference actually points to; read "storm detection"/"storm_episode"
here as synonymous with the chapter's "dust-anomaly-episode classification"/
"dust-anomaly episode."

### Task 1 — Dust-opacity forecasting

| Marker | Real value |
|---|---|
| Task type | Regression (confirmed) |
| Target | `target_dust_optical_depth_h{h}(c,s) = dust_optical_depth(c,s+h)` — read from a genuinely observed future cell-sol only; a row with no exact match at `s+h` is dropped, never fabricated |
| Metrics | MAE, RMSE, MSE, R², bias, correlation |
| CRPS | **Not implemented anywhere in this repository** — confirmed via `modeling/metrics.py` (only `regression_metrics`/`classification_metrics` exist) |

Source: `configs/task_dust_regression.yaml`, `modeling/metrics.py`.

**Test-split N per horizon** (regression has no positive class, so these are the only
per-horizon counts that apply — read directly from
`metrics/dust_regression/climatology_h{1,3,5}_evaluation.json`'s `splits.test.n`, confirmed
identical across every model at a given horizon): **h=1: 13,682; h=3: 16,590; h=5:
22,227.** These grow with horizon because a longer horizon has more genuinely-observed
`(c, s+h)` future cell-sols available to match against within the fixed MY37 test window
— consistent with the target-construction rule above (rows with no exact match at `s+h`
are dropped, not the reverse).

### Task 2 — Storm detection ("dust-anomaly-episode classification" in the chapter's prose — see terminology note above)

| Marker | Real value |
|---|---|
| Task type | Binary classification (confirmed) |
| Target | `horizon_{h}sol` — 1 iff any qualifying `storm_episode` occurs within h sols (`storm_episode` is a genuine, gated SUBSET of `atmospheric_anomaly` at primary k=2/m=3 settings, Section 1 — not identical to it) |
| Metrics | tss, hss, pod, far, precision, recall, f1, brier, log_score, pr_auc, roc_auc |

Note: this pipeline's "FAR" is false-alarm **rate** (FP/(FP+TN)), not false-alarm
**ratio** — worth a footnote if Chapter 6 uses the term. Source:
`configs/task_storm_detection.yaml`, `modeling/metrics.py::classification_metrics`.

### Models trained per task

Both tasks share the same model progression, matching Section 4.6's narrative of
climatology baselines → classical ML → sequence deep learning:

- **Baselines (both tasks)**: `climatology`, `persistence`, `seasonal_mean`
  (`modeling/baselines.py`).
- **Classical models**: Task 1 (regression) → `RandomForestRegressor`,
  `GradientBoostingRegressor`. Task 2 (classification) → `LogisticRegression`,
  `RandomForestClassifier`, `GradientBoostingClassifier`, `KNeighborsClassifier`
  (`modeling/classical_models.py`).
- **Sequence models**: `nn.LSTM`/`nn.GRU` (`modeling/sequence_models.py::SequenceModel`,
  `hidden_size=16`, `num_layers=1`, `dropout=0.0`), window lengths {3, 5, 10} sols
  (10 = primary in every task config).
- **No Transformer, CNN, or attention-based model exists anywhere** in `src/martian`.

### Input variables / feature groups per task

Both tasks draw from the **same shared feature table** (`preprocessing/features.py`,
`configs/features.yaml`), built from `dust_optical_depth` (+ its z-score) only: trailing
lags {1,2,3} sols; trailing 5-sol mean/std/max/min; first difference; 5-sol slope; a
persistence-run counter; spatial 8-neighbour mean/max/exceedance-count; Ls/LTST
sine–cosine harmonics; and QC/interpolation indicator flags (`is_observed`,
`interpolation_flag`, `low_support_flag`, `observation_count`). In the real run this
produced **32 engineered feature columns**. `modeling/datasets.py` assembles each task's
own model input from this shared table plus current-state raw columns — a structural
guard prevents any task's feature table from containing another task's `target_*` column.

---

## 3. Section 4.3 — EMIRS and Mars Climate Database Data

### EMIRS L2 product version/ID and access date

**Downloaded 2026-08-04, via `download_emirs_sdc.py` against the EMM Science Data
Center's own REST API** (AWS Cognito auth + API Gateway, `instrument_id=emr`,
`data_level` = `l3atm`/`l3emiss`) — 7,445 real files, md5-verified. `processing_level:
L3` (**L3, not L2** — worth reconciling with the chapter's "EMIRS Level 2 retrieval
product" framing). **No single fixed version string applies**: real files span ~30
distinct format-version strings (e.g. l3atm `v02-01` through `v03-19`; l3emiss `v01-01`
through `v04-23`) — the SDC serves each file's own latest available version
(`latest=true`), not one pipeline-wide version. Real per-retrieval `md5checksum` and
`file_name`/`source_fits_file` are retained in provenance for exact traceability of any
individual value back to its source file. Coverage: **2021-02-20 to 2025-01-14** (see
caveat 1) — the SDC has released no EMIRS data beyond that date as of the download.

### MCD version, dust scenario, query method

See caveat 4 above for the full, current status: scenario and interpolation are confirmed
real values; `mcd_version` is a target, not a checked installation; the dust climatology
actually built and usable today comes from a different real source (Montabone et al.,
caveat 1), not from a live or precomputed `call_mcd` query.

### Instrument noise (NEΔT)

**Not found anywhere in this repository.** Remains an unresolved open item requiring
literature lookup against the real Edwards et al. (2021) paper.

### Per-retrieval uncertainty / quality flags in aggregation

**Uncertainty is used (weighting); quality flags are used (binary accept/reject).**
Quality control (`preprocessing/quality_control.py`, `configs/quality_control.yaml`) is an
8-rule, first-failing-rule accept/reject pipeline; retrievals passing it are then combined
per cell–sol via **inverse-variance weighting** using each retrieval's own
`<variable>_uncertainty` field (`preprocessing/aggregation.py`). **Real QC breakdown from
the completed run** (1,385,897 input retrievals):

| Rule | n_flagged | n_rejected (first-failing) |
|---|---|---|
| missing_required_variable | 0 | 0 |
| invalid_quality_flag | 0 | 0 |
| impossible_coordinate_or_timestamp | 531,368 | 531,368 |
| negative_optical_depth | 0 | 0 |
| physically_impossible_temperature | 257,704 | 254,506 |
| excessive_retrieval_uncertainty | 559,256 | 273,700 |
| unacceptable_viewing_geometry | 0 | 0 (disabled — no real field exists, see below) |
| duplicate_retrieval | 0 | 0 |
| **Total rejected** | | **1,059,574** (76.5%) |
| **Total accepted** | | **326,323** (23.5%) |

Two rules are notable, real findings worth a methods-section mention: (1) rule 7
(viewing-geometry exceedance) is **disabled for real data** — no `emission_angle_deg` or
equivalent field exists in the real EMIRS L3 product schema, confirmed against real
downloaded files, unlike an earlier speculative schema that assumed one; (2) the
`impossible_coordinate_or_timestamp` and `excessive_retrieval_uncertainty` rules together
account for nearly all rejections — a genuinely high real-data rejection rate (76.5%),
consistent with EMIRS's documented non-sun-synchronous, all-local-time science orbit
producing many geometrically marginal or high-noise retrievals alongside high-quality ones.

### Reference wavelength for dust optical depth

**Specified in the real schema**: `configs/real_emirs_schema.yaml` — dust column optical
depth at **1075 cm⁻¹**, water-ice column optical depth at **825 cm⁻¹**.

---

## 4. Section 4.3.1 / 4.3.2 — Temporal Coverage and Seasonal Stratification

> "By mid-2025, this corresponds to coverage spanning nearly two Martian Years,
> encompassing Martian Year (MY) 36 in full... Approx. 22,000 orbits / Approx. 28,000
> orbits / ~50,000 orbital observations total."

**Now measurable — real values from the completed run**, filtered to `mars_year_range:
[36, 37]` (`configs/run.yaml`), matching `configs/splits.yaml`'s train = MY36 sols
1–600 / val = MY36 sols 601–668 / test = MY37 sols 1–501 (1,169 total sols in range):

- **1,385,897 real retrievals** loaded within MY36–37 (post mars-year-range filter, pre-QC).
- **326,323 QC-accepted retrievals** (23.5% acceptance rate, see Section 3's breakdown).
- **142,849 aggregated cell-sols** — one row per (cell, mars_year, sol) with ≥1 accepted
  retrieval.

These are genuinely measured counts of real data, not orbit counts (EMIRS orbit numbers
are present in file metadata but not currently aggregated into a reported orbit count by
this pipeline — do not write "~22,000/~28,000/~50,000 orbits" into the chapter as a value
this codebase confirms; the pipeline reports retrieval and cell-sol counts, not orbit
counts).

### Coverage density (cells × sols)

With 2,592 grid cells (Section 6) and 1,169 possible sols in the configured MY36–37
range, the maximum possible cell-sol space is 2,592 × 1,169 = 3,030,048.
**142,849 / 3,030,048 = 4.71% of possible cell-sols were actually populated** by at least
one accepted retrieval — real, sparse spatiotemporal coverage, consistent with a single
non-imaging atmospheric sounder rather than a mapping instrument. Mean accepted retrievals
per populated cell-sol: 326,323 / 142,849 ≈ **2.28**.

### Dust optical depth seasonal figure (Figure 4.2)

`evaluation/regression_figures.py` produces per-model, per-horizon observed-vs-predicted
and residual diagnostic figures (with `.provenance.json` sidecars) from the real run — real
and working, and now populated with real data. These remain **model-diagnostic** plots,
not a dedicated dust-optical-depth-vs-Ls seasonal climatology figure; no script matching
Figure 4.2's described "seasonal dust cycle across MY 36–37" content exists in this
repository. That figure should still be treated as illustrative/schematic unless a
dedicated generation script is built.

---

## 5. Section 4.3 — Savitzky–Golay Filter (Figure 4.3)

**window_length = 7, polyorder = 2**, applied to the **sol-aggregated cell-sol time
series** of `dust_optical_depth` (grouped by `cell_id, mars_year`) — **not** raw
per-retrieval spectra. This codebase has no spectral/raw-radiance processing at all, so
Figure 4.3(c)'s current single-panel raw-spectrum illustration does not reflect what the
working pipeline actually filters. Two variants exist: **causal/trailing** (the only
variant permitted to reach a model as a feature, column suffix `_sg_causal`) and
**centred** (descriptive-only, structurally blocked from reaching a model by
`assert_feature_safe`, which raises `DescriptiveOnlyColumnError` if violated). Both
require an unbroken, sol-contiguous window or emit NaN. Source:
`configs/causal_filter.yaml`, `preprocessing/causal_filters.py`.

---

## 6. Section 4.4 — Gridding and Preprocessing (retention rules)

### Grid cell size

**Confirmed 5° × 5°.** `configs/grid.yaml`, `resolution_deg: 5.0` → 36 latitude bands ×
72 longitude bands = **2,592 cells**. A sensitivity sweep over [2.5, 5.0, 10.0]° also
exists in config.

### n_min (minimum accepted retrievals per cell-sol)

**Primary: n_min = 1** (`configs/aggregation.yaml`); sensitivity sweep [1, 3, 5]. There is
also `min_pressure_level_coverage_frac = 0.5` (primary; sweep [0.5, 0.7, 0.9]) — a cell-sol
failing either threshold is **flagged** (`low_support_flag`), not dropped outright. Real
note: since real EMIRS provides no per-pressure-level uncertainty (Section 1), no
`atm_temp_k_level_*` columns are populated for real data, so
`pressure_level_coverage_frac` is NaN for every real cell-sol —
`preprocessing/aggregation.py::compute_low_support_flag` treats NaN coverage as
"not applicable," not a failure, so `low_support_flag` in the real run is driven entirely
by the `n_min` (observation-count) threshold.

### Feature-window retained-sol fraction

**No such threshold exists anywhere in this codebase — a genuine gap, not an unreported
value.** Trailing-window statistics are computed from however many valid values happen to
fall in the window, with no minimum enforced.

### Polar latitude exclusion

**No polar exclusion exists.** Polar cells are retained, never merged or excluded.
Polar-row cells simply have 5 neighbours instead of 8 in the connected-component
adjacency (no wraparound over the pole itself).

### Interpolation policy and maximum gap length

**Causal, trailing-only (last-observation-carried-forward) fill**, never bidirectional,
never applied to labels. **Maximum gap bound: 2 sols** (primary; sensitivity sweep
[0, 1, 2, 3]) — `configs/missing_data.yaml`'s `max_trailing_fill_gap_sols`. A gap wider
than this bound is dropped from the output entirely, never fabricated. In the real run,
`apply_trailing_fill` produced **272,681 rows** in its output series (aggregated real
observations plus causally-carried-forward fill rows within the 2-sol bound). Source:
`preprocessing/missing_data.py`.

### Mean retrievals per cell per sol / percentage of cells populated

**Now measurable — see Section 4's "Coverage density" subsection**: 4.71% of possible
cell-sols populated; ≈2.28 mean accepted retrievals per populated cell-sol.

---

## 7. Section 4.4.1 — Label Validation Against Documented Events (Table 4.6)

**Reinstated 2026-08-05 — real, working, and executed.** A real, freely-licensed
documented dust-storm catalogue was located and integrated:
Guha, B. K., et al. (2024), "Seasonal and Diurnal Variations of Dust Storms in Martian
Year 36 Based on the EMM-EXI Database," J. Geophys. Res. Planets, 129, e2023JE008156
(dataset: doi:10.5281/zenodo.10275974, Zenodo, CC BY 4.0). Chosen after confirming the two
well-known Mars dust-storm catalogues both predate real EMIRS coverage — MDAD (Battalio &
Wang 2021) covers MY24-32, MDSSD covers MY24-33, both ending before MY36 (2021-02-20) —
neither can validate this pipeline's real labels. The Guha et al. catalogue covers exactly
MY36, from the same mission (Emirates Mars Mission) as the real EMIRS data, and reports
exactly the fields needed: storm onset/end time (from each detection's own EXI image
filename, a real UTC timestamp parsed the same way as EMIRS filenames), centroid lat/lon,
and areal extent. 248 raw per-image detections collapse to **98 distinct documented
storms** (`data_sources/real_documented_events.py`, grouped by the catalogue's own
`Storm no.` column) — both counts independently confirmed against the source paper's own
reported totals.

**MY36 only** — the catalogue has no MY37 coverage; both detected episodes and documented
events are restricted to `mars_year == 36` before matching, so real MY37 material is
neither penalised nor credited.

**Matching rule** (`configs/catalogue_validation.yaml`, RESEARCHER VERIFICATION REQUIRED
IN FULL — carried over unchanged from the original design, not independently confirmed
against this catalogue's own registration accuracy): temporal — the documented event's
`[onset_sol, end_sol]` window, expanded by `temporal_tolerance_sols=2` on each side, must
interval-overlap a qualifying episode's own window. Spatial — the documented event's
centroid cell, BFS-expanded by `spatial_region_radius_cells=1` grid-step, must contain at
least one of the episode's member cells.

**Real Table 4.6** (`labeling/catalogue_validation.py`, run as pipeline Step 9; corrected
post-hoc — see caveat 7's third bug — for a Mars-Year-boundary-straddling documented
storm that originally inflated these counts):

| Quantity | Value |
|---|---|
| Documented events (MY36) | 98 |
| Qualifying detected episodes (MY36, k=2/m=3) | 155 |
| Hits (documented events matched) | 11 |
| Misses | 87 |
| **POD** | **11.2%** |
| Matched detected episodes | 8 |
| False-positive candidates | 147 |
| **Episode precision** | **5.2%** |
| Mean temporal offset (matched pairs) | 4.4 sols |

**Honest reading, not smoothed over**: detection performance against this real,
independent catalogue is modest — POD 11.2%, precision 5.2%. This is a genuine finding
about the current z-score/persistence-gate storm detector's real-world detection rate
against an independently-sourced ground truth, not a validation-pipeline defect (the
matching logic itself has dedicated unit tests, `tests/test_catalogue_validation.py`).
Two real, structural reasons likely contribute, stated as hypotheses not confirmed facts:
(1) the documented catalogue is built from a visual-imagery detection method (EXI camera),
while this pipeline detects from a purely dust-optical-depth z-score threshold — the two
methods are sensitive to physically different storm signatures; (2) the ±2-sol/1-cell
matching tolerance is a carried-over placeholder, not confirmed against this catalogue's
own real registration accuracy (see Section 10, open item 3).

Outputs: `tables/table_4_6_independent_catalogue_validation.{csv,md}`,
`labels/catalogue_matched_pairs.csv`, `labels/catalogue_documented_events.csv`,
`labels/catalogue_episode_status.csv`.

### Table 4.6b — Blind dev/test tuning of the matching tolerance (2026-08-12, real, implemented and run)

**This resolves open item 4 below (previously: "not yet implemented, and this session
doesn't know what dev/test methodology was chosen earlier").** New module
`labeling/catalogue_tolerance_tuning.py` (20 tests, `tests/test_catalogue_tolerance_tuning.py`,
all passing), new config `configs/catalogue_tolerance_tuning.yaml`, wired into
`mars/run_all.py` as Step 9b, additive only — **the original Table 4.6 above is
untouched**, never overwritten. Design, per researcher decision recorded here:

- **Split**: the 98 documented MY36 events are split by fixed-seed random permutation
  (`seed=42`, `dev_fraction=0.5` → 49/49), not chronologically and not by storm-number
  parity, to avoid biasing either half toward a particular season or detection order.
- **Tolerance search grid**: `temporal_tolerance_sols ∈ {0,1,2,3,4}` ×
  `spatial_region_radius_cells ∈ {0,1,2,3}` (20 combinations) — bounded and centred on
  the original ±2-sol/1-cell placeholder, evaluated on the **DEV events only**.
- **Selection criterion**: F1 of episode detection (harmonic mean of dev-side POD and
  dev-side episode precision) — the researcher's explicit choice over raw POD (unsafe:
  trivially maximised by an infinitely loose tolerance) or a precision-floor/TSS-style
  alternative, made 2026-08-12.
- **Held-out evaluation**: the frozen, dev-selected tolerance is then applied ONCE to the
  49 TEST events, never touched during selection — `tests/test_catalogue_tolerance_tuning.py::test_tuning_never_uses_test_events_to_select_tolerance`
  constructs a case where dev- and test-side optimal tolerances would differ if leaked,
  and confirms the test event plays no role in selection.

**Real result, run against the canonical run's own saved
`labels/labeled_cell_sol.csv` / `labels/episode_catalogue.csv` and a fresh load of the
real Guha et al. documented catalogue (2026-08-12) — written to
`outputs/real/run_20260805T115705602822Z_seed42/tables/table_4_6b_*`:**

| Quantity | Value |
|---|---|
| Dev / test split | 49 / 49 documented events (seed=42) |
| Selected tolerance (dev-optimal, F1) | **temporal_tolerance_sols=4, spatial_region_radius_cells=3** — the LOOSEST combination in the searched grid |
| Dev-side F1 at selected tolerance | 0.0686 (POD 10.2%, precision 5.2%) |
| **Held-out TEST POD** | **16.3%** (8/49 hits) |
| **Held-out TEST episode precision** | **7.7%** (12/155 qualifying episodes matched) |
| Mean temporal offset (matched pairs, test) | 5.5 sols |

**Honest reading — the selected tolerance sits at the edge of the searched grid, not an
interior optimum, and a follow-up exploratory sweep (temporal up to 20 sols, spatial up
to 15 cells, dev side only, not part of the committed config) shows F1 STILL increasing
at the edge of that much wider range too** (F1=0.412 at temporal=20/spatial=15, still
rising). This is itself a real, informative finding, not a failed search: it means the
POD/precision gap between this pipeline's z-score detector and the Guha et al.
visible-imagery EXI catalogue is not primarily a registration/tolerance problem — no
tolerance short of physically meaningless values (a 15-cell radius exceeds most storms'
own footprint; a 20-sol window exceeds the qualifying episode median duration several
times over) closes it. This is real quantitative support for hypothesis (1) already
stated below (different physical signatures: thermal-IR opacity z-score vs. visible dust
veil), strengthening rather than merely repeating that hypothesis. **The committed
`configs/catalogue_tolerance_tuning.yaml` grid was deliberately left bounded to
physically meaningful values** rather than widened to chase a higher F1 at a
scientifically meaningless tolerance — this is a researcher judgement call, stated
explicitly so it is not mistaken for an incomplete search.

**Caution comparing Table 4.6 (11.2%/5.2%) against Table 4.6b's test-side numbers
(16.3%/7.7%): these are NOT the same evaluation.** Table 4.6 validates all 98 MY36 events
under the original ±2-sol/1-cell tolerance; Table 4.6b's test-side number validates only
49 (a random half) under a much looser, dev-selected tolerance. The apparent improvement
is expected and mostly attributable to the looser tolerance and smaller/different event
set, not evidence that the detector itself improved. Do not report Table 4.6b's numbers
as "Table 4.6 after tuning" without this caveat.

Outputs: `tables/table_4_6b_blind_tolerance_validation.{csv,md}`,
`tables/table_4_6b_dev_tolerance_search.csv` (full 20-row grid, both real, written
2026-08-12).

---

## 8. General — Splits, Model Results, Event Rarity

**SUPERSEDED 2026-08-13 — this section describes the 5 August run
(`run_20260805T115705602822Z_seed42`), kept for historical comparison only. The current
canonical numbers (split, prevalence, model performance) are in Section 13.** Cited here
because Section 13's comparison table is only meaningful against these original values,
not because these are still the numbers to report in Chapter 6.

### Train/test split

**Real, executed chronological split.** `configs/splits.yaml` /
`splitting/chronological_split.py`: train = MY36 sols 1–600; validation = MY36 sols
601–668; test = MY37 sols 1–501 (matching the chapter's MY36-full/MY37-partial, Ls
~0-270°, structure). The real run's chronological split of the full labeled/featured
dataset produced **81,355 train / 50,621 test / 10,697 unassigned** rows. The splitter
is leak-audited: it asserts no `storm_episode` spans a split boundary (2 candidate
episodes were found crossing a boundary and correctly excluded, Section 1) and fits all
standardisation statistics on the train split only. Both leakage checks
(`episode_boundary_check`, `feature_manifest_target_leakage_check`) **passed** in the real
run.

### Event rarity / anomaly-label prevalence (Table 4.1's row)

**Now measurable.** Of 142,849 aggregated (observed) cell-sols, **18,776 were labeled
`atmospheric_anomaly`** (= `dust_anomaly`, one-sided z>2.0 exceedance) — a real prevalence
of **13.14%**. Under the k=2/m=3 primary regime, **only 10,962 of those (7.67% of all
aggregated cell-sols) also qualify as `storm_episode`** — the stricter gate is a real,
non-trivial filter, not identical to `atmospheric_anomaly` (Section 1). 4,436 candidate
spatiotemporal episodes were identified via connected-component analysis (203 of them
qualifying under k=2/m=3), of which 2 candidates were excluded for crossing a
chronological split boundary.

**Exact per-horizon positive counts (2026-08-12, read directly from
`metrics/storm_detection/*_evaluation.json`'s `splits.<split>.metrics.{tp,fn,prevalence}`
fields — identical across every model for a given split/horizon, since these are
properties of the labels, not of any model; confirmed against both `climatology_h*` and
`persistence_h*`):**

| Horizon | Test positives (tp+fn) | Test N | Test prevalence | Train positives | Train N | Train prevalence |
|---|---|---|---|---|---|---|
| h=1 | **440** | 50,621 | **0.869%** | 2,793 | 81,355 | 3.433% |
| h=3 | **1,186** | 50,621 | **2.343%** | 5,826 | 81,355 | 7.161% |
| h=5 | **1,637** | 50,621 | **3.234%** | 7,250 | 81,355 | 8.912% |

**This corrects the chapter draft's placeholder numbers, which do not match the real
run.** The draft (Table 4.1/4.8, marked `[VERIFY]`) states test-split prevalence
≈0.71%/1.77%/2.47% and positive counts ≈360/896/1,251 — these were evidently carried
forward as estimates before the real run existed. The actual real-run values above are
higher at every horizon (0.87%/2.34%/3.23%; 440/1,186/1,637 positives) — real class
imbalance, not a modeling artefact, but a different real number than currently drafted.
Every occurrence of these figures in the chapter prose (Section 4.2.1, Table 4.1, Table
4.8) needs updating to the values in this table.

**142,673-vs-142,849 residual — fully reconciled, not an approximation.**
`data/aggregated_cell_sol.csv` has exactly 142,849 rows; `labels/labeled_cell_sol.csv`
(written immediately after the split-boundary-crossing exclusion in
`mars/run_all.py`, Step 8) has exactly 142,673 rows — a difference of exactly **176
rows**, and *only* 176 rows: these are precisely the member cell-sols of the 2 candidate
episodes excluded for crossing a chronological split boundary (`exclude_mask =
labeled_df["episode_id"].isin(crossing_ids)`, `mars/run_all.py` ~line 182). There is no
separate "small number of rows outside the labelled ranges" beyond this — the chapter
draft's `[VERIFY: exact]` hedge on that point can be removed. Confirmed by direct
split-manifest arithmetic: train (81,355) + val (0) + test (50,621) + unassigned
(10,697) = 142,673 = 142,849 − 176 exactly.

**Two further real findings surfaced while reconciling this residual (2026-08-12, not
previously documented) — see caveat 9 for the val/early-stopping consequence:**
- `splits/split_manifest.json` reports **`val.n_rows = 0`** — the configured MY36 sols
  601-668 validation window contains zero aggregated cell-sols in the real run.
- The aggregated data has a genuine, measured gap: MY36 cell-sols span sol 12-464 only
  (nothing sol 465-668) and MY37 cell-sols span sol 188-668 only (nothing sol 1-187) —
  confirmed directly against `data/aggregated_cell_sol.csv`. The chapter's test-split
  description ("MY37 sols 1-501") is technically the configured range, but no real test
  row exists before MY37 sol 188.

### Model performance (real results — Chapter 6 material, not a Chapter 4 methodology value)

**Real results**, from `outputs/real/run_20260805T115705602822Z_seed42/` (seed=42,
k=2/m=3 primary regime, correctly labeled throughout — see caveat 2). Headline numbers
(test split, 90% bootstrap CI in the source reports):

**Dust-opacity regression** (MAE, lower is better; R² near 0 means little better than the
seasonal-mean baseline — a real, modest-skill result, not a failure to report around).
Unaffected by the k=2/m=3 regime change (the regression target doesn't depend on the
persistence/area gate), so these numbers are unchanged from the first-pass run:

| Horizon | Best baseline (MAE) | Best classical (MAE) | Best sequence (MAE) |
|---|---|---|---|
| 1 sol | seasonal_mean: 0.150 | gradient_boosting: 0.139 | gru w=10: 0.141 |
| 3 sol | seasonal_mean: 0.139 | gradient_boosting: 0.119 | gru w=5: 0.127 |
| 5 sol | seasonal_mean: 0.134 | gradient_boosting: **0.105** (R²=0.122, best overall) | gru w=3: 0.118 (R²=0.086) |

**Storm detection** (TSS, higher is better; 0 = no skill beyond majority-class). These
numbers CHANGED materially from the first-pass run — the k=2/m=3 labels are stricter and
more structured, and skill is now clearly better at every horizon, especially by ROC-AUC:

| Horizon | Best baseline (TSS) | Best classical (TSS / ROC-AUC) | Best sequence (TSS) |
|---|---|---|---|
| 1 sol | persistence: 0.285 | random_forest: 0.015 (ROC-AUC **0.844**, best ROC-AUC) | gru w=10: 0.103 |
| 3 sol | persistence: **0.310** | random_forest: 0.158 (ROC-AUC 0.826) | gru w=10: 0.163 |
| 5 sol | persistence: 0.296 | random_forest: 0.162 (ROC-AUC 0.824) | gru w=10: **0.258** (ROC-AUC 0.854, best overall) |

Full per-model, per-horizon metrics with confidence intervals are in
`outputs/real/<run_id>/reports/final_report.md` and
`outputs/real/<run_id>/metrics/<task_name>/*.json`. Chapter 6 should state: this
methodology has been implemented, tested (369 passing unit/integration tests as of
2026-08-12, up from 331 at the 2026-08-05 run: +20 from `labeling/catalogue_tolerance_tuning.py`,
Section 7, and +26 from `evaluation/preprocessing_figures.py`, Section 12 — using
synthetic fixtures for testing purposes only), and now **run end-to-end against real
data under the examiner-suggested stricter label regime**, producing real,
CI-supported, genuinely stronger skill for storm detection at longer horizons under
k=2/m=3 than the earlier k=1/m=1 placeholder run showed (GRU window=10 TSS=0.258 at
h=5, up from 0.153; classical Random Forest ROC-AUC up to 0.85, up from 0.77) — real
detection skill improves when the label itself is the examiner's own preferred,
stricter, spatiotemporally-gated definition, not the maximally-permissive one — and
weaker but non-trivial skill for dust-opacity regression (gradient boosting is the
strongest model, with positive R² only at the 5-sol horizon). Note `persistence`
remains a genuinely strong baseline throughout (TSS 0.28-0.31) — real storm activity
has real day-to-day persistence, a physically sensible finding worth stating plainly
rather than only reporting where a learned model wins.

---

## 9. Model hyperparameters and metric formulas (real, exact — Methods-section prose)

Consolidated here (previously scattered across now-removed `planning/` build-log
documents — see the note at the very end of this document) because a Chapter 4 Methods
section needs the exact equations/hyperparameters, not just which models were used.
Verified current against `configs/task_dust_regression.yaml` /
`configs/task_storm_detection.yaml` / `modeling/metrics.py` /
`evaluation/confidence_intervals.py` as of this run.

### Classical model hyperparameters

| Model | Hyperparameters |
|---|---|
| `RandomForestRegressor` / `RandomForestClassifier` | `n_estimators=200, max_depth=8, min_samples_leaf=3` (classifier adds `class_weight=balanced`) |
| `GradientBoostingRegressor` / `GradientBoostingClassifier` | `n_estimators=200, max_depth=3, learning_rate=0.05` |
| `LogisticRegression` (storm_detection only) | `class_weight=balanced` |
| `KNeighborsClassifier` (storm_detection only) | `n_neighbors=15` |

### Sequence model hyperparameters

`nn.LSTM`/`nn.GRU` (`modeling/sequence_models.py::SequenceModel`): `hidden_size=16,
num_layers=1, dropout=0.0`; training: `epochs=30, learning_rate=0.001,
early_stopping_patience=5`; window lengths {3, 5, 10} sols (10 = configured primary). A
feasibility gate (`modeling/sequence_feasibility.py`) refuses to fabricate a result for a
window with too few constructable training sequences — reported as `INFEASIBLE`, never
silently skipped or defaulted.

**Optimiser, loss function, batch size — confirmed exactly from source (2026-08-12),
resolving the chapter draft's three `[VERIFY]` markers:**

| Chapter's `[VERIFY]` item | Confirmed real value | Source |
|---|---|---|
| Optimiser | **`torch.optim.Adam`**, `lr=training_config.get("learning_rate", 1e-3)` (i.e. 0.001, matching `configs/task_*.yaml`) | `modeling/sequence_models.py:135` |
| Loss (regression head) | **`nn.MSELoss()`** | `modeling/sequence_models.py:136` |
| Loss (classification head) | **`nn.BCEWithLogitsLoss()`** — a numerically-fused sigmoid+BCE operating on the raw logit, not a separate `nn.Sigmoid()` layer inside the model | `modeling/sequence_models.py:136` |
| "sigmoid + BCE" claim | **Confirmed, but implemented in two separate places, not one.** Training loss is `BCEWithLogitsLoss` on the raw logit (above). At evaluation, `modeling/train.py:326-328` applies a **manual** sigmoid — `1 / (1 + np.exp(-train_output))` / `...test_output)` — to the raw model output before it is written to any metrics/report file. So "one logit passed through a sigmoid to a probability" is accurate as a description of the end-to-end behaviour, just split across training-time (fused) and inference-time (manual numpy) code paths. | `modeling/sequence_models.py:136`; `modeling/train.py:322-328` |
| Batch size | **No minibatching exists anywhere in this code path — full-batch (whole-training-set) gradient descent.** `train_sequence_model` (`modeling/sequence_models.py:110-181`) calls `model(X_train_t)` once per epoch on the *entire* `X_train_seq` tensor; there is no `torch.utils.data.DataLoader`, `Dataset`, or `batch_size` key anywhere in `modeling/sequence_models.py`, `modeling/train.py`, `configs/task_dust_regression.yaml`, or `configs/task_storm_detection.yaml` (confirmed by grep across all four). "Batch size" is therefore not an unresolved unknown — it is the full training-sequence count for each task/horizon/window combination (varies by task; read `n_train` in the corresponding `metrics/<task>/*_provenance.json` for the exact count used at each window length). | `modeling/sequence_models.py:110-181`; absence confirmed by repo-wide grep |

**Related real finding, not one of the three `[VERIFY]` items but surfaced by reading the
same function (`train_sequence_model`) end-to-end: early stopping is evaluated against the
TEST split, not a held-out validation split, because `val` is empty in the real run — see
caveat 9 and Section 8 for the full explanation and consequence.**

### Metric formulas (`modeling/metrics.py`)

**Regression:** `MAE = mean(|ŷ−y|)`; `MSE = mean((ŷ−y)²)`; `RMSE = √MSE`; `bias =
mean(ŷ−y)`; `R² = 1 − SS_res/SS_tot`; `correlation` = Pearson r (NaN if either series is
constant).

**Classification** (default decision threshold 0.5): from confusion counts `TP, FP, TN, FN`:

```
POD (recall) = TP / (TP+FN)
FAR (false-alarm RATE, not ratio) = FP / (FP+TN)
precision    = TP / (TP+FP)
F1           = 2·precision·recall / (precision+recall)
TSS          = POD − FAR
HSS          = 2(TP·TN − FP·FN) / [ (TP+FN)(FN+TN) + (TP+FP)(FP+TN) ]
Brier        = mean( (p̂ − y)² )
log_score    = −mean( y·log(p̂) + (1−y)·log(1−p̂) ),  p̂ clipped to [1e-12, 1−1e-12]
```

`PR-AUC`/`ROC-AUC` via `sklearn.metrics.average_precision_score`/`roc_auc_score`; both
return NaN (not a fabricated value) if only one class is present in a given evaluation
slice — this genuinely happens for some baselines at some horizons in the real run
(e.g. `climatology`'s `precision=nan` throughout, since it never predicts a positive).

### Uncertainty procedure (bootstrap confidence intervals)

Block-by-cell (not i.i.d.-row) bootstrap, `evaluation/confidence_intervals.py`: for each
of `n_replicates=200` replicates, resample the SET of unique `cell_id`s with replacement
(not individual rows), reassemble all rows belonging to each sampled cell, recompute
every metric on the resampled set, then take the `(1−ci_level)/2` and `1−(1−ci_level)/2`
percentiles across replicates for a `ci_level=0.90` interval. `seed=42` (independent of
every per-task model seed). This block choice is an explicit ENGINEERING DECISION
(adapted from an originally-planned single-time-series temporal block to a `(cell × sol)`
panel, where blocking by cell is the natural equivalent) — RESEARCHER VERIFICATION
REQUIRED if a literal temporal block was intended instead.

### Seed strategy

Per-task model seeds: 42 (`dust_regression`), 44 (`storm_detection`) — never a shared
global default. Bootstrap CI seed: 42, set independently in `configs/evaluation.yaml`.
All recorded verbatim in every run's `manifests/run_manifest.json`'s `seed_strategy`
field. Full detail: `planning/SEED_STRATEGY.md`.

---

## 10. Open research items — require external literature lookup, not resolvable from this repository

Carried forward from a now-removed planning document (`planning/evidence_package/08_OPEN_ITEMS_RESEARCH_GUIDE.md`)
because these remain genuinely unresolved and are the kind of item an examiner is likely
to probe. Each is a real, external lookup or a researcher's own methodological decision —
neither this codebase nor this session can resolve them by more engineering work.

1. **EMIRS instrument specifications** (spectral range/resolution, footprint, NEΔT noise
   value): primary source is the EMIRS instrument description paper (Edwards et al.,
   Space Science Reviews, 2021) — read the instrument-performance table directly from the
   paper, not a secondary citation. Do not substitute a number from a different Mars
   infrared instrument (TES, THEMIS, MCS) even if plausible-looking.
2. **Citations to verify before final submission** — look each up directly on NASA ADS
   or the publishing journal, don't carry forward an unchecked volume/page number:
   - Cantor et al. — MOC observations of the 2001 global dust storm; check the actual
     publication year on the paper itself (commonly cited with a year later than the
     storm itself).
   - Edwards et al. (2021) — EMIRS instrument description; confirm volume/issue/page.
   - Millour et al. — multiple MCD version papers exist; confirm the one matching the
     MCD version actually cited.
   - Montabone et al. — confirm this is the dust-climatology paper this pipeline
     actually uses (real, already downloaded — see caveat 1), not a different
     Montabone et al. paper.
   - Wang & Richardson (2015) / Zurek & Martin (1993) — the storm-taxonomy citations
     (Section 1) — confirm exact citation details before final use.
3. **Independent dust-event catalogue — DONE for MY36 (Section 7); MY37 remains open.**
   The Guha et al. (2024) EMM/EXI database resolved this for MY36. No equivalent MY37
   catalogue has been located yet — if one becomes available (check for a MY37 companion
   dataset from the same EMM/EXI team, or an updated Montabone et al. gridded CDOD release
   extending into MY36-37), Table 4.6 could be extended to cover the real EMIRS pipeline's
   full MY36-37 range rather than MY36 only. Separately, the ±2-sol/1-cell matching-rule
   thresholds used for Table 4.6 remain RESEARCHER VERIFICATION REQUIRED — not confirmed
   against this specific catalogue's own real registration accuracy.
4. **Real EMIRS footprint scale** (100-300 km, used to justify the 5° grid resolution,
   Section 6) — not confirmed against the real instrument specification; check against
   item 1's source paper.
5. **Minimum retained-sol fraction for a feature window** — no literature or instrument
   source has this number; it would be a new engineering threshold (like
   `min_obs_per_cell_sol`) if the researcher wants one. Not currently enforced anywhere
   (Section 6).

---

## 11. Open engineering items — confirmed NOT implemented anywhere in this repository; require new code and a new run, not a documentation lookup

Checked exhaustively (2026-08-12, extended 2026-08-13) by grep across `evaluation/`,
`preprocessing/`, `labeling/`, `modeling/`, `mars/`, and `configs/` for every plausible
name/keyword. Unlike Section 10's items, these are not external facts to look up — they
are analyses/figures/corrections this codebase does not yet produce or apply. Listed
here, rather than left as bare prose disclosures in the chapter text, so it is unambiguous
that resolving them means writing and running new code (and, for items 5 and 6, a real
re-run), not finding a number in an existing log or adding a caveat paragraph.

1. **Strengthened bootstrap re-run (≥1,000–2,000 replicates, 95% CI, event- or
   spatiotemporal-block resampling).** `configs/evaluation.yaml` currently specifies
   `n_replicates: 200`, `ci_level: 0.90` (confirmed, Section 9's "Uncertainty procedure"
   subsection), and `evaluation/confidence_intervals.py::_resample_indices` blocks only by
   `cell_id` — there is no event-block or spatiotemporal-block resampling mode implemented
   (grep for `spatiotemporal`/`event_block` across `evaluation/` and `configs/` returns
   nothing). Every number currently in Section 8's model-performance table and in
   `outputs/real/<run_id>/metrics/**/*_evaluation.json` is under the 200-replicate/90%
   settings, not the strengthened settings the chapter's §4.7 text describes as the
   final-submission target. To produce this: change `configs/evaluation.yaml`'s bootstrap
   block, add a block-selection option to `_resample_indices` (currently only supports
   `block_ids=None` or a single block-id array — an event/episode-id block array would
   need to be threaded through from the labeling stage), and re-run evaluation for both
   tasks at all horizons.
2. **CDOD retrieval-uncertainty Monte-Carlo label-stability analysis (label-flip rate near
   τ=2.0).** Not implemented anywhere: no file in this repository perturbs an accepted
   CDOD retrieval by a draw from its own reported uncertainty and recomputes labels (grep
   for `monte.carlo`/`label.flip`/`perturb` across all `.py` files outside `tests/`
   returns nothing relevant). This would need a new script combining
   `preprocessing/aggregation.py`'s per-retrieval uncertainty field with
   `labeling/anomaly_labels.py`'s exceedance/labeling logic under repeated noisy draws,
   then re-evaluating representative models on the perturbed labels.
3. ~~Figures 4.5-4.8 (missingness map, gap-fill map, semivariogram, ACF/PACF,
   episode-centred composites).~~ **DONE 2026-08-12 — see Section 12.** Implemented as
   `evaluation/preprocessing_figures.py` (26 tests, `tests/test_preprocessing_figures.py`,
   passing) plus two runner scripts, and generated for real from the canonical run's own
   saved outputs. Along the way, this work found and fixed two real bugs (a NaN-poisoned
   ACF, and a Mars-Year-mixing bug in Figure 4.5's raw-retrieval extraction) and surfaced
   two new, previously undocumented real-data findings (a geolocation-sentinel cell, and a
   heavy-tailed climatological z-score distribution) — see Section 12 for the full
   writeup; none of this was knowable from documentation lookup alone.
4. ~~Blind dev/test re-validation of the Table 4.6 matching tolerance (±2-sol /
   1-cell).~~ **DONE 2026-08-12 — see Section 7's "Table 4.6b" subsection.** Implemented
   as `labeling/catalogue_tolerance_tuning.py` + `configs/catalogue_tolerance_tuning.yaml`
   + `mars/run_all.py` Step 9b + `tests/test_catalogue_tolerance_tuning.py` (20 tests,
   passing), and run for real against the canonical run's saved labels: 49/49 fixed-seed
   random dev/test split of the 98 documented events, F1-selection grid search on dev
   only (temporal ∈ [0,4] sols × spatial ∈ [0,3] cells), held-out test result POD=16.3%/
   precision=7.7% at the dev-selected tolerance (4 sols, 3 cells) — with the honest
   caveat that this is the loosest tolerance in the searched grid and F1 does not plateau
   even far outside it (real evidence for a detection-methodology mismatch, not a
   registration/tolerance problem — see Section 7). The two remaining design choices this
   session made independently (fixed-seed random 50/50 split; F1 as the selection
   criterion, per the researcher's explicit 2026-08-12 choice over raw POD or a
   precision-floor/TSS-style alternative) are recorded in
   `configs/catalogue_tolerance_tuning.yaml`'s own comments for future audit.

5. ~~CRITICAL — LSTM/GRU retraining under a genuine validation split (fixes caveat 9's
   test-set leakage).~~ **FULLY DONE 2026-08-13, including the actual retrain — see
   Section 13.** Design + code fix, and now the retrain itself: run
   `run_20260813T195013457439Z_seed42` trained every baseline/classical/sequence model
   under the corrected split with val-only early stopping. Sequence-model storm-detection
   TSS dropped substantially at every horizon relative to the leaked 5 August numbers
   (e.g. GRU w=10 at h=5: 0.258 → 0.154) — the expected signature of a genuine leakage
   fix, not noise (see Section 13's full before/after table and the caveat that this
   correction was executed as a full re-run, not attempted piecemeal).

   **(a) Split design, chosen and verified against real data.** `configs/splits.yaml`'s
   `val_sol_range` changed from `[601, 668]` (real MY36 coverage never reaches past sol
   464, so this was always empty) to **`[432, 668]`**, with `train_sol_range` correspondingly
   `[1, 431]`. The boundary (sol 432) was found by scanning every candidate boundary from
   370-459 against the real canonical run's own `labels/episode_catalogue.csv` (full
   catalogue, qualifying and non-qualifying episodes both — the stricter, more
   conservative reading of "keep complete anomaly episodes within one partition") using
   `splitting.chronological_split.find_episodes_crossing_split_boundary`, keeping only
   zero-crossing boundaries, then picking the one giving a conventional ~80/20 train/val
   split with strong positive coverage at every horizon. Real counts at the chosen
   boundary (`labels/labeled_cell_sol.csv`, MY36 only):

   | Split | N (cell-sols) | h=1 positives | h=3 positives | h=5 positives | storm_episode |
   |---|---|---|---|---|---|
   | Train (sol 1-431) | 66,987 | — | — | — | — |
   | Val (sol 432-668) | 14,368 | 67 | 203 | 291 | 188 |

   Zero anomaly episodes (qualifying or not) cross the boundary. Sequence-window
   feasibility was checked directly against `data/trailing_filled_cell_sol.csv` restricted
   to the val range: even at the primary window length of 10 sols, 687 of 946 val cells
   have a sufficient consecutive run (`modeling/sequence_feasibility.py::run_feasibility_report`).
   Val is strictly later in sol than train (temporal causality) and entirely within MY36
   (MY37/test is never touched). Nearby alternatives were also computed and are viable if
   a different train/val trade-off is preferred: boundary 415 (27.0% val, h1=73 positives)
   and boundary 424 (22.2% val, h1=70 positives) both also have zero episode crossings and
   full window=10 feasibility.

   **(b) Real code bug found and fixed while implementing this**, distinct from the split
   itself: `modeling/train.py`'s sequence-model block built only two sequence datasets
   (`train`, `test`) and passed the TEST one into `train_sequence_model`'s early-stopping
   argument — so even after the val split is populated, the ORIGINAL code would have kept
   leaking test into early stopping, silently ignoring the new non-empty val. Fixed:
   the block now builds `train`/`val`/`test` sequence datasets separately, requires all
   three to be non-empty or the window is marked `infeasible` (never silently trained on
   a substitute), passes only `val` into early stopping, and reports/checkpoints/evaluates
   all three splits (`n_train_windows`/`n_val_windows`/`n_test_windows`,
   `TrainingRecord.n_val` now genuinely the val count, `eval_result["splits"]` now has a
   real `"val"` entry). Regression-tested:
   `tests/test_train.py::test_run_task_sequence_models_infeasible_when_val_empty` (proves
   training refuses to substitute test for an empty val) and
   `::test_run_task_sequence_models_feasible_window` (extended to assert genuine
   `n_val_windows > 0` and that provenance's `n_val` matches it, not test's count).
   `tests/test_chronological_split.py::test_val_split_boundary_is_non_empty_and_causally_after_train`
   guards the config itself. Full suite: 372/372 passing at this point in the pass (before
   the sentinel-row and window-completeness tests below brought the total to 378/378).

   **(c) DONE**: the actual retrain, executed together with items 6-7 below in one pass
   (deliberately not attempted three times separately) — `run_20260813T195013457439Z_seed42`,
   real val split 66,826 train / 14,355 val / 50,566 test rows, val positive counts
   (67/203/291 at h=1/3/5) matching this design's own predicted counts exactly. Full
   before/after model-performance comparison: Section 13.
6. ~~Sentinel-coordinate (0°, 0°) rows are excluded from representative-cell figures ONLY
   — NOT from the pipeline itself.~~ **DONE 2026-08-13 — fixed at ingestion, real revised
   counts measured, and the underlying issue turned out to be ~68x larger than the
   Section-12 estimate suggested.**

   **Real scale, not 0.33%.** Loading the full real archive and inspecting the sentinel
   rows directly (not just their aggregated footprint) found **308,186 of 1,385,897 raw
   retrievals (22.2%)** carry this pattern — not the 466 aggregated cell-sols (0.33%)
   Section 12 measured, which was only how many of them survive all the way through QC and
   aggregation. Every l3atm-side physical field is exactly 0.0 simultaneously — `chi2temp`,
   `chi2a` (real per-retrieval goodness-of-fit statistics), `dust_optical_depth`,
   `atmospheric_temperature_mean_k`, `latitude`, `longitude`, `local_true_solar_time`,
   `water_ice_optical_depth`, and five more — while the l3emiss-side fields
   (`surface_temperature_k`, `kinetic_temp_wave`) remain real and varying. Every row has a
   genuine, parseable, unique `utc_timestamp` (not caught by the existing unparseable-UTC
   filter) and (0,0) is within any sane QC coordinate range (not caught by
   `impossible_coordinate_or_timestamp` either).

   **Most reliable metadata-based identifier, per the task's own instruction**: not
   `latitude == 0 & longitude == 0` (a physical coordinate that is not itself impossible —
   Mars genuinely has a point at 0°N, 0°E), but **`chi2temp == 0.0 AND chi2a == 0.0`** — a
   real chi-square goodness-of-fit statistic is essentially never exactly zero for an
   authentic fit to noisy real spectra, and `configs/real_emirs_schema.yaml`'s own header
   had already flagged these two fields as "usable for a genuine QC rule... pending" before
   this fix. Implemented in `data_sources/real_emirs_fits.py::load_real_emirs_directory`,
   immediately after the existing unparseable-UTC filter, scoped to l3atm only (l3emiss has
   no chi2 columns). Four new tests added
   (`tests/test_real_emirs_fits.py::test_load_real_emirs_directory_drops_chi2_null_sentinel_rows`
   and three siblings covering the AND-not-OR logic and l3emiss non-applicability).

   **Real revised counts** (re-ran ingestion → QC → gridding → aggregation with the fix;
   not a full pipeline re-run, no model touched):

   | Quantity | Before (2026-08-05 run) | After (2026-08-13 fix) |
   |---|---|---|
   | Raw retrievals loaded | 1,385,897 | 1,077,710 (−308,187) |
   | Rows with lat=lon=0 remaining | 308,186 | **0** |
   | QC-accepted | 326,323 (23.5%) | 268,673 (24.9%) |
   | `impossible_coordinate_or_timestamp` rejections | 531,368 | 531,368 (**unchanged** — confirms this rule never caught these rows) |
   | `physically_impossible_temperature` rejections | 254,506 | 3,969 (**−250,537** — confirms most sentinel rows were previously being rejected via temperature, not coordinate, since `atmospheric_temperature_mean_k=0` fails that rule) |
   | `excessive_retrieval_uncertainty` rejections | 273,700 | 273,700 (unchanged) |
   | Aggregated cell-sols | 142,849 | 142,608 (−241) |
   | Mean accepted retrievals per populated cell-sol | ≈2.28 | ≈1.88 |

   **The arithmetic reconciles exactly**: 326,323 − 268,673 = 57,650 fewer accepted
   retrievals; 250,537 of the 308,187 removed rows were already being rejected (via
   temperature), leaving 308,187 − 250,537 = 57,650 that had previously been slipping
   through ALL 8 QC rules and counted as genuine "accepted" data — matching exactly. That
   57,650/326,323 ≈ 17.7% of the OLD accepted pool was silently contaminated by null rows,
   a materially larger problem than the 0.33% aggregated-footprint figure implied.

   **Real, reassuring finding**: cell 1296 (the grid cell whose 5°×5° bounds contain
   geographic (0°,0°)) is **not** a pure artefact — it is a genuine equatorial/
   prime-meridian cell with real observations of its own. After the fix it retains **225**
   real aggregated cell-sols (down from the old, contaminated 466 = 225 genuine + 241
   null), landing squarely in the normal range of other well-observed cells (249-259 —
   see Section 12's representative-cell table) rather than standing out as a ~2x outlier.
   Cell 1393 (Figures 4.5-4.8's representative cell) is entirely unaffected (its own real
   lat/lon never coincides with the null-row signature), so no figure needs regenerating
   for this fix alone — only the headline pipeline counts above do, in the full re-run
   (item 5(c)).
7. ~~Missing-data / gap-fill sensitivity comparison.~~ **DONE 2026-08-13 — real 18-combination
   sweep executed, primary configuration selected and applied to `configs/missing_data.yaml`
   / `configs/features.yaml`.**

   **Design, confirmed with the researcher before running** (a real judgement call, not an
   engineering default): proxy model = gradient boosting (already in the pipeline's own
   model progression, fast enough to make an 18-combination sweep tractable — full LSTM/GRU
   across every combination would have meant retraining sequence models 18× before the
   real final retrain even started); selection metric = val TSS (storm detection, maximise)
   and val MAE (dust regression, minimise), each averaged across horizons 1/3/5, combined
   by summed rank across both tasks. Grid: `n_min` ∈ {1, 2} × `max_trailing_fill_gap_sols`
   ∈ {0, 1, 2} × `min_window_completeness_frac` ∈ {none, 0.6, 0.8} = 18 combinations, matching
   the task's own exact specification (not the codebase's broader default sweeps).

   **Real run**: `run_missing_data_sensitivity_sweep.py`, one ingestion+QC+gridding pass
   reused across all 18 combinations (each combination re-running aggregation → causal
   fill → Savitzky-Golay → climatology z-score/labeling → feature construction → chronological
   split → gradient-boosting fit on train → evaluation on val). **Total wall time: 16,404s ≈
   4.6 hours** — far longer than initially estimated, since each combination's feature
   construction is itself comparable in cost to the original run's own ~4-minute feature-build
   step, not the few-seconds-per-fit cost that was estimated for the proxy-model piece alone.
   Full 18-row result table: `outputs/real/missing_data_sensitivity_sweep_20260813.csv`.

   **Honest finding: the `n_min` axis was a no-op.** Every `n_min=1` row's metrics are
   IDENTICAL, to the last decimal, to its `n_min=2` twin, across all 9 gap×completeness
   combinations. Root cause: `aggregate_by_cell_sol`'s `min_obs_per_cell_sol` parameter only
   sets `low_support_flag` on a cell-sol that fails the threshold — it never actually excludes
   the row (already documented, Section 6: "a cell-sol failing either threshold is FLAGGED...
   not dropped outright"). This should have been anticipated before running the sweep; it
   is reported here rather than glossed over. The `max_trailing_fill_gap_sols` and
   `min_window_completeness_frac` axes DID vary the data and DID produce genuinely different
   metrics (e.g. gap=0/completeness=none: val MAE 0.1138/val TSS 0.0456 vs. the selected
   gap=1/completeness=0.6: val MAE 0.1125/val TSS 0.0530), so the selection itself rests on
   real, varying evidence for 2 of its 3 nominal axes.

   **Selected primary configuration** (lowest combined rank sum, train/val evidence only):

   | Parameter | Old primary | New primary (2026-08-13) |
   |---|---|---|
   | `max_trailing_fill_gap_sols` (`configs/missing_data.yaml`) | 2 | **1** |
   | `min_window_completeness_frac` (`configs/features.yaml`) | n/a (didn't exist) | **0.6** |
   | `min_obs_per_cell_sol` (`configs/aggregation.yaml`) | 1 | 1 (unchanged — proven not to matter) |

   val_mae_mean = 0.1125 (rank 4/18), val_tss_mean = 0.0530 (rank 6/18) — not the single best
   value on either metric alone, but the best COMBINED rank, which is what was specified.
   Both config files updated with the real values and a note pointing back to this sweep;
   full suite re-verified at 378/378 passing after the change.
8. **Formal preprocessing-validation acceptance thresholds — do not exist.** Confirmed:
   no code anywhere computes or checks a maximum climatological bias, a minimum
   field-level correlation, a distributional-divergence bound, or a maximum
   gridding-artefact rate. The completed run's preprocessing validation (Section 4.4's
   qualitative/quantitative checks) was diagnostic-only, with no pass/fail criterion
   applied. Required: define the four threshold values (each is a methodological choice,
   not something derivable from code) and re-check the completed run's preprocessing
   outputs against them.
9. **Figures 4.3 and 4.4 remain SCHEMATIC, not empirical — distinct from Figures 4.5-4.8,
   which item 3 above resolved.** The chapter's own captions still say so explicitly:
   Figure 4.3 ("[SCHEMATIC — not empirical evidence]... An empirical seasonal-coverage
   figure derived from the analysed EMIRS data is deferred to the corrected run") and
   Figure 4.4 ("[SCHEMATIC] Panels illustrate the processing steps using representative
   (not empirical) data; an empirical version showing actual EMIRS sampling, Montabone
   alignment, and the causal-filter before/after comparison for a named cell is deferred to
   the corrected run"). `evaluation/preprocessing_figures.py` (Section 12) does not cover
   these two — it was scoped to 4.5-4.8 only. A real Figure 4.3 (seasonal CDOD cycle across
   MY36-37 by Ls, from `data/aggregated_cell_sol.csv`) and a real Figure 4.4 (raw sampling
   times, climatology alignment, and the causal Savitzky-Golay before/after comparison for
   one representative cell, from data already extracted for Figure 4.5 plus
   `data/filled_with_causal_savgol.csv`) are both buildable with real data already on disk
   — neither requires a new pipeline run — but neither has been built.

Items 1, 2, and 7-9 remain real, well-scoped follow-on engineering tasks. Items 5 and 6
are the two the chapter's own text now treats as blocking Chapter 6's validity, not merely
descriptive gaps — flagging that severity distinction accurately is itself a useful output
of this document's 2026-08-13 pass.

---

## 12. Figures 4.5-4.8 — generated from real data, 2026-08-12

**Regenerated 2026-08-13 against the new canonical run** (`run_20260813T195013457439Z_seed42`
— Section 13) using the same scripts described below, updated to point at the new run
directory. Same representative cell (1393, unaffected by any of the three fixes — its own
814 raw retrievals and 526/288 accept/reject split are bit-identical to the original
generation). Visually confirmed consistent: the lag-9 pooled-ACF spike and the
`mid_south`/Ls-Q1 semivariogram outlier (Finding 2, below) both reproduce under the new
run, confirming they are real, repeatable structure, not an artefact of the specific
5 August data. The gap-fill map (Figure 4.6b) visibly thins as expected under the
sensitivity-selected `max_trailing_fill_gap_sols=1` (down from 2). Figure 4.5/4.6/4.8's
train-range x-axis now correctly stops at sol ~431 (the new split boundary) rather than
464. New file paths: `outputs/real/run_20260813T195013457439Z_seed42/figures/preprocessing/`.
The write-up below (representative-cell selection, Findings 1-2, the two bugs found and
fixed) describes the original 2026-08-12 generation and remains accurate; only the
specific run directory and the numbers baked into filenames/captions changed.

New module `evaluation/preprocessing_figures.py` (data-computation helpers, unit-tested
independently of matplotlib — 26 tests, `tests/test_preprocessing_figures.py`, all
passing), new config `configs/preprocessing_figures.yaml`, and two runner scripts:
`generate_figure_4_5_raw_data.py` (one-off, ~15 min, loads the full real EMIRS archive to
extract one representative cell's raw pre-QC retrievals — not saved by the main pipeline
run, since saving all 1.38M raw retrievals for every run would be excessive) and
`generate_chapter4_preprocessing_figures.py` (the actual figure generation, seconds to
run once the raw-data CSV exists). All four figures were generated for real from
`outputs/real/run_20260805T115705602822Z_seed42`'s own saved outputs and written to that
run's `figures/preprocessing/` directory with full provenance (`.provenance.json` +
source `.csv` sidecar per figure, via the existing
`evaluation.figure_provenance.save_figure_with_provenance` machinery, using the chapter's
own fixed figure numbers 4.5-4.8 rather than the separate auto-incrementing registry that
`figure_provenance.py` uses for per-model diagnostic figures — those are a different
figure sequence starting at 4.8 for a different purpose; a deliberate choice, stated here
to avoid confusion).

### Representative cell

**Cell 1393** (lat=8.34°, lon=125.30°) — the most-observed cell in the MY36 training
split among cells with physically plausible retrieval density and CDOD values. Selection
rule and exclusion list: `configs/preprocessing_figures.yaml`.

### Finding 1 (new, 2026-08-12): a geolocation-sentinel/full-disk cell

While selecting the representative cell, the single most-observed cell in the entire
training split was found NOT to be physically representative: **cell 1296 (lat=0.0,
lon=0.0 exactly)** has 466 aggregated cell-sols across the full run — far more than any
genuine cell (next-highest has 259) — but a mean `dust_optical_depth` of ~2×10⁻¹¹
(indistinguishable from zero) and ~125 raw retrievals per cell-sol (vs ~2.3 average
everywhere else). This is consistent with a real, distinct EMIRS observation mode (e.g.
full-disk or calibration frames, which have no single nadir ground footprint) being
assigned a placeholder (0,0) geolocation that the `impossible_coordinate_or_timestamp` QC
rule's simple range check does not catch, since (0,0) is a valid coordinate pair on its
own. This affects 466/142,849 (0.33%) of aggregated cell-sols in the canonical run — small
in proportion, but the single densest "cell" in the entire grid, so it would silently
dominate any per-cell or representative-cell analysis if not excluded. **Not corrected in
the canonical run's own headline statistics** (out of scope for a figure-generation task;
those statistics are unaffected in aggregate at the 0.33% level) — flagged here, and
excluded via `configs/preprocessing_figures.yaml`'s `excluded_cell_ids`, so it does not
silently contaminate Figures 4.5/4.8's "representative cell." A dedicated QC rule closing
this gap (e.g. flagging exact (0,0) or a documented full-disk/calibration flag if the real
product carries one) is a follow-up item, not executed here.

### Finding 2 (new, 2026-08-12): heavy-tailed climatological z-scores

Building Figure 4.7 surfaced a second, more consequential real finding: `dust_optical_depth_z`
(Eq. 4.2's climatological z-score) is heavily right-skewed with extreme outliers —
**1,323 of 142,464 labeled rows (0.93%) have |z|>50**, and the maximum is **z≈2026**. This
is driven by real (cell, 30°-Ls-bin) climatology cells with a small but non-degenerate
`sigma_clim` (e.g. 0.005-0.02) — NOT caught by Eq. 4.2's epsilon=1×10⁻⁶ floor, which only
guards against near-zero sigma, not merely small sigma — combined with real observed CDOD
substantially exceeding that narrow expected range. **This does not change any
`atmospheric_anomaly`/`storm_episode` label**: the exceedance test is `z > τ=2.0`, a
threshold crossing unaffected by how far past 2.0 z lands, so Section 1's/Section 8's
prevalence and episode counts are unaffected. It DOES mean any statistic computed on the
raw MAGNITUDE of z is dominated by this small tail:
- Figure 4.7(a)'s episode-centred composite now plots **median and IQR**, not mean/std
  (originally implemented with mean/std, which produced climatological z-scores of ±75-100
  in the shaded band — an artifact of the tail, not genuine episode structure; caught and
  fixed before this document was written, not left in).
- Figure 4.7(c)'s semivariogram uses the **median of squared differences** per distance
  bin (a standard, simpler robust alternative to the Cressie-Hawkins estimator), not the
  mean, for the same reason.
- Even under the median estimator, the `mid_south, Ls Q1` curve still shows an elevated
  semivariance (≈588) in its most distant bin (5,000-8,000 km, n=75 pairs) — a genuine
  signal, not a display artifact, that this specific latitude-band/season combination has
  a real concentration of small-sigma_clim cells (several of the individual |z|>50 rows
  identified fall in exactly this region/season). **Not further investigated** (would
  require auditing the underlying Montabone climatology's per-bin sample sizes, out of
  scope for a figure-generation task) — flagged as a genuine open question about Eq.
  4.2's climatology construction, distinct from (but related to) caveat 9.

### Bugs found and fixed during this figure's own construction (2026-08-12)

1. **Figure 4.5 initially mixed MY36 and MY37 raw retrievals on one shared sol axis.**
   `generate_figure_4_5_raw_data.py` extracts one cell's raw retrievals without an
   upstream mars_year filter; since sol numbering resets every Mars Year, plotting both
   years together silently overlaid unrelated observations at the same x-position. Fixed
   by having `figure_4_5_raw_qc_comparison` filter to a single `mars_year` internally
   (not left to caller discipline).
2. **`compute_pooled_lag_autocorrelation` returned NaN for nearly every lag even though
   tens of thousands of valid pairs existed.** Root cause: `np.mean(products)` was called
   on a list that could contain NaN products (from real rows with an undefined z-score, a
   separate, ordinary and expected ~122-row occurrence in the 81,355-row training split) —
   `np.mean` propagates a single NaN to the whole result. Direct verification against the
   real data: 33,196 genuine lag-1 (cell, mars_year, sol)/(cell, mars_year, sol+1) pairs
   exist in the training split, none of them originally visible in the ACF. Fixed by
   skipping NaN endpoints explicitly before appending to the per-lag product list. The
   corrected, real pooled ACF/PACF shows a genuine, interesting structural feature: a
   **spike at lag=9 sols (ACF≈0.14)** amid otherwise near-zero correlation at every other
   lag from 1-20 — plausibly related to Hope's ~55-hour (≈2.23-sol) orbital revisit period
   (9 sols ≈ 4 orbits), though this is stated as a hypothesis, not confirmed, and would
   need cross-checking against real orbit ephemeris to verify.

### Outputs

| Figure | File | Representative content |
|---|---|---|
| 4.5 | `figures/preprocessing/figure_4_5_raw_qc_comparison.png` | Raw per-retrieval CDOD by QC outcome (cell 1393, MY36, n=814 raw retrievals, 526 accepted/288 rejected); QC'd + trailing-filled cell-sol series; retained/rejected observation count per sol |
| 4.6 | `figures/preprocessing/figure_4_6_missingness_gapfill_distribution.png` | Cell-sol missingness map and trailing-fill map (latitude band containing cell 1393, all 72 longitude cells); pre/post-fill CDOD distribution (n=81,355 pre-fill / 152,264 post-fill) vs. the Montabone climatology mean |
| 4.7 | `figures/preprocessing/figure_4_7_spatiotemporal_structure.png` | Qualifying storm_episode-centred composite (n=203 episodes, median/IQR); pooled panel ACF/PACF (lags 0-20); CDOD z-score semivariogram by latitude band × Ls quartile (median estimator) |
| 4.8 | `figures/preprocessing/figure_4_8_seasonal_series.png` | Sol-aggregated CDOD, cell 1393, MY36, with real Ls=90°/180° crossing markers (Ls=0°/270°/360° fall outside this cell's observed sol range) |

Each PNG has a companion `.csv` (source data) and `.provenance.json` (figure number,
caption, units, n, split, script, config paths) alongside it, plus
`figure_4_7_episode_composite.csv` and `figure_4_7_acf_pacf.csv` (the full underlying
series for panels (a) and (b), not just the plotted summary).

**Known minor cosmetic issue, not fixed**: `evaluation.figure_provenance`'s stamped
caption block (shared infrastructure, used by every figure in this repository) can
visually crowd a figure's own x-axis label at the bottom edge when the caption's
`data_source`/`caption` strings are long, as here — legible but not polished. Left as-is
rather than modifying shared, already-tested infrastructure for a cosmetic-only reason
outside this task's scope.

---

## 13. The 2026-08-13 corrected re-run — full pipeline retrained under all four fixes

This section is the authoritative source for Chapter 6 going forward. It documents the
complete corrected re-run requested in response to the examiner's four critical problems
(caveat 9 / Section 11 items 5-7, plus the missing-data sensitivity comparison), what
changed in the code and configuration to produce it, and a full before/after comparison
against the superseded 5 August run. **New canonical run:
`outputs/real/run_20260813T195013457439Z_seed42/`.**

### What was fixed, in order, before this run

1. **Train/validation split redesigned** (`configs/splits.yaml`): val moved from
   `[601, 668]` (always empty) to `[432, 668]`, chosen by scanning every boundary 370-459
   for zero anomaly-episode crossings and picking the earliest with a conventional ~82/18
   split and full window=10 sequence feasibility. See caveat 9 and Section 11 item 5 for
   the full design and justification, done and confirmed BEFORE this run was started.
2. **`modeling/train.py` code fix**: sequence models now build genuine train/val/test
   sequence datasets and early-stop on val only, refusing to train if any of the three is
   empty (previously: only train/test were built, and test was silently used for early
   stopping regardless of val's state). Covered by new regression tests
   (`tests/test_train.py`).
3. **Sentinel-row ingestion fix** (`data_sources/real_emirs_fits.py`): rows where
   `chi2temp == 0.0 AND chi2a == 0.0` (308,186 of 1,385,897 raw retrievals, 22.2% of the
   archive — far larger than the 0.33% aggregated-footprint estimate this document
   originally carried) are now excluded at ingestion, before QC. See Section 11 item 6 for
   the full metadata analysis and the exact QC-count reconciliation.
4. **Missing-data sensitivity comparison executed** (`run_missing_data_sensitivity_sweep.py`,
   18 real combinations, gradient-boosting proxy model, train/val evidence only — see
   Section 11 item 7): selected `max_trailing_fill_gap_sols=1` (was 2) and
   `min_window_completeness_frac=0.6` (a mechanism that didn't exist before this pass);
   `min_obs_per_cell_sol` left at 1 (proven a no-op under current pipeline semantics).

Full test suite re-verified at every step; 378/378 passing immediately before this run was
launched.

### Pipeline-level counts: before vs. after

| Quantity | 5 Aug run | 13 Aug run | Change |
|---|---|---|---|
| Raw retrievals loaded | 1,385,897 | 1,077,710 | −308,187 (sentinel fix) |
| QC-accepted | 326,323 (23.5%) | 268,673 (24.9%) | −57,650 |
| Aggregated cell-sols | 142,849 | 142,608 | −241 |
| Trailing-filled feature rows | 272,681 | 232,702 | −39,979 (gap 2→1 sols) |
| `atmospheric_anomaly` cell-sols | 18,776 | 18,793 | +17 |
| `storm_episode` cell-sols (k=2/m=3) | 10,962 | 10,994 | +32 |
| Candidate episodes | 4,436 | 4,437 | +1 |
| Episodes excluded (split-boundary crossing) | 2 | 2 | unchanged (different boundary, still exactly 2) |
| **Train** | 81,355 (sol 1-600) | **66,826 (sol 1-431)** | val now real |
| **Val** | **0** (sol 601-668, empty) | **14,355 (sol 432-668)** | fixed |
| **Test** | 50,621 (sol 1-501, MY37) | 50,566 (sol 1-501, MY37) | −55, all negative rows |
| Unassigned | 10,697 | 10,685 | −12 |
| Total pipeline wall time | not recorded per-step in this form | 4,515s ≈ 75 min | — |

**Test-split positive counts are UNCHANGED** at every horizon — 440/1,186/1,637 at
h=1/3/5, exactly matching the 5 August run and Section 8's table — confirming the −55
test-row change was entirely negative (non-anomalous) rows the sentinel/missing-data fixes
affected, not a shift in which real storm episodes MY37 contains. **Val positive counts
exactly match the pre-run design estimate** (67/203/291 at h=1/3/5 — see caveat 9), a
direct confirmation the split behaved as designed once real data was run through it.

### Model performance: before vs. after

**Dust-opacity regression (MAE, lower better).** Real, moderate changes — the
missing-data config change (tighter fill gap, new completeness threshold) reduces
available training density somewhat, and the sentinel fix removes a small amount of
contaminated data:

| Horizon | Best baseline | Best classical | Best sequence |
|---|---|---|---|
| 1 sol | seasonal_mean: 0.150 (unchanged) | random_forest: 0.138 (was gradient_boosting 0.139) | gru w=3: 0.141 (was gru w=10: 0.141 — same value, different window) |
| 3 sol | seasonal_mean: 0.138 (was 0.139) | random_forest: 0.116 (was gradient_boosting 0.119) | gru w=5: 0.120 (was gru w=5: 0.127) |
| 5 sol | persistence: 0.123 (was seasonal_mean 0.134 — baseline ranking itself changed) | gradient_boosting: 0.113, R²=0.075 (was 0.105, R²=0.122 — real decline) | gru w=5: 0.115 (was gru w=3: 0.118) |

**Storm detection (TSS, higher better) — the headline result of this whole correction
exercise.** Sequence-model TSS drops substantially at every horizon once early stopping
uses genuinely held-out val instead of leaked test — exactly the expected direction and
magnitude for a real leakage fix, not noise:

| Horizon | Best baseline | Best classical (TSS / ROC-AUC) | Best sequence (TSS) |
|---|---|---|---|
| 1 sol | persistence: 0.285 (unchanged) | logistic_regression: 0.052 (ROC-AUC 0.647); random_forest ROC-AUC **0.833** best | gru w=3: **0.039** (was gru w=10: 0.103 — a real ~62% relative decline) |
| 3 sol | persistence: 0.310 (unchanged) | knn: 0.125 (ROC-AUC 0.743); random_forest ROC-AUC **0.839** best | gru w=10: **0.059** (was 0.163 — a real ~64% relative decline) |
| 5 sol | persistence: 0.296 (unchanged) | knn: 0.160 (ROC-AUC 0.757); random_forest ROC-AUC **0.841** best | gru w=10: **0.154** (was 0.258 — a real ~40% relative decline) |

**Read this plainly, not softened**: the 5 August run's sequence-model storm-detection
skill was inflated by test-set leakage. The corrected numbers are real, methodologically
sound, and lower. `persistence` and classical ROC-AUC are essentially unchanged (both are
either leakage-independent or threshold-independent), which is itself a useful internal
consistency check — the drop is concentrated exactly where the leakage mechanism could
have acted (sequence-model TSS, driven by early-stopped training duration), not spread
uniformly across every metric and model, which would have suggested a data problem instead
of a leakage fix.

**A further real, honest finding from this run**: several LSTM configurations now report
`tss≈0` with `tp=0` (predicting no positives at all) at multiple horizons/windows — e.g.
lstm w=3/w=5/w=10 at h=1 sol. Under genuine validation-based early stopping, LSTM in this
configuration (hidden_size=16, single layer, no dropout) appears to sometimes collapse to
a trivial always-negative predictor rather than learn a useful decision boundary, while
GRU remains consistently informative at every horizon. This was NOT visible in the 5
August run, where test-leaked early stopping evidently kept training going long enough
(or stopped it at a point) that avoided this collapse. Reported as a genuine finding, not
investigated further (would require a dedicated architecture/training-stability study,
out of scope here).

### Table 4.6b: barely changed, as expected

Guha et al. catalogue validation is computed from `labeled_df`/`episode_catalogue`, which
shifted only slightly (+32 `storm_episode` cell-sols, +1 candidate episode) — the
dev/test tolerance-tuning result is correspondingly almost identical:

| Quantity | 5 Aug | 13 Aug |
|---|---|---|
| Qualifying episodes (MY36, k=2/m=3) | 155 | 156 |
| Selected tolerance (dev, F1) | ±4 sols, 3 cells | ±4 sols, 3 cells (unchanged) |
| Held-out test POD | 16.3% (8/49) | 16.3% (8/49, unchanged) |
| Held-out test episode precision | 7.7% (12/155) | 7.7% (12/156) |
| False-positive candidates | 147 | 148 |

### Figures 4.5-4.8: regenerated against the new run — DONE

`generate_figure_4_5_raw_data.py` (full real-archive reload, confirmed 1,077,710 raw
retrievals — the sentinel-fixed count) and `generate_chapter4_preprocessing_figures.py`
both re-run against `run_20260813T195013457439Z_seed42`. Same representative cell (1393),
bit-identical raw-retrieval count (814, 526 accepted/288 rejected — this cell was never
touched by the sentinel/missing-data fixes). All four PNGs, their source CSVs, and
provenance JSONs regenerated and visually verified: the lag-9 pooled-ACF spike (Figure
4.7b) and the `mid_south`/Ls-Q1 semivariogram outlier (Figure 4.7c) both reproduce
exactly under the new run, confirming they are real, repeatable atmospheric/climatology
structure rather than an artefact of the superseded 5 August data. Figure 4.6b's gap-fill
map visibly thins relative to the original generation, consistent with
`max_trailing_fill_gap_sols` dropping from 2 to 1. Full detail: Section 12.

### Confirming MY37 (test) was touched only after the full procedure was frozen

Stated explicitly, with the evidence chain, per the task's own requirement:

1. **The split boundary (item 1 above) was chosen using only `labels/episode_catalogue.csv`
   and `data/aggregated_cell_sol.csv` from the ALREADY-SUPERSEDED 5 August run** — i.e.
   using data that existed before this run started, not by peeking at this run's own test
   results. The boundary was fixed in `configs/splits.yaml` before the corrected run began.
2. **The missing-data sensitivity sweep (item 4) evaluated every one of its 18
   combinations on TRAIN and VAL ONLY** — `run_missing_data_sensitivity_sweep.py` never
   constructs a test dataset or reads a test-split metric at any point; this is enforced by
   the script's own structure (test simply isn't loaded), not by discipline alone. The
   selected configuration was written into `configs/missing_data.yaml` /
   `configs/features.yaml` before the corrected run began.
3. **No hyperparameter search occurred in the corrected run** — every classical and
   sequence model uses the fixed parameters in Section 9's tables, unchanged from before
   this pass; the 0.5 classification decision threshold is unchanged and was never tuned.
4. **Sequence-model early stopping used only the val split** (item 2 above), enforced
   structurally (`modeling/train.py` now refuses to train if val is empty, per the new
   regression test) — test predictions are computed and reported only AFTER training
   (including early stopping) has already concluded.
5. **The catalogue-tolerance selection (Table 4.6b) is independent of the pipeline's
   MY37 test split entirely** — it tunes against a dev/test split of the Guha et al.
   documented-event catalogue (Section 7), a different train/test axis that never touches
   EMIRS MY37 data at all.

Test-split metrics were computed by `evaluate_and_report` after every model (baseline,
classical, sequence) had already been fully fit and, for sequence models, already
early-stopped — the same evaluation code path as every previous run, unchanged by this
correction pass. No result in this run informed any decision that was made before the run
was launched.

---

**Note on this document's history**: this document previously coexisted with a larger
`planning/` corpus, including `planning/evidence_package/` (a 9-file "thesis evidence
package" analogous to this document, but built entirely from a since-deleted simulated
3-task pipeline run) and several pre-implementation planning artifacts
(`DATA_AUDIT.md`, `DECISIONS_REQUIRED.md`, `CLAIMS_TRACEABILITY.md`,
`IMPLEMENTATION_PLAN.md`, `ENVIRONMENT.md`, `REPRODUCE_COMMANDS.md`,
`EMIRS_FIELD_VERIFICATION_REPORT.md`). All of these were removed 2026-08-05 as fully
superseded by real results and real, tested code — their still-relevant content
(open research items, exact formulas/hyperparameters, storm-taxonomy citations) was
folded into this document first, not simply discarded. The remaining `planning/*.md`
files (`PIPELINE_DOCUMENTATION.md`, `CONFIGURATION_DOCUMENTATION.md`,
`DATA_DICTIONARY.md`, `FEATURE_DICTIONARY.md`, `MODEL_CARDS.md`, `SEED_STRATEGY.md`,
`SCHEMA_DOCUMENTATION.md`, `PREPROCESSING_DOCUMENTATION.md`, `LABELING_DOCUMENTATION.md`,
`MODELING_DOCUMENTATION.md`, `EVALUATION_DOCUMENTATION.md`) were kept as a technical
mechanism reference corpus (still linked from `README.md`'s documentation index) — they
describe *how the code works*, a different purpose from this document's *what values to
cite*, though some of their detail still describes the pre-real-only pipeline shape (see
each file's own content, or `README.md`'s caveat).
