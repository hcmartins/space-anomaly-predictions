# Chapter 3 [INSERT]/[CONFIRM] Values — Real, From This Repository

Every value below is read directly from real, working, tested code and real, persisted run
outputs in this repository (`src/solar`), never invented to make the chapter read cleanly.
Where the actual implementation does not do what the chapter's prose assumes, or where a
chapter requirement was not built, that is stated plainly rather than glossed over —
consistent with this project's no-fabrication convention (see memory:
solar-engineering-conventions). Several real counts recomputed directly from persisted
data files during this pass are marked **[COMPUTED THIS SESSION]**; everything else cites
an existing, already-reported source file. Source file/line references are given so every
value can be re-verified independently.

---

## 0. Scope caveats to read before filling anything in

These affect how several `[INSERT]`/`[CONFIRM]` markers below should be interpreted or
reworded, not just what number to write in:

1. **Tables 3.1–3.3's real numbers come from the DISK-INTEGRATED pipeline, not a literal
   per-HARP one — even though Section 3.2.1 formally defines the forecasting unit as "the
   tracked active-region patch (HARP), indexed by `r`".** Table 3.2's counts are verified
   (Section 3 below) to be an exact match for
   `results/real_full_20100501_20250724/dataset_with_split.csv` — the classical/tabular
   pipeline's dataset, built from `labeling/flare_labels.py`'s disk-wide GOES-threshold
   proxy (no HARP index at all), not `labeling/harp_catalogue_labels.py`'s real
   catalogue-attributed per-HARP labels. A separate, real, already-built per-HARP pipeline
   *does* exist (`results/real_full_per_harp_20100501_20250724/`, Phase 1/2 of this
   project — 3,941,133 forecast instances, 6,930 distinct HARPs, `label_primary`-defined
   split sizes 1,770,323/133,562/1,568,837) with materially different numbers from Table
   3.2's. Either (a) reword Section 3.2.1/Table 3.2 to describe the disk-integrated task
   honestly, or (b) replace Table 3.2 with the real per-HARP numbers if the per-HARP
   framing is meant literally — the two must not be conflated. The Appendix below reports
   real results for both.
2. **$X_S(r,t_0)$ as an actual $(T_S,d_S)$ multi-timestep SHARP sequence tensor, per
   Algorithm 3.2, is not built anywhere in this repository.** Real SHARP treatment takes
   one of two forms instead: (a) flat, engineered lag/trailing-window summary features
   (`preprocessing/features.py`, used by the classical/tabular baselines in Section 3.5),
   or (b) a disk-aggregated (not per-HARP) SHARP series merged directly onto GOES's own
   sequence tensor into a single combined $(T,d)=(30,20)$ input (a later, Chapter-5-driven
   extension — see caveat 6). Neither is a literal per-HARP $(T_S,d_S)$ tensor. $T_S$,
   the SHARP look-back $\Delta_S$, and $d_S$ as formally defined by Algorithm 3.2 are
   **NOT AVAILABLE**.
3. **Tables 3.4 (CNN-LSTM), 3.5 (FC-SHARP as a standalone deep net), 3.6 (Transformer),
   3.7 (representation-level multimodal fusion of $h_G$/$h_S$), and the focal-loss
   formula (Section 3.7) are not implemented anywhere in this repository** — confirmed via
   a repository-wide search for `Transformer`, `CNN-LSTM`, `FC-SHARP`, `focal`,
   `MultiHead`/attention, and `bidirectional`: zero matches in any `.py` file under
   `src/solar`. Only Table 3.3 (the plain GOES-only LSTM) is real. This matches
   `src/cross-domain`'s own README, which independently documents the same gap: "Chapter
   3.6's fuller solar stack (CNN-LSTM, FC-SHARP, Transformer, within-solar multimodal
   fusion)... [is] not the complete Chapter 3.6 design."
4. **Only ONE of Section 3.2.1's "nine binary tasks per issuance time" (3 horizons ×
   3 classes) has ever actually been trained: ≥M-class at the 24-hour horizon**
   (`label_primary`, `configs/labeling.yaml`'s `forecast_horizon_hours: 24` and
   `primary_target_class: M`). `configs/labeling.yaml`'s `sensitivity_horizons_hours:
   [6, 12, 24, 48]` is declared but never consumed by any real training entry point
   (`run_pipeline.py`, `run_pipeline_sequence.py`) — confirmed via a repo-wide search: it
   appears only in that config file, in planning docs, and in one unit test.
5. **Section 3.5's claim that "random forest baselines are trained using the same
   imbalance-aware sampling strategies applied to more advanced models" does not mean
   oversampling anywhere in this repository, and it is not applied uniformly across
   baselines.** Real: `configs/task_flare_forecast.yaml` sets `class_weight: balanced`
   (sklearn's standard inverse-frequency reweighting) for `logistic_regression` and
   `random_forest_classifier` only; `gradient_boosting_classifier` and `knn_classifier`
   have no imbalance handling at all. No resampling/oversampling exists anywhere in
   `modeling/classical_models.py`.
6. **Chapter 3.6's own text (page 34) explicitly contrasts its intended fusion design
   against "concatenating raw features at the input level, which can obscure
   modality-specific meaning" — but the one real GOES+SHARP model that exists in this
   repository does exactly that.** `preprocessing/sharp_sequences.py` (built 2026-07-29,
   for `src/cross-domain`'s Chapter 5 needs, not as a direct implementation of Table 3.7)
   concatenates the 2 GOES channels and 18 SHARP descriptors at the raw INPUT level into
   one $(T,d)=(30,20)$ tensor consumed by a single LSTM — not two separate encoder
   branches ($h_G$, $h_S$) combined by an explicit fusion operator (concatenation/gating)
   as Table 3.7 and the surrounding prose describe. It is cited below only as the closest
   real artifact to Table 3.7, with this architectural mismatch stated plainly rather than
   presented as satisfying Table 3.7's design.
7. **The despiking step (Algorithm 3.1 step 3) uses a CENTERED window, not a causal one**,
   unlike every step after it (interpolation, smoothing, background normalization, all
   real trailing/causal operations). This is a deliberate, documented real design choice
   (`preprocessing/goes_preprocessing.py`'s own module docstring): despiking is a one-time,
   offline archive-cleaning pass over historical data, run once before any forecast
   instance is ever constructed — not a per-forecast feature computed at issuance time —
   so no future information about a specific flare reaches a specific forecast instance
   through this step. Worth stating explicitly if Chapter 3.2.1's causality claims are
   read as applying literally to every processing step.

---

## 1. Section 3.2 — Prevalence marker

> "...it is precisely these events whose rarity (under [**INSERT**: exact prevalence from
> Table 3.2] % of forecast instances) makes them the forecasting challenge..."

**Real value, per split** (from Table 3.2, verified in Section 3 below):
train **18.93%**, validation **3.04%**, test **34.25%**. **Overall** (all 526,763
disk-integrated rows, `label_flare_m_or_above` positive rate): **22.80%**
(119,871 positive / 525,818 defined — `results/real_full_20100501_20250724/run_summary.json`'s
`target_summary` entry for `label_primary`). Which one this sentence should cite depends on
intent: "under X%" reads most naturally as a single headline number, for which the real
**overall 22.80%** or the real **test-split 34.25%** (the honest out-of-cycle generalization
number) are the two defensible choices — not a single figure this repository can pick for
the author.

---

## 2. Section 3.2.1 — Forecast Task Definition

### Issuance cadence

> "...forecasts are issued at discrete issuance times $t_0$ on a fixed cadence of
> [**INSERT**: issuance cadence, e.g. every 1h / every 12min / once per SHARP record]."

**Real value: every 12 minutes — "once per SHARP record."** Both the disk-integrated
dataset (`dataset_with_split.csv`) and the per-HARP dataset are built on
`preprocessing/alignment.py::build_aligned_timeline`'s SHARP-cadence timeline (SHARP's
native 720s/12-min `T_REC` cadence, `configs/alignment.yaml`) — every real aligned row is
used as an issuance time, with no subsampling stride applied anywhere in
`modeling/datasets.py`/`modeling/harp_datasets.py`.

### Input tensors — $X_G(t_0)$

| Marker | Real value | Source |
|---|---|---|
| $d_G$ | **2** (both GOES XRS channels: `goes_seq_xrsb_flux_w_m2`, `goes_seq_xrsa_flux_w_m2`) — matches the chapter's own worked example exactly | `configs/goes_sequences.yaml`'s `feature_columns` |
| $T_G$ | **30** steps | `preprocessing/goes_sequences.py::window_length_steps`, computed as `round(look_back_hours*60/cadence_minutes) = round(6*60/12) = 30` |
| Post-resampling cadence | **12 min** | `configs/goes_preprocessing.yaml`'s `resample_cadence_minutes: 12` |
| $\Delta_G$ | **6 h** | `configs/goes_sequences.yaml`'s `look_back_hours: 6` — matches the chapter's own worked example exactly |

### Input tensors — $X_S(r,t_0)$

**Not available as a literal $(T_S,d_S)$ tensor — see scope caveat 2.** The closest real
analogues: $d_S = 18$ (the real `SHARP_PHYSICAL_KEYWORDS`, `data_sources/schema.py` —
USFLUX, TOTUSJZ, TOTUSJH, ABSNJZH, SAVNCPP, MEANPOT, TOTPOT, R_VALUE, MEANGAM, MEANGBT,
MEANGBZ, MEANGBH, MEANALP, MEANJZH, MEANJZD, MEANSHR, AREA_ACR, SHRGT45 — not the
chapter's own example of 16), used both by the classical pipeline's flat features and by
the combined GOES+SHARP sequence extension. $T_S$, the SHARP resampled cadence (if any),
and $\Delta_S$ have no real value to report: no code builds a genuine per-HARP,
multi-timestep SHARP-only tensor. The classical pipeline's closest analogue to a
"look-back window" is `configs/features.yaml`'s trailing/lag design: `lags_steps: [1, 2,
5, 10]` (12/24/60/120 minutes back) and `trailing_window_steps: 10` (120 minutes) — a
120-minute (2h) flat summary-statistics window, not a 30-or-so-step raw tensor.

### Feature manifest composition (page 7 marker)

> "...the full engineered manifest contains 246 features ⟨exact composition of the
> remaining 66 non-SHARP features: confirm from configs/features.yaml⟩."

**The chapter's own arithmetic does not match the real manifest — flagged plainly rather
than forced to fit.** **[COMPUTED THIS SESSION]** from
`results/real_full_20100501_20250724/feature_manifest.json`'s `feature_manifest` list and
`run_summary.json`'s `n_features_in_manifest`: the real total is **257 features**, not
246, and the real count "remaining" beyond the 180 SHARP-`feat_`-derived features is
**77**, not 66:

| Component | Count | Real source |
|---|---|---|
| SHARP `feat_`-derived (18 vars × (4 lags + 5 trailing stats + diff1 + slope) = 18×10) | **180** | matches the chapter's own 18×10 arithmetic exactly |
| SHARP bare current-value columns (`USFLUX`, `R_VALUE`, ... themselves, unprefixed) | **18** | `modeling/datasets.py`'s `raw_current_state_cols` — not counted anywhere in the chapter's 180+66 split |
| Non-SHARP `feat_`-derived (5 vars × 10: `n_active_regions_tracked`, `goes_xrsb_flux_trailing_max_w_m2`, `goes_xrsb_flux_trailing_mean_w_m2`, `goes_xrsa_flux_trailing_max_w_m2`, `goes_xrsa_flux_trailing_mean_w_m2`) | **50** | `configs/features.yaml`'s `variables` list |
| Non-SHARP bare current-value columns (the same 5 variables, unprefixed) | **5** | same |
| Persistence counter (`feat_persistence_elevated_flux_run_steps`) | **1** | `configs/features.yaml`'s `enable_persistence_counter: true` |
| QC/missingness-indicator columns (`feat_is_observed`, `feat_interpolation_flag`, `feat_gap_length_steps`) | **3** | `preprocessing/missing_data.py::apply_trailing_fill`'s output columns, carried straight into the manifest — see also the Section 5 correction below: this directly contradicts the chapter's separate claim (page 26) that "no separate missingness-indicator features are supplied to the models" |
| **Total** | **257** | `run_summary.json`'s `n_features_in_manifest: 257` |

The 246-feature figure cited elsewhere in the chapter's own prose (page 37, "the
classical baselines... use the 246-feature engineered vector") should be corrected to
**257** throughout, or the feature set deliberately trimmed to 246 if that is the number
actually intended to ship — the real, already-built pipeline produces 257 and no
selection step currently reduces it.

#### Full name/source/transform/lag/unit list — the 55 non-SHARP features

The 18 SHARP-keyword features (bare + `feat_`-derived) are already named via
`SHARP_PHYSICAL_KEYWORDS` above. The 55 non-SHARP columns (5 raw + 50 derived) are not
named anywhere else in this document, so the full list is given here, **[COMPUTED THIS
SESSION]** from `feature_manifest.json` cross-referenced against
`preprocessing/alignment.py`, `preprocessing/features.py`, and `configs/alignment.yaml`.

**The 5 raw (bare) base columns**, all built in `preprocessing/alignment.py` at the point
GOES and SHARP are merged onto the SHARP-cadence disk-integrated timeline (**not** in
`preprocessing/features.py`, which only consumes them):

| Name | Source function | Transform | Window | Unit |
|---|---|---|---|---|
| `n_active_regions_tracked` | `aggregate_sharp_disk` | Count of distinct `HARPNUM` values among QC-accepted SHARP records sharing the same `T_REC` timestamp (disk-wide active-region census at that instant) | Instantaneous (no window) | count (dimensionless integer) |
| `goes_xrsb_flux_trailing_max_w_m2` | `attach_goes_trailing_state` | Max of the disk-aggregated 1–8 Å (XRSB, "long") GOES flux over the trailing window ending at (inclusive of) the SHARP timestamp | Trailing **12 min** (`configs/alignment.yaml`'s `goes_trailing_window_minutes: 12` — i.e. since the previous SHARP-cadence step) | W m⁻² |
| `goes_xrsb_flux_trailing_mean_w_m2` | `attach_goes_trailing_state` | Mean, same window as above | Trailing 12 min | W m⁻² |
| `goes_xrsa_flux_trailing_max_w_m2` | `attach_goes_trailing_state` | Max of the disk-aggregated 0.5–4 Å (XRSA, "short") GOES flux, same window | Trailing 12 min | W m⁻² |
| `goes_xrsa_flux_trailing_mean_w_m2` | `attach_goes_trailing_state` | Mean, same window as above | Trailing 12 min | W m⁻² |

**Real detail worth stating explicitly**: the four GOES-derived base columns are
themselves already a 12-minute trailing aggregate of raw 1-minute GOES flux
(`attach_goes_trailing_state`) *before* `preprocessing/features.py` applies its own,
separate 120-minute lag/trailing/diff/slope scheme on top — i.e. the 50 derived features
below are trailing statistics of a trailing statistic, not of the raw 1-minute series
directly. A fifth alignment-stage column, `goes_n_obs_in_trailing_window` (count of raw
GOES 1-min observations found in that same 12-min window), is computed by
`attach_goes_trailing_state` but is **not** in `configs/features.yaml`'s `variables` list
and therefore never enters the 257-feature manifest at all.

**The 50 derived (`feat_`-prefixed) columns** — each of the 5 raw variables above
receives the identical mechanical treatment from `preprocessing/features.py::build_features`
(config: `configs/features.yaml`, `lags_steps: [1, 2, 5, 10]`,
`trailing_window_steps: 10`, `cadence_minutes: 12`), giving exactly 10 derived columns
per variable:

| Suffix pattern | Transform | Lag / window (real minutes, `cadence_minutes: 12`) | Unit |
|---|---|---|---|
| `_lag1` | Exact-timestamp value at $t_0-1$ step | 12 min back | same as base variable |
| `_lag2` | Exact-timestamp value at $t_0-2$ steps | 24 min back | same as base variable |
| `_lag5` | Exact-timestamp value at $t_0-5$ steps | 60 min back | same as base variable |
| `_lag10` | Exact-timestamp value at $t_0-10$ steps | 120 min back | same as base variable |
| `_trailing10_mean` | Mean of the 10 steps strictly before $t_0$ (`s.shift(1).rolling(10, min_periods=1).mean()`) | Trailing 120 min, excludes $t_0$ itself | same as base variable |
| `_trailing10_std` | Std (ddof=0), same window | Trailing 120 min | same as base variable |
| `_trailing10_max` | Max, same window | Trailing 120 min | same as base variable |
| `_trailing10_min` | Min, same window | Trailing 120 min | same as base variable |
| `_diff1` | First difference, $s_{t_0}-s_{t_0-1}$ (`s.diff(1)`) | Single 12-min step | same as base variable (differenced) |
| `_slope10` | Closed-form OLS slope of the value against lag position over the 10 trailing lag steps (`_trailing_ols_slope`) | Trailing 120 min | same as base variable per 12-min step |

Applying this pattern to the 5 raw variables gives the real, full 50-name list:
`feat_n_active_regions_tracked_{lag1,lag2,lag5,lag10,trailing10_mean,trailing10_std,trailing10_max,trailing10_min,diff1,slope10}`,
and the identical 10-suffix set for `feat_goes_xrsb_flux_trailing_max_w_m2_*`,
`feat_goes_xrsb_flux_trailing_mean_w_m2_*`, `feat_goes_xrsa_flux_trailing_max_w_m2_*`,
and `feat_goes_xrsa_flux_trailing_mean_w_m2_*` — 5 × 10 = 50 columns, verified present
verbatim in `feature_manifest.json`.

The remaining 2 non-SHARP, non-base features already documented elsewhere in this file
are `feat_persistence_elevated_flux_run_steps` (count of consecutive prior steps with
elevated flux, unit: steps — Section 6/Section 7 baseline discussion) and the 3
missingness-indicator columns `feat_is_observed`, `feat_interpolation_flag` (both
boolean/0-1), `feat_gap_length_steps` (unit: steps) documented in the Section 5
correction below. 5 (raw) + 50 (derived) + 1 (persistence) + 3 (missingness) = **59**;
adding the 18 bare SHARP current-value columns gives the **77** "remaining beyond 180"
figure above.

### Label function

| Marker | Real value | Source |
|---|---|---|
| Flare catalogue $\mathcal{F}$ | **NOAA's GOES Flare Report, via NCEI** (real, mission-length CSV, verified live 2026-07-25/26) | `data_sources/flare_catalogue_ingestion.py`, `configs/flare_catalogue.yaml`'s header verification log |
| Flux thresholds $\theta_C,\theta_M,\theta_X$ | **$10^{-6}$, $10^{-5}$, $10^{-4}$ W m⁻²** — exact match to the chapter's own stated values | `configs/labeling.yaml`'s `flare_class_thresholds_w_m2` |
| Event-time convention $t_{\text{ev}}(f)$ | **Peak time** ($t_{\text{peak}}$), not start time | `labeling/harp_catalogue_labels.py::build_harp_catalogue_labels` and `labeling/flare_labels.py` both key window membership on `time_utc_peak` throughout |
| Primary target class | **M** (≥M-class), 24-hour horizon | `configs/labeling.yaml`'s `primary_target_class: M`, `forecast_horizon_hours: 24` |

### Forecast horizons and issuance cadence

| Marker | Real value | Source |
|---|---|---|
| Horizons $H$ actually trained | **Only 24h** — see scope caveat 4; $\{6,12,24\}$ h is a config-declared set (`sensitivity_horizons_hours`), never run | `configs/labeling.yaml` |
| Stride between issuance times | **12 min** (every real SHARP-cadence row) | see "Issuance cadence" above |
| Overlap length (for the one real horizon, 24h) | **23h48min** ($H$ − stride) | derived |

### Event association

| Marker | Real value | Source |
|---|---|---|
| Converse case: does one NOAA AR span two HARPs? | **Yes, it occurs** — real **1.18%** of ≥C-class flares (305 of 25,760) are attributed to more than one HARPNUM. **[COMPUTED THIS SESSION]** from `results/real_full_per_harp_20100501_20250724/flares_attributed.csv` | Handled by **crediting every attributed HARPNUM**, not arbitrarily picking one — `labeling/harp_catalogue_labels.py::build_harp_catalogue_labels`'s own docstring: "A flare attributed to multiple HARPNUMs... credits every one of them" |
| % of unassigned ≥C-class flares | **42.86%** unassigned to any HARP overall — of which 31.91% (8,220/25,760) never received a NOAA AR number from SWPC at all, and the remainder had an AR assigned but no matching tracked HARPNUM. **[COMPUTED THIS SESSION]** from the same file | The real bottleneck is upstream of the HARP-attribution mechanism itself (SWPC/NCEI simply not assigning an AR to a flare), not an attribution-logic gap — see `planning/FLARE_CATALOGUE_DOCUMENTATION.md` |
| Unassigned-flare handling policy | **NOT a blanket exclusion/masking.** Real policy is per-class-threshold conditional: a class threshold's would-be NEGATIVE label is masked to NaN only if an unassigned flare's OWN class also reaches that specific threshold within the same window — an unassigned C-class flare can mask C-or-above but never M-or-above (since it provably never reached M-class). Never overrides an already-genuine positive from an attributed flare. | `labeling/harp_catalogue_labels.py::build_harp_catalogue_labels`, lines computing `masked = (positive == 0.0) & (unassigned_max_flux >= thresholds[cls])` |

### Limb exclusion

**Real value: $|{\text{Stonyhurst longitude (LON\_FWT)}}| \le 68°$** — the chapter's second
listed option, not the $\mu=\cos\theta$ threshold form. Source: `configs/alignment.yaml`'s
`per_harp.limb_longitude_threshold_deg: 68`, applied in
`preprocessing/harp_alignment.py::prepare_sharp_per_harp`.

### Overlapping-window de-duplication

**Real policy: option (a) — all overlapping instances are retained**, with metrics
reported at instance level. No explicit de-duplication step exists anywhere in
`modeling/datasets.py`/`modeling/harp_datasets.py`/`modeling/sequence_train.py`.
Cross-split contamination from a label's forward-looking horizon crossing a split boundary
is instead prevented by the real chronological split + `embargo_hours` mechanism
(`splitting/chronological_split.py::find_rows_with_label_horizon_crossing_split_boundary`/
`apply_embargo`, `configs/splits.yaml`'s `embargo_hours: 24`) — a different, real
mechanism from what the chapter's option (a) literally names ("the temporal
train/validation/test buffer"), but functionally the same protection.

---

## 3. Section 3.3 / 3.3.1 / 3.3.2 — Data Sources, Tables 3.1–3.3

### Table 3.1 (flare counts by class, Cycle 24 vs. Cycle 25)

**VERIFIED REAL, no changes needed** — every cell in the draft's Table 3.1 is an exact
match against `planning/FLARE_CATALOGUE_DOCUMENTATION.md`'s real, full 15-year
NOAA_ARS-attribution report (run 2026-07-26, all 329/329 chunks succeeded, 3.9h wall
time): Cycle 24 (2010-05-01..2019-10-01): C=12,641, M=1,194, X=75, Total=13,910,
%M+X=9.12%, %AR=40.0% (9,772/24,413 incl. B). Cycle 25 (2020-01-10..2025-07-24):
C=10,111, M=1,656, X=83, Total=11,850, %M+X=14.68%, %AR=79.6% (11,982/15,058 incl. B).
**Run ID/script path for the caption**: this is the real full-range flare-catalogue
attribution report, `preprocessing/harp_attribution.py`, documented in
`planning/FLARE_CATALOGUE_DOCUMENTATION.md`, real run 2026-07-26. The chapter's own
sanity-check note about Cycle 24's %M+X (9.12% real vs. 9.2% approximated in an earlier
draft) is itself real and already resolved in that same planning doc.

### Table 3.2 (chronological data partition)

**VERIFIED REAL, exact byte-match — no changes needed to the numbers, but see scope
caveat 1 for what dataset they describe.** **[COMPUTED THIS SESSION]**, recomputed
directly from `results/real_full_20100501_20250724/dataset_with_split.csv`:

| Split | Date range | N instances | N≥C | N≥M | N≥X | Prev. ≥M |
|---|---|---|---|---|---|---|
| Train | 2010-05-01 – 2016-10-01 | 237,879 | 169,832 | 45,028 | 4,452 | 18.93% |
| Validation | 2017-03-15 – 2019-12-30 | 77,100 | 12,990 | 2,342 | 431 | 3.04% |
| Test | 2020-01-01 – 2025-07-24 | 211,671 | 157,532 | 72,501 | 8,134 | 34.25% |

Every single value above is an exact match to the draft's own Table 3.2. **Run ID/repo
path for the caption**: `results/real_full_20100501_20250724/` (the classical/tabular
disk-integrated pipeline's real full-range run, `run_pipeline.py`, 2026-07-26) —
specifically `dataset_with_split.csv` and `run_summary.json`'s `split_sizes` field.

### Per-split unique event counts and distinct HARPs (companion to Table 3.2)

Not present in the original parameter file — **[COMPUTED THIS SESSION]** from
`results/real_full_per_harp_20100501_20250724/flares_attributed.csv` (real, per-flare
catalogue attribution with peak flux and attributed `harpnums` list per event) and
`results/real_full_20100501_20250724/sharp_qc_accepted.csv`. Splits use
`configs/splits.yaml`'s raw config cutoffs (train `< 2017-01-01`, val `< 2020-01-01`,
test `< 2025-07-25`) applied to each flare's own peak time — the correct basis for "which
split does this event belong to," independent of whether a forecast-instance row happens
to exist at that exact moment (see the real 165-day gap below). Class membership uses
the flux thresholds $\theta_C,\theta_M,\theta_X$ directly (cumulative — a ≥M event is
also counted in ≥C), not the `flare_class` string, for exact consistency with Section
3.2.1's label definition.

**Cross-check**: summed across all three splits these reproduce Table 3.1's real
cycle-level totals exactly — ≥C: 13,290+620+11,850 = **25,760** (Table 3.1's C+M+X
total); ≥M: 1,200+69+1,739 = **3,008** (= Cycle 24's 1,194+75=1,269 plus Cycle 25's
1,656+83=1,739, split only at a slightly different boundary than the Cycle 24/25 cut);
≥X: 69+6+83 = **158** (= Table 3.1's 75+83). This match is a real, independent
verification of both computations, not a coincidence.

| Split | N events ≥C | Distinct HARPs ≥C | N events ≥M | Distinct HARPs ≥M | N events ≥X | Distinct HARPs ≥X |
|---|---|---|---|---|---|---|
| Train | 13,290 | 588 | 1,200 | 185 | 69 | 31 |
| Validation | 620 | 2 | 69 | 1 | 6 | 0 |
| Test | 11,850 | 581 | 1,739 | 236 | 83 | 40 |

"Distinct HARPs" here means distinct `HARPNUM`s appearing in that event's `harpnums`
attribution list (events with `harpnums == []`, i.e. unassigned, contribute to the event
count but not to any HARP count — consistent with the real ~42.9%-unassigned rate
already documented above). **The validation split's near-total absence of
HARP-attributed events (2 distinct HARPs across all of ≥C, 0 for ≥X) is a real,
significant finding**, not a computation artifact: it is a direct consequence of the same
165-day SHARP outage discussed below eating into the start of the validation period, on
top of validation's already-thin SHARP coverage (326 distinct HARPNUMs per
`configs/splits.yaml`'s header vs. train's 3,827) — worth flagging explicitly if
validation-split HARP-level analysis is planned for Chapter 6.

**Total distinct HARPs tracked per split** (any SHARP activity, not just flare-attributed
— the per-split analogue of Section 3.3.2's full-range 7,145 figure),
**[COMPUTED THIS SESSION]** from `sharp_qc_accepted.csv`: train **3,591**, validation
**278**, test **3,277** (sums to more than 7,145 because a HARP tracked across a split
boundary is counted once per split it appears in). These are close to but not identical
to the illustrative per-split HARP counts in `configs/splits.yaml`'s own header comment
(3,827 / 326 / 3,277) — the header comment's train/val figures appear to predate this
session's exact recomputation; the test figure matches exactly.

### Split boundary-selection rule (page 17 marker)

> "The gaps between split boundaries... follow from the cycle-aware assignment of periods
> to splits together with the 24-hour boundary embargo ⟨exact boundary-selection rule:
> confirm from configs/splits.yaml⟩."

**Real answer: there is no separate "boundary-selection rule" beyond plain calendar
cutoffs, and the 24h embargo is NOT what produces the gaps Table 3.2 shows.**
`configs/splits.yaml`'s real raw cutoffs are round calendar-year boundaries:
`train_end: 2016-12-31`, `val_start: 2017-01-01`, `val_end: 2019-12-31`,
`test_start: 2020-01-01` — `splitting/chronological_split.py::assign_split` does nothing
more than a plain `time_utc` date-range membership test against these cutoffs (no
cycle-phase logic and no embargo at assignment time). The real `embargo_hours: 24`
mechanism (`apply_embargo`) only reassigns rows whose 24h label horizon crosses a split
boundary to a separate `"embargo"` bucket — real count **112 rows** total across all
boundaries (`run_summary.json`'s `split_sizes.embargo`), nowhere near large enough to
explain Table 3.2's real gaps.

**The real cause, [COMPUTED THIS SESSION]** by checking `aligned.csv`'s row density
directly: there is a genuine **165-day data gap** in the aligned disk-integrated timeline,
from **2016-10-01 00:00 UTC to 2017-03-15 23:00 UTC** — no aligned GOES+SHARP row exists
in this window at all in the real ingested data (consistent with Table 3.3's
already-reported 372 SHARP-missing calendar days, "concentrated before 2021-02-08"). This
real gap happens to straddle the config's `train_end`/`val_start` cutoff (2016-12-31 /
2017-01-01), which is why Table 3.2's actually-observed date ranges (train max
2016-10-01, val min 2017-03-15) differ so much from the config's own round cutoffs. This
is a property of real underlying data availability, not a deliberate "boundary-selection
rule" the pipeline applies — worth stating in the thesis text as a data-availability
artifact rather than implying a designed rule produced it.

#### Outage evidence (page 17 marker, expanded)

**[COMPUTED THIS SESSION]**, isolating exactly which stream caused the gap by checking
`goes_qc_accepted.csv` and `sharp_qc_accepted.csv` independently over 2016-09-01 to
2017-04-01:

| Stream | Rows in window (QC-accepted) | Max internal gap in window | Verdict |
|---|---|---|---|
| GOES | 734,792 | 1 minute (none) | **Continuous** — GOES has zero gap through this period at its native 1-min cadence |
| SHARP | 5,370 | **165 days 11 hours** | **The gap is SHARP-only** |

**Precise boundary timestamps** (QC-accepted SHARP records immediately either side of the
gap): last accepted SHARP record before the gap = **2016-10-01 12:00:00 UTC**; first
accepted SHARP record after = **2017-03-15 23:00:00 UTC**. (`aligned.csv`'s own gap,
2016-10-01 00:00 to 2017-03-15 23:00, is these same bounds rounded onto the aligned
timeline's grid.) Since `preprocessing/alignment.py::build_aligned_timeline` keys every
aligned row off the SHARP-cadence (`T_REC`) grid (`aggregate_sharp_disk`, joined via
`attach_goes_trailing_state`), a SHARP-only outage is sufficient on its own to blank out
the aligned timeline for its full duration even though GOES kept reporting throughout.

**Complete absence from source vs. quality-filtered — [COMPUTED THIS SESSION]** by
re-checking the same window in the **pre-QC** raw file
(`data_cache/sharp/normalized/sharp_normalized_full_20100501_20250724.csv`, the direct
input to `preprocessing/quality_control.py`, referenced as `sharp_csv` in
`run_summary.json`): **92,643 raw SHARP records exist in this window**, with a maximum
internal gap of only 5 days — JSOC kept publishing SHARP records essentially
continuously. This is therefore **quality-filtered, not a complete absence from source**:
of those 92,643 raw records, only **5,377 (5.8%)** carry `QUALITY == 0` (the pipeline's
sole accepted value, `configs/quality_control.yaml`'s `sharp.rules[1]`) — a stark contrast
to the real full-range overall SHARP QC acceptance rate of 91.4%
(4,541,205/4,967,165, `run_summary.json`'s `sharp_qc`). The dominant rejected value in
this specific window is **`QUALITY == 128`** (83,583 of the 92,643 raw records, 90.2%) —
**RESEARCHER VERIFICATION REQUIRED**: this repository does not independently confirm
JSOC's bitmask semantics for `QUALITY == 128` against SHARP/HMI documentation, only that
this exact value dominates the raw records throughout the outage window (matching the
same verification caveat already stated for the quality-flag pass/fail semantics
elsewhere in this file). Whatever the physical cause, the real, verifiable fact is that
the data was being produced upstream and was excluded by this project's own QC rule, not
missing at the source.

### Table 3.3 (Chronological — GOES XRS / SHARP row counts)

**VERIFIED REAL, exact match — no changes needed.** GOES rows: **16,597,686**; SHARP
rows: **4,967,165**; date range covered by both: **2010-05-01 to 2025-07-24**; fully
missing calendar days: GOES **2** (2012-03-22, 2016-03-11 — RESEARCHER VERIFICATION
REQUIRED against NOAA's own outage records before citing a cause), SHARP **372**
(concentrated before 2021-02-08, none after). Source: `planning/REAL_FULL_RANGE_RUN.md`,
real full-range ingestion report, 2026-07-26.

### Section 3.3.2 prose — event/HARP counts

> "...corresponding to around 27,700 [**INSERT**: exact event count from the same run]
> labelled flare events..." / "...corresponding to around 25,000 [**INSERT**: exact HARP
> count from the same run]..."

**Neither placeholder matches a real, countable quantity found in this repository — flag
plainly rather than silently substitute a close-but-different number.**

- GOES event count: the real, closest candidates are **25,760** (Cycle 24 + Cycle 25
  combined, C+M+X only — Table 3.1's own convention) or **39,471** (all classes including
  B, both cycles combined — the real full attribution report's "Total flares (All)"
  figure). Neither equals "~27,700." Source: `planning/FLARE_CATALOGUE_DOCUMENTATION.md`.
- SHARP HARP count: the real, verified figure is **7,145 distinct HARPNUMs**
  (QC-accepted SHARP records, full 2010–2025 range). **[COMPUTED THIS SESSION]** —
  `pd.read_csv('results/real_full_20100501_20250724/sharp_qc_accepted.csv')['HARPNUM'].nunique()`.
  This does not match "~25,000" either; the per-HARP forecasting dataset (after limb
  exclusion and feature-window feasibility filtering) uses only **6,930** distinct
  HARPNUMs (`results/real_full_per_harp_20100501_20250724/run_summary.json`'s
  `n_distinct_harps`), an even larger gap from the draft's placeholder.

---

## 4. Section 3.4.1 — GOES Preprocessing (Algorithm 3.1)

All real values below are from `configs/goes_preprocessing.yaml` and
`preprocessing/goes_preprocessing.py`, and mirror the exact operations Algorithm 3.1
lists.

| Step | Marker | Real value | Source |
|---|---|---|---|
| 1. Quality control | Excluded quality flag values | `xrsb_quality` column, `pass_value: 0` accepted (nonzero rejected) — **RESEARCHER VERIFICATION REQUIRED**: the value semantics of nonzero codes are not independently confirmed against NOAA documentation, only that the column exists and is float32 | `configs/quality_control.yaml`'s `goes.rules[1]` (`invalid_quality_flag`) |
| 3. Despiking | Kernel length $k$ | **51 minutes** (51 native 1-min steps) — **NOT** the chapter's own worked example ($n_\sigma=5$, an implicit short kernel) | `configs/goes_preprocessing.yaml`'s `despike_kernel_minutes: 51` |
| 3. Despiking | Clip level $n_\sigma$ | **20.0** | `despike_n_sigma: 20.0` |
| 3. Despiking | Real justification for deviating from the chapter's example | A real pilot-week sweep (2017-09-04..10, the X9.3 reference event) found $k=11\text{min}/n_\sigma=5$ flagged 69% of bins and left ZERO feasible 6h sequence windows — a real flare's rise/decay genuinely looks like a local outlier at that scale, so a narrow kernel deletes real flare signal, not just instrument noise. $k=51\text{min}/n_\sigma=20$ flagged only 7% and produced 879 feasible windows in the same week. Researcher-confirmed. | `configs/goes_preprocessing.yaml`'s own header comment |
| 4. Gap handling | $g_{\max}$ | **10 minutes** | `max_interpolated_gap_minutes: 10` |
| 5. Smoothing | Window length $w$ | **15 minutes**, causal one-sided trailing average — matches the chapter's own formula exactly (pandas' default `.rolling(...).mean()` is right-aligned/trailing) | `smoothing_window_minutes: 15` |
| 6. Background normalization | Estimator $b_t$ | **Rolling 10th-percentile** ($q=10$) of the smoothed series over a trailing window of $L=$**2880 minutes (48h)** — matches the chapter's own example form ("the rolling q-th percentile... over a trailing window of length L") exactly | `background_percentile: 10.0`, `background_window_minutes: 2880` |
| 7. Resampling | Cadence $\tau_G$ | **12 minutes**, matching SHARP's native cadence | `resample_cadence_minutes: 12` |
| 7. Resampling | Aggregation rule | **Mean** within each 12-min bin | `resample_aggregation: mean` |

Figure 3.8's specific caption values (bin width, N, and the illustrated
6-hour-segment/Q–Q-plot generation script/run ID) are **NOT AVAILABLE**: this figure has
not been regenerated from real data in this delivery (same caveat class as Chapter 5's
Figure 5.2 — schematic/illustrative until a dedicated figure-generation script exists).
The despiking $k$/$n_\sigma$ values it references (51 min / 20.0) *are* known, per above.

---

## 5. Section 3.4.2 — SHARP Preprocessing (Algorithm 3.2)

| Marker | Real value | Source |
|---|---|---|
| Quality masking: accepted QUALITY value(s) | **QUALITY == 0** (nonzero rejected) — same **RESEARCHER VERIFICATION REQUIRED** caveat as GOES's quality flag: column/dtype confirmed real, bitmask value semantics not independently verified | `configs/quality_control.yaml`'s `sharp.rules[1]` (`invalid_quality_flag`) |
| Limb criterion | Same as Section 3.2.1 above: $\lvert\text{LON\_FWT}\rvert \le 68°$ | `configs/alignment.yaml`'s `per_harp.limb_longitude_threshold_deg` |
| Winsorisation percentile $p$ | **1** (clips to the 1st/99th percentile) — matches the chapter's own example exactly | `configs/scaling.yaml`'s `winsorize_lower_percentile: 1.0`/`winsorize_upper_percentile: 99.0` |
| Robust scaling estimator | **Median / IQR**, matching the chapter's primary formula $z^{(i)}=(s_{\text{win}}^{(i)}-m^{(i)})/\text{IQR}^{(i)}$ exactly — **not** the z-score alternative the chapter's `[INSERT]` note flags as a possibility | `preprocessing/scaling.py::fit_robust_scaling`/`apply_robust_scaling` |
| Descriptor smoothing (Figure 3.5) | **NOT IMPLEMENTED anywhere in this repository** — no rolling/EWM/moving-average operation is ever applied to a SHARP descriptor column. Figure 3.5 is illustrative/schematic only. | Confirmed via search of `preprocessing/scaling.py` and `preprocessing/features.py` |

### Missing-data and scaling confirm markers (page 26)

> "The maximum permissible carry-forward interval and the treatment of a zero or
> near-zero interquartile range during robust scaling are set in the preprocessing
> configuration ⟨confirm exact values from configs/missing_data.py and
> preprocessing/scaling.py⟩; no separate missingness-indicator features are supplied to
> the models ⟨confirm⟩."

| Marker | Real value | Source |
|---|---|---|
| Maximum carry-forward interval | **3 steps × 12 min = 36 minutes** (`max_trailing_fill_gap_steps: 3`) — a step whose gap since the last real observation exceeds this bound is dropped entirely, never fabricated | `configs/missing_data.yaml` |
| Zero/near-zero IQR treatment | **Guarded, not left to divide-by-zero.** A column with IQR == 0 after Winsorizing (constant within train) gets `scale = 1.0` substituted instead | `preprocessing/scaling.py::fit_robust_scaling`: `"scale": iqr if iqr > 0 else 1.0` |

**Real correction to the chapter's own claim**: "no separate missingness-indicator
features are supplied to the models" is **FALSE** as written. Three real
missingness-indicator columns exist and are part of the real 257-feature manifest (see
the page-7 correction in Section 2 above): `feat_is_observed`, `feat_interpolation_flag`,
`feat_gap_length_steps`, produced by `preprocessing/missing_data.py::apply_trailing_fill`
and carried straight through into `feature_manifest.json`. This should be corrected in
the thesis text rather than confirmed as stated.

---

## 6. Section 3.4.3 — Algorithms 3.1 / 3.2

Algorithm 3.1's steps are fully resolved in Section 4 above; step 10's exact $d_G$
feature list is the 2 GOES channels themselves (`goes_seq_xrsb_flux_w_m2`,
`goes_seq_xrsa_flux_w_m2`) — the sequence LSTM consumes the preprocessed series directly
as per-step features, with no additional per-step feature engineering on top (unlike the
classical/tabular pipeline's lag/trailing/diff/slope features).

Algorithm 3.2, as formally written (a per-HARP $(T_S,d_S)$ sequence builder with an
explicit minimum-coverage discard step), **has no real implementation** — see scope
caveat 2. Its real analogue is a different pipeline shape entirely:
`preprocessing/harp_alignment.py` (limb exclusion) →
`preprocessing/harp_attribution.py` (NOAA AR → HARPNUM mapping, Phase 1) →
`preprocessing/features.py` (flat lag/trailing features, applied per HARP group) →
`labeling/harp_catalogue_labels.py` (labels) →
`modeling/harp_datasets.py` (inner join). There is no step anywhere that discards a
row for insufficient SHARP coverage within a look-back window (step 6's "if coverage <
[INSERT: minimum record count] then discard") — the real per-HARP pipeline instead
relies on `preprocessing/missing_data.py`'s LOCF gap-filling upstream, then an inner join
that naturally drops rows with no aligned features at all. **NOT AVAILABLE** as a literal
coverage-count threshold.

---

## 7. Section 3.5 — Baseline Models

Real baselines built and trained (more than the chapter's own text explicitly lists — a
superset, not a gap): **climatology** (train-mean), **persistence**,
**logistic_regression**, **random_forest_classifier**,
**gradient_boosting_classifier** (disk-integrated run) /
**hist_gradient_boosting_classifier** (per-HARP run — swapped in for real speed reasons,
not a methodology change: plain `GradientBoostingClassifier` took 2,947s/49min on the
disk-integrated run's 237k rows; the per-HARP dataset is ~8.6× larger), and
**knn_classifier**. Source: `modeling/classical_models.py`, `configs/task_flare_forecast.yaml`.

**Real imbalance-handling detail** (see scope caveat 5): `class_weight: balanced` is set
for `logistic_regression` (`max_iter: 3000`) and `random_forest_classifier`
(`n_estimators: 200, max_depth: 8, min_samples_leaf: 3`) only.
`gradient_boosting_classifier` (`n_estimators: 200, max_depth: 3, learning_rate: 0.05`)
and `knn_classifier` (`n_neighbors: 15`) have **no** imbalance-aware handling — Section
3.5's blanket claim that this applies to "random forest baselines" specifically is
accurate for random forest, but should not be read as applying to every baseline.

Chapter 3.5's "third baseline category... classical statistical forecasting techniques and
simple autoregressive formulations" beyond persistence itself is **NOT IMPLEMENTED** as a
separate model class — `persistence` (reading `feat_persistence_elevated_flux_run_steps`)
is the only time-series-oriented baseline that exists; no ARIMA/exponential-smoothing/etc.
model is built anywhere.

---

## 8. Section 3.6 / 3.6.1 — Deep Learning Architecture

### LSTM cell parameters and Table 3.3

| Marker | Real value | Source |
|---|---|---|
| Hidden size $n_h$ | **32** | `configs/task_flare_forecast_sequence.yaml`'s `hyperparameters: {hidden_size: 32, num_layers: 1, dropout: 0.2}` |
| Input dimension $d=d_G$ | **2** (GOES-only checkpoint) | same, `input_size` derived dynamically from `X_train.shape[2]` |
| Uni- or bidirectional | **Unidirectional** — `nn.LSTM` constructed with no `bidirectional=True` | `modeling/sequence_models.py::SequenceModel.__init__` |
| Dropout $p$ | **0.2**, applied via a single `nn.Dropout(0.2)` layer ONCE — after the LSTM's final hidden state, before the output head (not per-gate, not between stacked layers; `num_layers=1` means PyTorch's own inter-layer LSTM dropout is inactive regardless of the configured value) | `modeling/sequence_models.py::SequenceModel.forward`: `h = self.dropout(self.encode(x))` |
| Output layer | Dense, 1 unit, **sigmoid** — exact match to Table 3.3's template | `self.head = nn.Linear(hidden_size, 1)`, `torch.sigmoid(...)` |
| Total trainable parameters (GOES-only, `real_full_sequence_20100501_20250724`) | **4,641** = 4,608 (LSTM) + 33 (head, `Linear(32,1)`) | Verified real checkpoint `state_dict` shapes: `weight_ih_l0 (128,2)`, `weight_hh_l0 (128,32)`, `bias_ih_l0 (128,)`, `bias_hh_l0 (128,)` |

**Real correction to the chapter's own LSTM parameter-count formula**: Section 3.6 states
$4(n_hd+n_h^2+n_h)$ parameters per layer (one bias vector per gate). PyTorch's real
`nn.LSTM` implementation uses **two** separate bias vectors per gate
(`bias_ih` + `bias_hh`), giving $4(n_hd+n_h^2+2n_h)$ — for $n_h=32,d=2$: the chapter's
formula gives $4(64+1024+32)=4480$; the real, verified count is $4(64+1024+64)=4608$, a
difference of exactly $4n_h=128$ parameters. Worth stating in the thesis text itself
rather than leaving the formula as literally written.

A second real checkpoint exists with a wider input (`real_full_sequence_goes_sharp_20100501_20250724`,
$d=20$, 2 GOES + 18 SHARP, concatenated at the input level — see scope caveat 6): total
trainable parameters **6,945** (6,912 LSTM + 33 head). This is **not** what Table 3.3
describes (Section 3.6's own text frames this table as "the first class of deep-learning
models... focuses on temporal sequence learning using GOES-derived features" — GOES-only).
The **4,641**-parameter, $d_G=2$ checkpoint is the real value for Table 3.3 as literally
scoped; the 6,945-parameter checkpoint is reported in the Appendix for completeness.

### LSTM training configuration (page 38 marker)

> "The optimiser, learning rate, batch size, maximum epochs, early-stopping patience and
> weight decay are set in configs/task_flare_forecast_sequence.yaml ⟨confirm exact values
> from repo⟩."

| Marker | Real value | Source |
|---|---|---|
| Optimiser | **Adam** | `modeling/sequence_models.py`: `torch.optim.Adam(model.parameters(), lr=...)` — not itself named in the YAML config, only constructed in code |
| Learning rate | **0.001** | `configs/task_flare_forecast_sequence.yaml`'s `sequence.training.learning_rate` |
| Batch size | **256** | same, `sequence.training.batch_size` |
| Maximum epochs | **50** (real training early-stopped at epoch 6, per the chapter's own already-correct text and the Appendix below) | same, `sequence.training.epochs` |
| Early-stopping patience | **5** | same, `sequence.training.early_stopping_patience` |
| Weight decay | **0 (not applied)** — `torch.optim.Adam` is constructed with no `weight_decay` argument, so PyTorch's default of `0.0` is used | `modeling/sequence_models.py`, the exact `Adam(...)` call site above |

**Real correction to Section 3.7's claim**: "Techniques such as dropout, weight decay, and
early stopping are applied consistently across architectures" is only partially true for
the real LSTM — dropout (0.2) and early stopping (patience 5) are real, but weight decay
is **not** applied (Adam's default `weight_decay=0.0` is left unset). This should be
corrected in the thesis text (Section 3.7) rather than left implying L2 regularisation is
in use for the sequence model.

### LSTM vs. Transformer justification (page 36)

> "...[**INSERT**: your actual, evidence-based reason for using each... state what the
> comparison in Chapter 6 showed. Do not let a generic justification stand in for the
> real one.]"

**No real comparison exists to cite** — per scope caveat 3, the Transformer was never
built, so Chapter 6 cannot report a real LSTM-vs-Transformer result. The real, honest
reason only the LSTM was built: this delivery deliberately scoped Chapter 3.6 to
LSTM-only, with CNN-LSTM/FC-SHARP/Transformer/within-solar multimodal fusion as explicit,
stated follow-on phases (not started) — the LSTM was chosen as "the established recurrent
reference for irradiance sequences" (a real scoping decision, not a result of empirical
comparison). This should be stated as a scoping limitation, not dressed up as an
evidence-based architectural choice this project's own results support.

### Tables 3.4–3.7

**NOT AVAILABLE — every cell.** Per scope caveat 3: CNN-LSTM (Table 3.4), FC-SHARP (Table
3.5), Transformer (Table 3.6, including $d_{\text{model}}$, heads $h$, $d_{ff}$,
positional-encoding type, pooling), and the multimodal fusion model (Table 3.7, including
the fusion operator and joint head) do not exist anywhere in this repository.

---

## 9. Section 3.6.2 — Use at Inference

> "...conversion to a binary alert uses a threshold $\tau^*$ selected on validation data
> only, by [**INSERT**: criterion, e.g. maximising TSS]..."

**Real value: maximizing TSS**, exactly the chapter's own example. Source:
`configs/task_flare_forecast_sequence.yaml`'s `threshold_selection: {metric: tss,
grid_min: 0.01, grid_max: 0.99, grid_steps: 99}`, implemented in
`modeling/metrics.py::select_threshold`. Real threshold for the GOES-only LSTM: $\tau^*
= 0.36$ (see Appendix). This is the same real convention `src/cross-domain`'s Chapter 5
work independently confirmed for the solar Stage-1a selection criterion.

---

## 10. Section 3.7 — Class Imbalance Handling and Model Optimisation

| Marker | Real value | Source |
|---|---|---|
| $w_+$ | **$N/(2N_+)$** — exact match to the chapter's own example formula, computed on TRAIN split only | `modeling/sequence_models.py::compute_class_weights` |
| $w_-$ | **$N/(2N_-)$** (the symmetric real formula) | same |
| Focal loss $\alpha$, $\gamma$, selection procedure | **NOT IMPLEMENTED anywhere** — per scope caveat 3, `modeling/sequence_models.py`'s `weighted_binary_cross_entropy` is the only loss function used by the LSTM; no focal-loss implementation exists in this repository | Confirmed via repository-wide search for `focal` |

Class weights are recomputed once per training call (from `y_train` at the start of
`train_sequence_model`), not per-epoch or per-batch — applied identically across every
minibatch for the duration of that run.

---

## 11. Section 3.8 — Evaluation Strategy and Model Interpretability

Threshold selection is covered in Section 9 above. Real metrics computed and reported for
every model (`modeling/metrics.py::classification_metrics`, `metrics:` list in
`configs/task_flare_forecast_sequence.yaml`): **tss, hss, pod, far, precision, recall, f1,
brier, log_score, pr_auc, roc_auc** — a real superset covering both the deterministic
skill scores (TSS/HSS/POD/FAR) and the probabilistic scoring rules (Brier, log score) the
chapter's text describes, plus ranking metrics (PR-AUC, ROC-AUC).

**Real gap**: attribution/interpretability analysis for the deep-learning (LSTM) model is
**NOT IMPLEMENTED**. `modeling/interpretability.py` exists and is real, but applies only
to the classical models (coefficient analysis for logistic regression, feature
importances for tree-based models) — no SHAP, integrated-gradients, or attention-weight
analysis exists for the sequence LSTM. Section 3.8's claim that "attribution methods are
used to analyse how different inputs influence forecast outputs" for deep models is not
yet true in this repository; only the classical-baseline half of that claim is real.

---

## 12. Examiner round 9 (disk-integrated) — submission-blocker corrections

This section resolves the six submission blockers and the additional corrections from the
round-9 examiner review of the disk-integrated rewrite. Each item below is the real,
verified answer from the repository, or an explicit statement that the repository does not
yet contain the answer — never a fabricated number to fill the gap.

**Round-12 status update** — the round-12 draft (`Chapter 3 - Revised (examiner round
12).pdf`) was cross-checked line by line against this section and against the repository
directly. Every one of the six blockers is now **RESOLVED**: the round-12 chapter text
matches the real repository values/behaviour documented here almost verbatim (in several
places, e.g. the disk-aggregation reducer table and Algorithm 3.2's corrected step order,
the wording is close enough to this file's own recommended language to indicate it was
used as the source), and the two items that genuinely could not be closed by editing —
the paired common-subset comparison (part of Blocker 4) and the full statistical-uncertainty
analysis (Blocker 5) — were closed this same session by actually running inference against
the real trained checkpoints and the real resampling analysis on top (Sections 12.8–12.9).
**As of this update, the only remaining open items in this whole Section 12 are the two
genuinely non-repository ones**: a visual check that Figures 3.1/3.5 were actually
redrawn (Section 12.7 — cannot be verified from extracted PDF text), and committing the
repository to version control for a citable tag/archive identifier (Section 12.7).

### 12.1 Blocker 1 — exact SHARP disk-aggregation rule (was: placeholder in Section 3.2.1) — **RESOLVED in round 12**

Round 12's Section 3.2.1 and its new Appendix 3.A Table 3A.2 now state the reducer table,
the unweighted/all-limb-passing-records rule, and the "a timestamp with no surviving
region produces no row" behaviour explicitly, word-for-word matching the real values
below. Round 12 also adds one further real, physically important observation not
previously stated: because the 11 intensive descriptors each take their disk-wide maximum
**independently**, the maxima at a single timestamp can come from **different** active
regions — so the resulting 18-dimensional vector is "a disk-level risk representation
rather than a description of a single coherent active region." This is a correct
description of `aggregate_sharp_disk`'s real behaviour (each column's `.max()` is computed
independently per keyword in the same `groupby().agg()` call, with no joint
same-HARP constraint across columns) and is a good-faith addition worth preserving.

**Real, per-descriptor reducer**, `configs/alignment.yaml`'s `sharp_keyword_aggregation`
map, applied in `preprocessing/alignment.py::aggregate_sharp_disk`:

| Descriptor | Reducer | Physical rationale (config header comment) |
|---|---|---|
| USFLUX | **sum** | extensive (total unsigned flux) — meaningful summed across regions |
| TOTUSJZ | **sum** | extensive (total current) |
| TOTUSJH | **sum** | extensive (total current helicity) |
| ABSNJZH | **sum** | extensive |
| SAVNCPP | **sum** | extensive |
| TOTPOT | **sum** | extensive (total photospheric excess magnetic energy) |
| AREA_ACR | **sum** | extensive (area) |
| MEANPOT | **max** | intensive ("mean-per-region" complexity measure) |
| R_VALUE | **max** | intensive |
| MEANGAM | **max** | intensive |
| MEANGBT | **max** | intensive |
| MEANGBZ | **max** | intensive |
| MEANGBH | **max** | intensive |
| MEANALP | **max** | intensive |
| MEANJZH | **max** | intensive |
| MEANJZD | **max** | intensive |
| MEANSHR | **max** | intensive |
| SHRGT45 | **max** | intensive |

Rule: **sum** for the 7 extensive/"total" quantities (USFLUX, TOTUSJZ, TOTUSJH, ABSNJZH,
SAVNCPP, TOTPOT, AREA_ACR), **max** for the 11 intensive/"mean-per-region" complexity
quantities — the config header states this explicitly as "the single most
complex/stressed region on the disk is the flare-risk driver, not a disk-wide average
that a quiet region would dilute." This is documented as an **engineering judgement
call, not a thesis-specified rule** (`planning/PREPROCESSING_DOCUMENTATION.md`).

Answering the examiner's five specific sub-questions:

- **Weighted by active-region area?** No. Both `sum` and `max` in
  `aggregate_sharp_disk` are plain, unweighted pandas `.agg()` calls
  (`df.groupby("time_utc").agg(**agg_kwargs)`) — no area-weighting term is applied to
  any descriptor, including the intensive (`max`) ones.
- **Do all passing HARPs contribute?** Yes — every SHARP record surviving the
  per-record QUALITY==0 and limb (`|LON_FWT| ≤ 68°`) filters at that exact `T_REC`
  timestamp is included in the groupby; there is no additional HARP-selection or
  top-N logic.
- **What happens when no HARP survives filtering at a timestamp?** `aggregate_sharp_disk`
  is a `groupby("time_utc")` over the already-filtered records — a timestamp with zero
  surviving records simply produces **no row at all** in `sharp_disk` (not a
  zero-filled or NaN-filled row). This is the exact mechanism behind the real 165-day
  SHARP outage documented in Section 3 above: the outage is not a special case, it is
  the general "zero surviving HARPs → absent timestamp" behaviour, subject afterward to
  `missing_data.py`'s 36-minute LOCF cap (Section 12.7 below).
- **Aggregation before or after Winsorisation/scaling?** **Aggregation happens first, at
  the raw per-record level, then Winsorisation/scaling is applied afterward to the
  disk-aggregated (already-summed/maxed) series** — the reverse of the order Algorithm
  3.2 currently states (steps 2–3, "compute percentiles… Winsorise and scale every
  record," listed *before* step 4, "disk-aggregate"). The real pipeline order, traced
  through `run_pipeline.py::run_pipeline` and `modeling/train.py::run_task`, is:
  (1) `build_aligned_timeline` → `aggregate_sharp_disk` (raw sum/max across HARPs,
  **unscaled**); (2) `apply_trailing_fill` (36-min LOCF on the now-aggregated disk
  series); (3) `build_features` (lag/trailing/diff/slope feature engineering on the
  aggregated+filled series); (4) only then, at model-training time in
  `modeling/train.py`, `fit_robust_scaling`/`apply_robust_scaling` Winsorises and
  scales the resulting **feature-engineered disk-level columns** (fit on the train
  split only). Algorithm 3.2's step order should be corrected to match: quality/limb
  filtering (record-level) → disk aggregation → gap-filling → feature engineering →
  Winsorise/scale (on the aggregated, engineered columns, not on raw per-record
  values).
- **How is disk aggregation reproduced in the (30, 20) LSTM tensor?** The combined
  GOES+SHARP LSTM's SHARP channels are built by
  `preprocessing/sharp_sequences.py::build_combined_sequence_table`, which calls the
  **exact same** `aggregate_sharp_disk` function (re-used, not reimplemented) and then
  performs a **strict exact-`time_utc`-match** left-merge onto GOES's own 12-minute
  grid (`validate="one_to_one"`) — never a nearest-neighbour or interpolated join. A
  GOES timestamp with no identically-timed SHARP-disk row gets `NaN` in every SHARP
  channel, flagged via `sharp_seq_missing_any`, which the sequence-feasibility check
  then treats as a hard gap (same causal-exclusion logic as Algorithm 3.1's flagged
  GOES gaps). Winsorise/scale (`fit_robust_scaling`/`apply_robust_scaling`) is then
  fit on this merged table's own train-range rows and applied — **a separate scaler
  fit from the classical/tabular pipeline's**, even though both ultimately scale the
  same underlying `aggregate_sharp_disk` output.

### 12.2 Blocker 2 — residual HARP-level language — **RESOLVED in round 12**

All six flagged passages are rewritten in round 12, each matching the recommended
replacement below almost verbatim: "predicting whether at least one flare... will occur
anywhere on the visible solar disk" (Section 3.2); "SHARP magnetic descriptors are
aggregated across the visible, quality-accepted active regions into one disk-wide vector
per timestamp" (Section 3.3); "flare likelihood is inferred from recent flare history or
background activity levels of the disk state" (Section 3.5, persistence baseline —
previously "of an active region"); "Persistence and climatological models struggle to
adapt to rapid changes in the disk-wide magnetic and irradiance state" (Section 3.5,
failure modes — previously "active-region behaviour"); "gradient boosting is substituted
on the larger disk-level tabular dataset for computational efficiency" (Section 3.5
baseline configs — previously "per-HARP dataset"); labelling "at the disk level,
consistent with the disk-integrated label of Section 3.2.1" (Section 3.4). No residual
per-region framing was found in a full read-through of the round-12 text.

The real, verified fact this rewrite needed to follow throughout (already established in
Sections 2–3 above): the trained models are disk-integrated with no HARP index; SHARP
descriptors are 18 disk-wide values per timestamp (Section 12.1); one forecast instance
exists per 12-minute disk timestamp, not per active region. Table retained for reference:

| Flagged phrasing | Real, disk-consistent replacement | Round-12 status |
|---|---|---|
| "whether an active region will produce…" | "whether at least one qualifying flare will occur anywhere on the visible disk" | Fixed (Section 3.2) |
| "SHARP features… serve as the spatial units of prediction" | "SHARP descriptors are aggregated across all visible, quality-accepted active regions into one disk-wide vector per timestamp" | Fixed (Section 3.3) |
| "predictions are issued for individual active regions" | "one forecast is issued per disk timestamp" | Fixed (Section 3.3) |
| baseline probabilities "associated with an active region" | "associated with the full-disk state at that timestamp" | Fixed (Section 3.5) |
| labelling performed "at the active-region level" (Section 3.4) | "at the disk level" | Fixed (Section 3.4) |
| failure modes "changes in individual active-region behaviour" | "changes in disk-wide magnetic and irradiance state" | Fixed (Section 3.5) |

### 12.3 Blocker 3 — QUALITY = 128 bit definition — **RESOLVED in round 12 (via the empirical-only fallback)**

Round 12's Section 3.3.1 now reads: "No physical meaning is assigned to QUALITY = 128 or
to any other nonzero value: the implemented rule simply retains only records with
QUALITY = 0 and rejects all others, applying no per-bit masking of individual flags." This
is exactly the fallback this file recommended below — the confirmation marker is gone,
and no undocumented bit interpretation is asserted. The authoritative JSOC bit definition
itself remains genuinely unavailable (see the verification attempt below); round 12
correctly avoids asserting one rather than waiting for it.

**[ATTEMPTED THIS SESSION, NOT RESOLVED]**: the examiner-cited authoritative source,
`https://jsoc.stanford.edu/doc/data/hmi/sharp/old/sharp.MB.htm`, was fetched twice
(https and http) and both attempts failed with a TLS/SSL negotiation error
(`OPENSSL_internal:UNSUPPORTED_PROTOCOL` — the JSOC server's TLS configuration is
incompatible with the fetch tool available in this session, an infrastructure limitation,
not a content-access refusal). No other DRMS/JSOC bit-table source for the
`hmi.sharp_720s` or `hmi.sharp_cea_720s` `QUALITY` keyword was consulted, so this
repository **cannot supply the authoritative decimal/hexadecimal bit definition for
`QUALITY = 128`** the examiner is asking for. The 90.2%-dominant value during the real
165-day outage window (Section 3 above) remains an empirically observed fact, not an
interpreted one.

**Per the examiner's own explicit fallback instruction, the chapter text should be
corrected to report only the empirical rule**: records are excluded whenever
`QUALITY != 0`; the pipeline applies no per-bit masking of "serious" vs. "cosmetic"
flags, and no physical meaning should be assigned to `QUALITY = 128` (or any other
nonzero value) without an authoritative citation. The confirmation marker itself
(`⟨…confirm…⟩`) must not remain in the submitted text — replace it with this empirical
statement, not with a guessed interpretation.

### 12.4 Blocker 4 — model comparison fairness (different effective samples) — **RESOLVED (non-paired labelling in round 12; paired common-subset comparison computed this session — see 12.8–12.9)**

Round 12 adds Table 3.6 to Section 3.8, reproducing the exact per-model split
counts/prevalence table below (identical numbers) and stating explicitly: "The three
models are scored on their own maximum-available datasets, not on a common held-out
subset; the differing prevalences... mean cross-model differences are not a paired
comparison and should not be read as architecture effects alone." This resolves the
examiner's "comparison 2, clearly labelled non-paired" requirement in full. The
examiner's "comparison 1" — all three models scored on an identical, common
sequence-feasible test subset with paired per-instance differences — is **still not
implemented anywhere in the repository**; round 12 does not claim otherwise. This is one
of the two genuinely outstanding items — see Section 12.8.

**Real, exact per-model split counts and prevalence**, [COMPUTED THIS SESSION] from
`run_summary.json` in each model's own results directory (positives derived from each
split's `tp + fn`, matching `n_train`/`n_val`/`n_test` exactly in every case):

| Model | Split | N instances | N positive (≥M) | Prevalence |
|---|---|---|---|---|
| Tabular baselines (`real_full_20100501_20250724`) | Train | 237,879 | 45,028 | 18.93% |
| | Validation | 77,100 | 2,342 | 3.04% |
| | Test | 211,671 | 72,501 | 34.25% |
| GOES-only LSTM (`real_full_sequence_20100501_20250724`) | Train | 66,352 | 13,689 | 20.63% |
| | Validation | 66,029 | 1,034 | 1.57% |
| | Test | 107,489 | 27,697 | 25.77% |
| GOES+SHARP LSTM (`real_full_sequence_goes_sharp_20100501_20250724`) | Train | 28,719 | 5,872 | 20.45% |
| | Validation | 21,179 | 418 | 1.97% |
| | Test | 37,598 | 10,493 | 27.91% |

**This confirms the examiner's concern is real, not hypothetical**: the three models see
materially different samples and prevalences at every split — validation prevalence
alone ranges from 1.57% (GOES-only) to 3.04% (tabular), nearly a 2× spread, entirely a
consequence of Algorithm 3.1's strict "any flagged gap in the 30-step window → discard"
windowing (tightened further for the GOES+SHARP LSTM by the exact-match SHARP merge of
Section 12.1). Reported TSS/ROC-AUC differences between the tabular baselines and either
LSTM in the Appendix below **cannot be attributed to model architecture alone**.

**Real gap**: a common, identically-evaluated sequence-feasible subset (examiner's
requested comparison 1 — all three models scored on the same held-out rows) is **NOT
IMPLEMENTED anywhere in this repository** — no such intersection dataset or evaluation
script exists. The counts above are each model's own maximum-available dataset
(examiner's comparison 2), which is what Chapter 6 currently reports, but it is not
explicitly labelled there as a non-paired comparison. Both the paired subset evaluation
and the explicit non-paired labelling are real, currently-missing pieces of work — this
file states the gap rather than fabricating a paired result.

### 12.5 Blocker 5 — statistical uncertainty (single-seed only, confirmed absent) — **RESOLVED this session on the common subset (see 12.8–12.9); full-test-set intervals for every model individually remain a possible future extension**

Round 12's Section 3.8 text was essentially unchanged in substance from round 9 on this
point when reviewed — it still stated single-seed point estimates, no paired
block-bootstrap intervals, no McNemar tests, and identified the full reliability
decomposition as an "open item" for the solar models, correctly, since this genuinely
could not be resolved by further editing. This session ran the actual day-block-bootstrap,
paired-difference, McNemar, and reliability-binning analysis the examiner specified,
on the real per-instance predictions — see Section 12.8 for the method and Section 12.9
for the real numbers and a drop-in results paragraph for §3.8.

**Real, confirmed absence** — cross-checked directly against `run_summary.json` for
every real solar run (`real_full_20100501_20250724`,
`real_full_sequence_20100501_20250724`, `real_full_sequence_goes_sharp_20100501_20250724`):
every run uses `"seed": 44` exactly once, with a single point-estimate `metrics` block
per split. Repository-wide search confirms **no** bootstrap, McNemar, paired-difference,
or confidence-interval code exists anywhere under `src/solar` (`modeling/metrics.py`
computes only the point estimates listed in Section 11 above). This matches — and does
not contradict — the chapter's own already-honest text ("single-seed point estimates…
identified as future work").

**What the examiner is asking the chapter to add is genuinely not yet built**, not a
documentation gap this parameter file can resolve by citing a real value. If this work
is undertaken before submission, the examiner's own checklist is the real specification
to implement against: resampling unit = event or time-block (not instance, given the
documented overlapping-window dependence — Section 2's overlap-retention policy);
block length, replicate count, interval level, and seeds would all need to be chosen and
reported explicitly; paired difference intervals would need the common-subset evaluation
of Section 12.4 first (a paired test requires paired predictions). None of this exists
in the repository today, and no number should be invented to fill it.

### 12.6 Blocker 6 — per-HARP material prominence (editorial, not a value) — **RESOLVED in round 12**

Round 12 moves Table 3.2b, the event-association paragraphs, and the full
NOAA-AR-to-HARPNUM catalogue-chain discussion into a new **Appendix 3.B — Per-region
(per-HARP) event accounting**, opening with an explicit framing paragraph: "The
disk-integrated models of this chapter carry no active-region index and do not use
active-region attribution... None of it defines the modelled instances or the reported
results." Section 3.3's main body now retains only a one-sentence pointer ("Per-split
unique flare-event and distinct-HARP counts... are reported in Appendix 3.B"). Exactly
the restructuring recommended below, with no numbers changed.

Not a code value — a structural suggestion for the chapter's layout. The real, verified
per-HARP numbers themselves (Table 3.2b's event/HARP counts, the 7,145-HARP figure, the
1.18%/42.9% attribution statistics) remain accurate and cross-validated (Section 3
above); the examiner's point is only that, since no per-HARP model is trained or
reported in this chapter, this material's current placement in the main body may read as
disproportionate. Moving Table 3.2b, the event-association paragraphs, and the
NOAA-AR-to-HARPNUM catalogue-chain discussion into an appendix (retaining only a
one-sentence pointer in Section 3.3.2, as the chapter's own "Two populations must not be
conflated" paragraph already partly does) is a real, low-risk restructuring — no numbers
change, only their location in the document.

### 12.7 Additional corrections

| Item | Real answer | Round-12 status |
|---|---|---|
| "once per SHARP record" phrasing | Should read **"once per 12-minute disk timestamp"** — confirmed via `aggregate_sharp_disk`'s `groupby("time_utc")`: multiple SHARP records (multiple HARPs) routinely share one timestamp and are reduced to one disk-wide row, so "per SHARP record" is imprecise. | **RESOLVED** — Section 3.2.1 now reads "once per 12-minute disk timestamp (once per 12-minute disk timestamp; several SHARP records may share one timestamp and are reduced to a single disk-wide row)" throughout, including the previously-residual Appendix 3.A/opening-paragraph instances. |
| Are the 18 SHARP channels in the combined LSTM aggregated before or after gap-filling? | **No gap-filling (LOCF) is applied at all** in the combined-LSTM path. `sharp_sequences.py::build_combined_sequence_table` calls `aggregate_sharp_disk` directly and merges by exact timestamp; a missing disk-aggregated timestamp becomes `NaN` + a `sharp_seq_missing_any` flag (window-exclusion), not a filled value. This is a real, material difference from the classical/tabular pipeline, which does apply the 36-minute LOCF between aggregation and feature engineering. | **RESOLVED** — Section 3.4.2's missing-data paragraph now states this explicitly and correctly: "The combined GOES+SHARP LSTM is handled differently: its SHARP channels are merged onto the GOES grid by exact timestamp with no carry-forward, and a timestamp with no disk-aggregated SHARP row becomes missing and causes the whole window to be excluded (the same treatment as a flagged GOES gap)." |
| Does the 36-minute carry-forward apply to aggregated disk descriptors or individual HARP records? | **Aggregated disk descriptors only.** `preprocessing/missing_data.py::apply_trailing_fill` is called in `run_pipeline.py` strictly *after* `build_aligned_timeline`/`aggregate_sharp_disk` — its input is already the disk-level (summed/maxed) series, never raw per-HARP SHARP rows. | **RESOLVED** — Algorithm 3.2's step order is now corrected to match the real pipeline exactly: step 1 (per-record QUALITY/limb filter) → step 2 (disk-aggregate) → step 3 ("Gap-fill the aggregated disk series by LOCF up to 36 minutes... then build the lag/trailing/diff/slope features on the filled series") → step 4 (Winsorise/scale the engineered columns). This is the same corrected order this file recommended in Section 12.1 above. |
| Figure 3.1 / Figure 3.5 updates | Not a repo value — a diagram-redraw task. The real process the figures need to depict: many SHARP records (multiple HARPNUMs) at one T_REC → one disk-wide vector via `aggregate_sharp_disk`'s per-keyword sum/max, not a single-HARP pipeline. | **CANNOT BE VERIFIED FROM TEXT** — the round-12 PDF's Figure 3.1/3.5 captions are unchanged from round 9, and this file has no way to inspect whether the embedded diagram images themselves were redrawn (only the surrounding text was reviewed). Requires a direct visual check of the two figures before submission. |
| Table 3.2 correspondence to disk-level dataset | Already correct and previously verified (Section 3 above): Table 3.2's numbers are an exact byte-match to `results/real_full_20100501_20250724/dataset_with_split.csv`. | **RESOLVED, unchanged** — round 12's Table 3.2 numbers are identical to round 9's, still an exact match; round 12 additionally strengthens the caption to state N≥C/N≥M/N≥X are "positive instance counts... not unique-event counts, which are given in Table 3.2b [now Appendix 3.B]." |
| Chapter 6 disk-wide-label consistency | Cannot be verified from this repository pass — Chapter 6's text was not supplied for review. | **STILL CANNOT BE VERIFIED** — Chapter 6 was not supplied in round 12 either. Must be checked directly against the Chapter 6 draft before submission; do not assume it inherits Chapter 3's disk-integrated corrections automatically. |
| Replace local function/path references with a repository tag/archive identifier | `git status` returns "not a git repository" — no commit hash, tag, or archive identifier exists to cite. | **STILL NOT RESOLVED** — not re-checked as part of round 12 (this is an infrastructure prerequisite outside the chapter text itself, not something a text edit can fix); confirm current git/archival status before final submission. |

### 12.8 Common-subset comparison + statistical uncertainty — **RESOLVED this session (real code run, not documentation)**

Both remaining items are now closed by actually running inference against the real,
already-trained checkpoints and the real resampling analysis on top — not by further
editing. Full method, scripts, and every output file are below; Section 12.9 gives the
real numbers and a drop-in results paragraph for §3.8.

**What was built and run**, all under `src/solar/`, none retraining anything:

1. `export_baseline_predictions.py` — reloads all 6 real baseline/classical checkpoints
   (`checkpoints/real_full_20100501_20250724/*.joblib`), deterministically re-derives the
   same train-fit SHARP robust-scaling params + median imputer `modeling/train.py::run_task`
   used originally (pure functions of the train data, no randomness), applies them to the
   test split, and calls `predict_proba`.
2. `export_lstm_predictions.py --variant {goes,goes_sharp}` — reloads the two real LSTM
   checkpoints (`checkpoints/real_full_sequence_20100501_20250724/` and
   `checkpoints/real_full_sequence_goes_sharp_20100501_20250724/lstm_label_primary.pt`),
   rebuilds the exact test-split sequence tensors `run_pipeline_sequence.py` built
   originally (same GOES Algorithm 3.1 preprocessing, same disk-aggregated SHARP merge for
   the combined variant), and runs the frozen network forward. Run as two separate process
   invocations, not two calls in one process — building `(T=30, d=20)` tensors for the
   full ~526k-row dataset before splitting hit a real `ArrayMemoryError` (2.35 GiB) on this
   machine; restricting to the test split before windowing and freeing intermediate frames
   with explicit `del`/`gc.collect()` fixed the GOES-only variant, but the combined variant
   still needed a fresh process to avoid memory fragmentation from the first variant's
   already-freed large arrays.
3. `analyze_common_subset.py` — intersects the three exports' `time_utc` columns into one
   common subset, recomputes TSS/HSS/ROC-AUC/PR-AUC/Brier for all 8 models on it, runs a
   day-block-clustered bootstrap (block = UTC calendar day, at least as coarse as the
   documented 24h-horizon window-overlap dependence; 200 replicates, 90% percentile
   interval, seed 44 — matching this project's seed convention and the Martian block-bootstrap
   convention already used in Chapter 6) for per-model intervals and paired TSS-difference
   intervals, runs continuity-corrected McNemar's test on paired thresholded decisions for
   the five narratively relevant model pairs, and computes 10-bin reliability tables.

**Every one of the 8 models' reconstructed test-split confusion matrix was checked against
the already-recorded `metrics.json`/`run_summary.json` values from the original training
run before any export was written** — all 8 matched exactly (0 mismatches); each export
script is written to hard-fail rather than write a silently-wrong file if reconstruction
had not matched. This is real, verified evidence that the exported probabilities are
identical to the ones the original reported metrics were computed from, not
independently-derived approximations.

**Output files** (`results/prediction_exports/`): `baseline_predictions_test.csv` (211,594
rows × 6 probability columns + `baseline_thresholds.json`), `goes_lstm_predictions_test.csv`
(107,489 rows), `goes_sharp_lstm_predictions_test.csv` (37,598 rows, each with its own
`_threshold.json`), and the five analysis outputs:
`common_subset_point_metrics.csv`, `common_subset_bootstrap_intervals.csv`,
`common_subset_paired_tss_diff.csv`, `common_subset_mcnemar.csv`,
`common_subset_reliability_bins.json`.

### 12.9 The real common-subset numbers, for §3.8

**The common subset itself**: intersecting the three models' test timestamps gives
**37,598 rows spanning 1,628 distinct UTC calendar days** — this equals the GOES+SHARP
LSTM's own test set exactly (the tightest of the three; its combined causal-gap
requirements are the binding constraint, confined within both the GOES-only LSTM's
107,489-row test set and the tabular baselines' 211,594-row test set). Every model below
is scored on these same 37,598 rows at its own already-selected (validation-selected)
threshold — a genuine paired comparison, unlike the Table 3.6/Appendix figures below,
which use each model's own maximum-available test set.

**Point metrics on the common subset** (`common_subset_point_metrics.csv`), ranked by TSS:

| Model | Threshold | TSS | HSS | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| persistence | 0.01 | **0.6369** | 0.5651 | 0.8185 | 0.5562 | 0.1973 |
| goes_sharp_lstm | 0.08 | 0.5552 | 0.4359 | 0.8540 | 0.7009 | 0.1514 |
| random_forest_classifier | 0.22 | 0.5441 | 0.4065 | 0.8955 | 0.7477 | 0.1317 |
| logistic_regression | 0.16 | 0.4828 | 0.3439 | 0.9083 | 0.7602 | 0.1441 |
| gradient_boosting_classifier | 0.04 | 0.4819 | 0.3442 | 0.8876 | 0.7361 | 0.1256 |
| knn_classifier | 0.01 | 0.3608 | 0.3025 | 0.7318 | 0.5208 | 0.1892 |
| goes_lstm | 0.36 | 0.1559 | 0.1205 | 0.6173 | 0.3718 | 0.2199 |
| climatology | 0.01 | 0.0000 | 0.0000 | 0.5000 | 0.2791 | 0.2091 |

**Real, headline finding**: on this genuinely paired, common-subset comparison, the
**persistence baseline still beats every trained model**, including both LSTMs and every
classical baseline — and its margin over the best LSTM (GOES+SHARP, TSS 0.555) is *larger*
here (+0.082) than on the two models' own separate maximum-available test sets (+0.011,
comparing the Appendix's 0.5656 vs the round-12 chapter's GOES+SHARP figure). This is not
an artefact of unequal samples — it is the same finding, confirmed on a fair footing.
Adding SHARP to the LSTM is real and large: GOES+SHARP (TSS 0.555) dramatically
outperforms GOES-only (TSS 0.156) on the identical instances. Random forest (TSS 0.544)
is statistically indistinguishable from the GOES+SHARP LSTM (see paired intervals below).

**Day-block bootstrap intervals** (`common_subset_bootstrap_intervals.csv`; 200 replicates,
90% interval, block = UTC calendar day):

| Model | TSS (90% CI) | ROC-AUC (90% CI) |
|---|---|---|
| persistence | 0.637 (0.604, 0.668) | 0.818 (0.802, 0.834) |
| goes_sharp_lstm | 0.555 (0.523, 0.587) | 0.854 (0.838, 0.868) |
| random_forest_classifier | 0.544 (0.518, 0.575) | 0.895 (0.881, 0.909) |
| logistic_regression | 0.483 (0.458, 0.513) | 0.908 (0.898, 0.920) |
| gradient_boosting_classifier | 0.482 (0.457, 0.510) | 0.888 (0.874, 0.902) |
| knn_classifier | 0.361 (0.321, 0.402) | 0.732 (0.710, 0.754) |
| goes_lstm | 0.156 (0.108, 0.196) | 0.617 (0.590, 0.639) |
| climatology | 0.000 (0.000, 0.000) | 0.500 (0.500, 0.500) |

**Paired TSS-difference intervals** (`common_subset_paired_tss_diff.csv`; same bootstrap,
paired — same resampled days used for both models in each replicate):

| Comparison | TSS diff | 90% CI | Excludes zero? |
|---|---|---|---|
| persistence − goes_lstm | +0.481 | (0.432, 0.520) | **Yes** — persistence significantly better |
| persistence − goes_sharp_lstm | +0.082 | (0.048, 0.113) | **Yes** — persistence significantly better |
| random_forest_classifier − goes_lstm | +0.388 | (0.338, 0.426) | **Yes** — random forest significantly better |
| random_forest_classifier − goes_sharp_lstm | −0.011 | (−0.033, 0.014) | **No** — not significantly different |
| goes_lstm − goes_sharp_lstm | −0.399 | (−0.439, −0.349) | **Yes** — adding SHARP significantly improves the LSTM |

**McNemar's test** (`common_subset_mcnemar.csv`; instance-level, continuity-corrected):
every one of the five pairs above is significant at $p<0.001$ (all five reported $p$-values
rounded to 0.0 in the raw output). **Real caveat, stated explicitly rather than presented
as exact**: this is a standard instance-level McNemar test, not a cluster-robust one —
Section 3.2.1's documented overlapping-window dependence (24h horizon, 12-minute stride)
means adjacent instances are not independent, so these $p$-values should be read as
anti-conservative (more significant than a cluster-correct test would report), consistent
with Section 3.8's own stated limitation about instance-level statistics under this
overlap structure. The day-block bootstrap intervals above, not the McNemar $p$-values,
are the more defensible uncertainty statement for this reason.

**Reliability (calibration), 10 equal-width bins, empty bins omitted**
(`common_subset_reliability_bins.json`) — one real finding worth stating in §3.8: the
**GOES-only LSTM is poorly calibrated at high confidence**, e.g. its $[0.9,1.0)$ bin
($n=41$) has mean predicted probability 0.906 but observed frequency only 0.146, and its
$[0.8,0.9)$ bin ($n=226$) predicts 0.851 but observes 0.385 — the model is confidently
wrong in its highest-probability decile. The GOES+SHARP LSTM's top bin is much better
calibrated ($n=4{,}801$, mean predicted 0.940 vs observed 0.798). Random forest and
gradient boosting are the best-calibrated models overall (e.g. random forest's
$[0.9,1.0)$ bin: predicted 0.936, observed 0.947); persistence has only two populated
bins ($[0,0.1)$ and $[0.9,1.0)$, both closely matching predicted-vs-observed, since its
score is not a smooth probability.

**Suggested drop-in paragraph for §3.8** (real numbers only, ready to adapt into the
chapter's prose):

> To address the difference in effective sample between model families (Table 3.6), all
> eight models were additionally re-scored on the intersection of their three test sets — a
> genuinely paired, identically-evaluated common subset of 37,598 instances spanning 1,628
> distinct UTC calendar days (bounded by the GOES+SHARP LSTM's stricter sequence-feasibility
> requirement). On this common subset, persistence remains the strongest model (TSS = 0.637,
> 90% day-block-bootstrap CI [0.604, 0.668]), ahead of the GOES+SHARP LSTM (TSS = 0.555, CI
> [0.523, 0.587]) and random forest (TSS = 0.544, CI [0.518, 0.575]); the paired
> day-block-bootstrap difference between persistence and the GOES+SHARP LSTM excludes zero
> (+0.082, CI [0.048, 0.113]), confirming this is not an artefact of the two models' unequal
> maximum-available samples. Random forest and the GOES+SHARP LSTM are statistically
> indistinguishable on this subset (TSS diff −0.011, CI [−0.033, 0.014]). Adding SHARP to the
> LSTM produces a large, significant improvement over the GOES-only LSTM (TSS diff −0.399 for
> GOES-only vs. GOES+SHARP, CI [−0.439, −0.349]). McNemar's test on paired thresholded
> decisions confirms significance for every pairwise comparison ($p<0.001$), though as an
> instance-level test on temporally overlapping windows this should be read as
> anti-conservative rather than exact; the block-bootstrap intervals above are the more
> defensible uncertainty statement. Reliability analysis on the same subset additionally shows
> the GOES-only LSTM is poorly calibrated in its highest-confidence decile (predicted ≈0.91,
> observed ≈0.15, $n=41$), a miscalibration not present in the GOES+SHARP variant.

### 12.10 Verification A — the 77-row test-set discrepancy, traced exactly

**[COMPUTED THIS SESSION]**, directly from `dataset_with_split.csv`: filtering to
`split == "test"` gives **211,671** rows — an exact match to Table 3.2 and
`run_summary.json`'s top-level `split_sizes.test`. Of those, **77** have `label_primary ==
NaN`; the remaining **211,594** have a defined label. `211,671 − 77 = 211,594`, an exact
accounting — all 77 rows are attributable to a single cause.

**All 77 rows are the tail of the dataset's forward-looking label window running out of
data, not embargo, not the imputer/scaler, and not a filter specific to the export
script**:

| Candidate cause | Rows attributable | Evidence |
|---|---|---|
| Undefined `label_primary` (labeling window exceeds data coverage) | **77 (100%)** | The 77 undefined-label test rows span `time_utc` **2025-07-24 04:48 UTC to 2025-07-24 23:48 UTC** — the last ~19.6 hours of the dataset. The dataset's own last timestamp is **2025-07-25 00:00 UTC**. A row issued at, say, 2025-07-24 23:48 needs flare-catalogue/GOES coverage out to 2025-07-25 23:48 (the 24h horizon) to determine whether `label_primary` is 0 or 1; that coverage does not exist past 2025-07-25 00:00, so `labeling/flare_labels.py` correctly leaves the label undefined (`NaN`) rather than fabricating a negative. This is the same "flares already in progress"/undefined-label mechanism already documented in Section 2 above, here manifesting at the dataset's own trailing edge rather than a split boundary. |
| Embargo-bucket rows | **0** | Embargo is a *separate* `split` value (`"embargo"`, real count **112** total across all boundaries, per `run_summary.json`'s `split_sizes.embargo`) — these rows are never labelled `"test"` in the first place, so they are already excluded from both the 211,671 and 211,594 figures; they cannot be part of this 77-row gap by construction. |
| Rows dropped by the median imputer or robust scaler | **0** | `SimpleImputer(strategy="median")` *fills* missing feature values in place; it does not drop rows. `apply_robust_scaling` clips and rescales values column-wise; it also does not drop rows. Neither function changes row count — confirmed by inspection of `preprocessing/scaling.py` and by the exported CSV's row count matching the post-dropna count exactly (see below). |
| Rows with NaN/invalid predicted probability after scoring | **0** | Every one of the 211,594 exported rows has a finite `predicted_probability`; `predict_proba` was never called on the 77 undefined-label rows in the first place, since they were removed before scoring, not after. |
| A filter specific to `export_baseline_predictions.py` | **0** | The script's only row-count-affecting operation is `dataset = dataset.dropna(subset=[TARGET_COL]).reset_index(drop=True)` (`export_baseline_predictions.py:53`) — the **exact same call**, on the **exact same column**, as the original training pipeline's `modeling/train.py::_prepare_frame` (`frame = dataset_with_split[cols].dropna(subset=[target_col]).reset_index(drop=True)`, `modeling/train.py:61`). This is why `run_summary.json`'s own `task_results.targets.label_primary.n_test` already reads **211,594** — the 77-row reduction happened in the *original* training/evaluation run, not something the export script introduced afterward. |

**Precise terminology to use in Chapter 3**:
- **"Raw eligible test instances" = 211,671`** — the count after chronological-split
  assignment and the 24-hour boundary embargo (Table 3.2's own reported figure;
  `run_summary.json`'s `split_sizes.test`).
- **"Successfully scored instances" (or "instances with a defined label") = 211,594`** —
  after additionally dropping the 77 rows whose forward-looking label window runs past the
  end of available data. This is the set every baseline/classical model was actually
  trained and evaluated against in the original run (`run_summary.json`'s
  `task_results.targets.label_primary.n_test`), not a number newly computed by this
  session's export. Table 3.2 should either footnote this 77-row difference explicitly or
  (preferably, for internal consistency) state 211,594 directly, since 211,594 is the
  number every downstream metric, threshold selection, and now this prediction export is
  actually computed from.
- **Common-subset membership** — **[VERIFIED THIS SESSION]** via an explicit
  timestamp-set intersection (not just a count comparison): the 37,598-row common subset
  is confirmed a **strict subset** of the 211,594-row baseline test set
  (`combo_ts.issubset(base_ts) == True`, `37,598 < 211,594`), and in fact the common
  subset's timestamp set is **identical** to the GOES+SHARP LSTM's own test set
  (`common_ts == combo_ts == True`) — confirming Section 12.9's statement that the
  GOES+SHARP LSTM's sequence-feasibility requirement is the binding constraint on the
  common subset, not an independent restriction.

### 12.11 Verification B — the bootstrap/McNemar implementation, quoted exactly

**[COMPUTED THIS SESSION]**, quoting `analyze_common_subset.py` directly (line numbers as
currently written):

| Question | Answer | Evidence |
|---|---|---|
| Bootstrap random seed | **44** — a single global seed, matching this project's seed convention (not a separate bootstrap-specific value). | `SEED = 44` (line 41); `rng = np.random.default_rng(SEED)` (line 217) — **one** `Generator` instance, created once and threaded through **all** 8 single-model bootstraps and all 5 paired-diff bootstraps in sequence. This means each model's/pair's 200 draws are a different, non-overlapping slice of the same deterministic RNG stream, not 13 independently-seeded runs — reproducible given the fixed seed and fixed call order, but the specific draws for `persistence`, say, would change if the loop order or model list were reordered. |
| Interval construction method | **Percentile** (plain empirical percentile of the replicate distribution), not basic (pivotal) and not BCa (bias-corrected-and-accelerated). | `lo, hi = np.nanpercentile(tss_reps, [(1-INTERVAL)/2*100, (1+INTERVAL)/2*100])` (lines 221–222, single-model); `lo, hi = np.percentile(diffs, [(1-INTERVAL)/2*100, (1+INTERVAL)/2*100])` (line 247, paired-diff). No bias-correction or acceleration term is computed anywhere in the script. |
| Are calendar days sampled with replacement? | **Yes.** | `sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)` (line 132, single-model; identically at line 153, paired-diff). |
| Days per replicate | **Exactly 1,628** — `size=len(unique_blocks)` (the real, observed number of distinct UTC calendar days in the common subset, Section 12.9) — every one of the 200 replicates, both single-model and paired. Because sampling is with replacement, a given replicate's *distinct* days are typically fewer than 1,628 (some days drawn more than once, some not at all), but the number of day-*draws* is fixed at 1,628 every time. | Same lines as above. |
| Undefined-metric handling (empty class in a resample) | **Inconsistent between the two bootstrap functions, but empirically inconsequential for this run — [VERIFIED THIS SESSION].** The single-model bootstrap (`block_bootstrap_single`) stores a raw `NaN` for any replicate where `tss_hss` returns `NaN` (i.e. the resampled block set contains zero positives or zero negatives — `tss_hss`'s own `pod`/`pofd` become `NaN` under `(tp+fn)==0` or `(fp+tn)==0`, `analyze_common_subset.py:93–96`) and then aggregates with `np.nanpercentile`, which **skips** `NaN` replicates. The paired-diff bootstrap (`block_bootstrap_paired_diff`) also stores raw `NaN`s when either side is degenerate, but aggregates with plain `np.percentile` (line 247), which does **not** skip `NaN` — a single `NaN` replicate would propagate `NaN` into *both* reported bounds. This is a real inconsistency in the implementation. **It had zero effect on the numbers already reported**: re-running the identical seeded single-model draws with explicit NaN-counting confirmed **0 of 200 replicates were `NaN` for every one of the 8 models** (persistence's ~28% base rate and 1,628 blocks make an all-one-class day-block resample practically impossible), and the paired-diff CIs already reported in Section 12.9 are all finite, which is only possible if none of their 200 draws were `NaN` either (plain `percentile` would have shown `NaN` bounds otherwise). If this analysis is ever rerun with a different subset, model, or seed, `block_bootstrap_paired_diff` should be changed to use `np.nanpercentile` for consistency and safety — noted here as a real, if currently inert, code fragility rather than silently left unstated. |
| Are thresholds held fixed across resampling? | **Yes, confirmed.** Each model's threshold is its already-recorded, validation-selected $\tau^*$ (`thresholds[name]`, sourced from `baseline_thresholds.json`/`goes_lstm_threshold.json`/`goes_sharp_lstm_threshold.json` — themselves copied unchanged from the original training run's `metrics.json`/`run_summary.json`), passed as a static argument into `tss_hss(..., threshold)` inside every bootstrap replicate. `modeling/metrics.py::select_threshold` (or any threshold search) is **never called** inside `block_bootstrap_single` or `block_bootstrap_paired_diff` — thresholds are never reselected per replicate. | `tss_hss(y_true, y_prob, threshold)` calls inside both bootstrap loops (lines 136, 156–157) always pass the same outer `threshold`/`thr_a`/`thr_b` parameter, sourced once, before the replicate loop begins. |

**McNemar specifics**:

| Question | Answer | Evidence |
|---|---|---|
| Exact (binomial) vs. asymptotic (chi-square) | **Asymptotic chi-square**, not the exact binomial version. | `stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)`; `p_value = 1 - chi2.cdf(stat, df=1)` (`analyze_common_subset.py`, `mcnemar()` function) — this is `scipy.stats.chi2`'s CDF, the standard asymptotic McNemar's test, not an exact binomial/sign test (which `statsmodels.stats.contingency_tables.mcnemar(..., exact=True)` would compute instead). |
| Continuity correction | **Yes, applied** — the `- 1` inside `(abs(n01 - n10) - 1) ** 2` before squaring is the standard Yates' continuity correction for McNemar's test. | Same line as above. |

**Exact statistic values for the two requested pairs** — **[COMPUTED THIS SESSION]**, with
precise (non-underflowed) $p$-values computed via `scipy.special.log_ndtr` in log-space
(plain `scipy.stats.chi2.sf` underflows to exactly `0.0` in float64 for both, which is why
the console output and `common_subset_mcnemar.csv` both show `p_value = 0.0` — a real
floating-point limitation, not a computation error):

| Pair | $n_{01}$ (A wrong, B right) | $n_{10}$ (A right, B wrong) | $\chi^2$ statistic | $p$-value |
|---|---|---|---|---|
| persistence (A) vs. GOES+SHARP LSTM (B) | 1,875 | 5,329 | **1655.08** | $\approx 10^{-361}$ (float64 underflows to 0.0) |
| random_forest_classifier (A) vs. GOES+SHARP LSTM (B) | 2,630 | 1,482 | **319.94** | $\approx 1.49\times10^{-71}$ (also underflows to 0.0 under `chi2.sf`, but recoverable in log-space) |

Both are significant at any conventional threshold by an overwhelming margin; the
$\chi^2$ statistics themselves (1655.08 and 319.94), not the uninformative rounded
$p=0.0$, are the figures worth quoting in Chapter 3 text, alongside the already-stated
caveat (Section 12.9) that this instance-level test is not cluster-robust given the
documented overlapping-window dependence.

---

## Appendix — Real results obtained with the values above (informational, for Chapter 6)

Not a Chapter 3 methodology value, but included since it is the direct, real output of
running the methodology with the parameters resolved above.

**Disk-integrated classical baselines** (`results/real_full_20100501_20250724/metrics.json`,
`label_primary`, real test-split TSS/ROC-AUC, threshold selected on validation only):

| Model | Test TSS | Test ROC-AUC | Threshold |
|---|---|---|---|
| climatology | 0.0000 | 0.5000 | 0.01 |
| persistence | 0.5656 | 0.7828 | 0.01 |
| logistic_regression | 0.3752 | 0.8770 | 0.16 |
| random_forest_classifier | 0.4459 | 0.8619 | 0.22 |
| gradient_boosting_classifier | 0.3751 | 0.8578 | 0.04 |
| knn_classifier | 0.2936 | 0.7115 | 0.01 |

**Real, worth stating plainly**: the simple **persistence baseline (TSS=0.5656) beats
every classical ML model AND the GOES-only sequence LSTM (TSS=0.1795, below) on this exact
same real task/target/split** — a genuine, non-obvious finding, not something to omit
because it is inconvenient for the deep-learning narrative.

**GOES-only sequence LSTM** (Table 3.3's real scope, `real_full_sequence_20100501_20250724`):
n_train=66,352 / n_val=66,029 / n_test=107,489 (the sequence-feasible subset — materially
smaller than Table 3.2's classical 237,879/77,100/211,671, a direct consequence of
Algorithm 3.1 step 9's strict "any flagged gap in the 30-step window → discard" policy).
Convergence: early-stopped at epoch 6. Test: **TSS=0.1795, ROC-AUC=0.6276, HSS=0.1314,
POD=0.6798, FAR=0.5004, precision=0.3205, Brier=0.2165**. Threshold $\tau^*=0.36$
(validation-selected).

**GOES+SHARP combined sequence LSTM** (scope caveat 6's closest-but-architecturally-different
real artifact, `real_full_sequence_goes_sharp_20100501_20250724`): n_train=28,719 /
n_val=21,179 / n_test=37,598. Test: **TSS=0.5552, ROC-AUC=0.8540**. A real, substantial
improvement over the GOES-only LSTM — plausible given SHARP's known flare-precursor
signal, not fabricated or cherry-picked — but not directly comparable to Table 3.3's
literal scope (2 vs. 20 input channels, input-level not representation-level combination).

**Per-HARP classical baselines** (`results/real_full_per_harp_20100501_20250724/metrics.json`,
`label_primary`; 3,941,133 total instances, 6,930 distinct HARPs, 246 features in
manifest; `label_primary`-defined split sizes n_train=1,770,323 / n_val=133,562 /
n_test=1,568,837):

| Model | Test TSS | Test ROC-AUC |
|---|---|---|
| climatology | 0.0000 | 0.5000 |
| persistence | 0.2928 | 0.6464 |
| logistic_regression | 0.7600 | 0.9563 |
| random_forest_classifier | 0.7702 | 0.9548 |
| hist_gradient_boosting_classifier | 0.7985 | 0.9623 |
| knn_classifier | 0.6218 | 0.8248 |

All of the above are single-seed point estimates (`seed: 44`), not averaged over multiple
seeds or bootstrap-resampled — the same real limitation `src/cross-domain`'s
CHAPTER_5_PARAMETER_VALUES.md already documents for the fusion results.
