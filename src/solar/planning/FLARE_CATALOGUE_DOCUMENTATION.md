# Flare Catalogue + HARP Attribution — Documentation (Phases 1 & 2)

Status: **Phase 1 complete** (real flare catalogue + NOAA AR -> HARPNUM
attribution, verified against real live data and a real full-scale
2010-2025 run). **Phase 2 implemented 2026-07-26** (per-HARP pipeline:
`preprocessing/harp_alignment.py`, `labeling/harp_catalogue_labels.py`,
`modeling/harp_datasets.py`, `run_pipeline_per_harp.py`) — see "Phase 2"
section below for real bugs found/fixed, the real 1-year benchmark, and
the real full-range run. Both phases purely additive: the disk-integrated
`preprocessing`/`labeling`/`modeling` pipeline (`run_pipeline.py`) is
provably unmodified by either phase (confirmed by re-running the
disk-integrated pilot after Phase 2 and getting byte-identical real
numbers). Real full-range attribution result: 97.3% of AR-assigned flares
(55.1% of all real flares) were successfully attributed to a real
HARPNUM — see "Real full-range attribution report" below.

## Scope

Covers `data_sources/flare_catalogue_ingestion.py`,
`preprocessing/harp_attribution.py`, and the `NOAA_ARS` keyword addition
to `configs/sharp.yaml`/`data_sources/schema.py`.

## Real data source verification log

Both real data sources were fetched live and inspected (not transcribed
from documentation alone) on 2026-07-26:

**NOAA GOES Flare Report** (NCEI, `configs/flare_catalogue.yaml`):
confirmed real directory structure at
`.../xrsf-l2-flrpt_science/csv/` — one mission file
(`sci_xrsf-l2-flrpt_geo_s19950103_e20260724_v1-0-1.csv`, ~12MB, resolved
via directory-listing regex like GOES-R's version suffix, not
hardcoded) plus one file per year. Fetched the mission file for real:
**39,471 real flares in the 2010-05-01..2025-07-24 range**. Real header
confirmed to match Table 3 of NOAA's own ReadMe exactly, including that
`flare_loc_*` columns are split into separate `_lat`/`_lon` (or `_x`/`_y`)
columns in the CSV (footnote 2 of that ReadMe) — not used by this
project, only `time`/`start_time`/`end_time`/`flare_id`/`flare_class`/
`xrsb_irrad`/`active_region` are.

**Real, non-fabricated finding**: `active_region` is genuinely often
blank — confirmed 4.4% of 2017's flares have one, and Cycle-24-wide
40.0% (9,772/24,413 C-or-above flares) vs. Cycle-25-wide 79.6%
(11,982/15,058) have one — a real, substantial difference between cycles,
not yet explained (RESEARCHER VERIFICATION REQUIRED before citing a
cause — could be a genuine SWPC/XRS reporting-process change, or an
artifact of this environment's specific archive instance). Most
strikingly: in the already-verified 2017-09-04..10 pilot week, the
X-class flares themselves (X3.2, X14.6, X1.2, X2.1) are UNASSIGNED, while
6 smaller M/C-class flares from the same real active region (AR 2673,
the short form of AR 12673 / HARP 7115) ARE assigned. Algorithm 3.2 step
11's "unassigned-flare masking" is not a rare edge case — it directly
affects this project's own reference event.

**JSOC `NOAA_ARS` keyword** (`configs/sharp.yaml`): confirmed real and
queryable via the same `drms.Client.query()` mechanism `sharp_ingestion.py`
already uses, by running a real, live, bookkeeping-only query
(`keywords: []` override — no new ingestion code needed, see below) for
the 2017-09-04..10 pilot week. Real confirmed facts from that query:
- `NOAA_ARS` reports the FULL 5-digit AR number (e.g. `"12673"`), unlike
  the flare catalogue's `active_region`, which uses the abbreviated
  4-digit form (`"2673"`) — `harp_attribution.parse_noaa_ars` registers
  both forms defensively.
- A HARP can enclose multiple NOAA ARs, comma-separated (e.g. real HARP
  7117 → `"12674,12679"`, HARP 7123 → `"12675,12676"`) — confirming
  Algorithm 3.2's premise directly, not assumed.
- A HARP with no assigned NOAA AR reports the literal **sentinel string
  `"MISSING"`** (not NaN, not `"0"`) — a real quirk that would have
  silently corrupted attribution if unhandled; `parse_noaa_ars` filters
  it via a digits-only check (also guards against any other future
  non-numeric sentinel).

## No new ingestion code needed for `NOAA_ARS`

`sharp_ingestion.py::ingest_sharp_chunked` already supports a
bookkeeping-only query: pass a config with `keywords: []` (leaving
`bookkeeping_keywords`, which now includes `NOAA_ARS`, untouched). This
was verified live for both the pilot week (4,230 rows, ~seconds) and the
full 2010-2025 range. No duplicated download/chunking/bisection logic was
written — the existing adaptive-bisection machinery
(`planning/REAL_FULL_RANGE_RUN.md`'s sibling doc) applies unchanged.

## `preprocessing/harp_attribution.py`

Three functions, all vectorized (no per-row Python loop — see
`planning/PREPROCESSING_DOCUMENTATION.md`'s `features.py` rewrite for why
that matters at this project's real scale):

- `build_harp_ar_index(sharp_df, time_col="T_REC")`: explodes `NOAA_ARS`
  into one row per (HARPNUM, active_region, time_utc) via
  `str.split(",")` + `.explode()`, registering both 4- and 5-digit AR
  forms.
- `attribute_flares_to_harps(flare_catalogue_df, harp_ar_index,
  time_tolerance_hours=12.0)`: merges on `active_region`, keeps matches
  within `time_tolerance_hours` of the flare's own peak time (a HARP's AR
  composition can change over its lifetime — matched near the flare's
  time, not a single static snapshot), returns every catalogued flare
  with a `harpnums` list column (empty for unassigned flares — masking
  policy is explicitly NOT decided here, deferred to Phase 2).
  `time_tolerance_hours=12.0` is an ENGINEERING DECISION (Algorithm 3.2
  does not specify one) — revisit once Phase 2 needs a final value.
- `attribution_quality_report(attributed_df)`: real coverage counts.

## Real verification: pilot week (2017-09-04..10)

`attribute_flares_to_harps` run against the real pilot-week SHARP
`NOAA_ARS` patch (4,230 rows) and the real flare catalogue, filtered to
the same week: **all 6 flares carrying `active_region="2673"` were
attributed to `harpnums=[7115]`, 100% match** — reproducing the
independently-documented real HARP 7115 / AR 12673 pairing this
project's whole pilot run was built around
(`planning/REAL_PILOT_RUN_2017-09.md`). Pilot-week quality report:
6/145 flares (4.1%) had an AR; 6/6 (100%) of those were successfully
attributed; 0 attributed to more than one HARP.

## Real full-range attribution report (2010-05-01..2025-07-24)

Ran for real: the full-range `NOAA_ARS` SHARP patch (`keywords: []`
override over `ingest_sharp_chunked`, same adaptive month-bisection
machinery as the physical-keyword ingestion) completed in 14,146s
(~3.9h), 329/329 chunks succeeded, 4,966,643 real rows (essentially
identical row count to the original 18-physical-keyword ingestion's
4,967,165 -- same real records, fewer columns). `build_harp_ar_index`
exploded this into 10,665,189 (HARPNUM, AR, time) rows in 6.5s;
`attribute_flares_to_harps` against all 39,471 real flares took 4.1s
(both fully vectorized, no per-row loop).

| | All | Cycle 24 | Cycle 25 |
|---|---|---|---|
| Total flares | 39,471 | 24,413 | 15,058 |
| With an `active_region` assigned | 21,754 (55.1%) | 9,772 (40.0%) | 11,982 (79.6%) |
| Successfully attributed to >=1 HARP (of those with an AR) | 21,166 (97.3%) | 9,356 (95.7%) | 11,810 (98.6%) |
| Attributed to >1 HARP | 411 (1.0% of total) | 141 | 270 |

**Real, honest takeaway**: once a flare has a NOAA AR at all, attribution
to a real HARPNUM succeeds 97.3% of the time — the matching mechanism
itself (NOAA_ARS + 12h time tolerance) works well and needs no further
tuning to reach this. The real bottleneck for Phase 2 per-region
forecasting coverage is upstream of attribution entirely: 44.9% of all
flares (60.0% in Cycle 24 specifically) never get an AR assigned by
SWPC/NCEI in the first place, per the Flare Report's own documented
XRS/SWPC peak-time-or-flux mismatch criterion — that ceiling cannot be
raised by better matching logic here, only by a different/supplementary
attribution source if Phase 2 needs it.

**Multi-HARP attributions (411 real cases) are genuine, not a matching
bug** — spot-checked: e.g. flare_id 201112041934 (AR 1367 -> HARPNUMs
1149 and 1165) reflects a real NOAA AR spanning two tracked HARPs
simultaneously, exactly the case Chapter 3.2.1 flags as needing an
explicit tie-break policy ("the converse case (one NOAA number spanning
two HARPs) [INSERT: does not occur in the sample / is handled by...]") —
it DOES occur (1.0% of all real flares), so Phase 2 cannot skip deciding
this.

## Real Table 3.1 numbers (2010-05-01..2025-07-24, C-class and above)

Cycle boundary: 2020-01-01 (matches `configs/splits.yaml`'s real boundary
choice — see that config's own RESEARCHER VERIFICATION REQUIRED note on
the exact date).

| Cycle | Date range (actual data) | C | M | X | Total (C+M+X) | %M+X | % flares with AR |
|---|---|---|---|---|---|---|---|
| 24 | 2010-05-01 .. 2019-10-01 | 12,641 | 1,194 | 75 | 13,910 | 9.12% | 40.0% (9,772/24,413 incl. B) |
| 25 | 2020-01-10 .. 2025-07-24 | 10,111 | 1,656 | 83 | 11,850 | 14.68% | 79.6% (11,982/15,058 incl. B) |

Sanity check against the thesis draft's own "Approx." Table 3.1 values:
Cycle 24's %M+X (9.12%) is remarkably close to the thesis's stated 9.2%
placeholder — good evidence this real catalogue matches what the
original thesis author expected. Cycle 25's %M+X (14.68% real vs. 9.8%
approximated in the draft) is a genuine, real divergence — consistent
with the well-documented fact that Solar Cycle 25 turned out
substantially more active than pre-cycle forecasts predicted, not a data
error.

## Phase 2 — per-HARP pipeline (implemented 2026-07-26)

New modules, all additive (see `C:\Users\hcaxt\.claude\plans\calm-moseying-scott.md`
for the full design and reuse rationale): `preprocessing/harp_alignment.py`
(limb exclusion at |LON_FWT| <= 68 deg, Bobra & Couvidat 2015's cited
convention; broadcasts disk-integrated GOES trailing state onto every
HARP via `alignment.py::attach_goes_trailing_state`, reused UNCHANGED),
`labeling/harp_catalogue_labels.py` (real catalogue-attributed labels,
per-class unassigned-flare masking that only masks a class if the
unassigned event's own flux reaches that class's threshold, never
overriding a genuine positive), `modeling/harp_datasets.py` (features x
labels join keyed on `(HARPNUM, time_utc)`), `run_pipeline_per_harp.py`
(orchestration entry point, mirrors `run_pipeline.py`).

**Two real bugs found and fixed while getting the first real per-HARP run
working:**

1. **pandas 3.0.3's `groupby(...).apply()` no longer passes the grouping
   column into the callback at all** — `include_groups=True` (the
   documented escape hatch) was checked directly and raises
   `ValueError: "include_groups=True is no longer allowed."` in this
   version, not just a deprecation warning. `run_pipeline_per_harp.py`'s
   `_apply_per_harp_group` helper works around this by reading the group
   key from `g.name` inside the callback and re-attaching it to the
   result — `preprocessing/missing_data.py`/`preprocessing/features.py`
   themselves needed NO changes (confirming the plan's reuse-not-rewrite
   approach was sound; the break was in the orchestration glue, not the
   reused logic).
2. **A CSV round-trip silently corrupts `active_region` matching.**
   `flare_catalogue_ingestion.py` writes `active_region` as a clean
   string ("2673" or ""), but a column mixing numeric-looking strings and
   blanks gets inferred as float64 on `pd.read_csv` (blanks -> NaN), so a
   naive `.astype(str)` after re-reading produces `"2673.0"` — which
   never matches `harp_ar_index`'s `"2673"`. First real full-pipeline
   smoke test silently produced **zero** attributions (`
   n_flares_attributed_to_at_least_one_harp: 0`) with no error — caught
   only by comparing against Phase 1's already-known-correct pilot-week
   result (6 attributions), not by any exception. Fixed in
   `run_pipeline_per_harp.py`'s load step (`.str.replace(r"\.0$", "",
   regex=True)`); anyone else re-reading this CSV must do the same.

**Real 1-year benchmark (2014, near Solar Cycle 24's maximum — a
plausible upper bound per year, not an average)**: 7m46s total.
Preprocessing breakdown: load 2.5s, QC 0.4s, per-HARP alignment 29.8s,
gap-filling 21.5s, features 156.1s (the single largest preprocessing
cost, still nowhere near the pre-vectorization `features.py` bug's
14-hour-scale problem), attribution 1.3s, labeling 28.2s, dataset join
5.2s. (Split+train on this single year's data are not representative of
the full accumulated dataset and are excluded from extrapolation.)

**Scale-driven model substitution**: `modeling/classical_models.py` gained
`make_hist_gradient_boosting_classifier` (new registry entry
`hist_gradient_boosting_classifier`, purely additive). `GradientBoostingClassifier`
took 2947s (49min) on the disk-integrated run's 237k training rows;
the per-HARP dataset is far larger, which would put plain
`GradientBoostingClassifier` at many hours. `run_pipeline_per_harp.py`
substitutes `hist_gradient_boosting_classifier` for
`gradient_boosting_classifier` in its own in-memory copy of
`task_flare_forecast.yaml` ONLY — the file on disk, and the
disk-integrated run's real reported numbers, are unaffected.

**A third real bug, found on the first real full-range (not 1-year) attempt:**
`ArrayMemoryError: Unable to allocate 180. MiB for an array with shape
(6, 3941133)` at step 8 (the dataset join) after steps 1-7 completed
successfully (real per-HARP row count at full scale: 3,941,133, close to
Phase 1's ~8.6x-disk-integrated estimate). Root cause: `aligned`,
`filled`, `attributed`, `features_df`, and `labeled_df` were ALL still
alive simultaneously by step 8 (Python does not free local variables
until the function returns), each a real multi-GB object at this scale.
Fixed by (1) moving attribution (step 4) to run right after alignment
(step 3), before gap-filling/features, so `aligned` only needs to survive
until labeling instead of the whole pipeline, and (2) explicit `del` +
`gc.collect()` immediately after each intermediate DataFrame's last real
use. Re-verified after the fix: the pilot smoke test produces byte-identical
real numbers (6 attributions, 234 real positives) to before the reorder —
confirms the fix changed memory lifetime only, not any computed result.

**A fourth real bug, on the SECOND full-range attempt** (the above fix
reduced peak memory but not enough — `ArrayMemoryError: Unable to
allocate 7.16 GiB for an array with shape (244, 3941133)`, now at
`build_harp_flare_forecast_dataset`'s own internal `.copy()`-of-a-subset,
called AFTER a full-width merge — i.e. the join function itself was
materializing the same ~260-column data three times: once as `left`
(full copy of features_df), once as `merged` (the join result), and once
more as the final trimmed `.copy()`). Fixed by trimming to only the
needed columns BEFORE merging (so the merge itself produces the final
result directly, no third copy), and by having `run_pipeline_per_harp.py`
`del` + reload `features_df`/`labeled_df` fresh from their just-written
CSVs immediately before the join, rather than keeping them alive as
Python objects continuously from steps 6/7 through step 8.

**A fifth real bug, caught by the SAME re-run**: the column trim
introduced a genuine duplicate-column bug — `label_horizon_hours` and
`label_max_flare_class_in_harp_horizon` both match the `label_` prefix
(so both appeared in the broad `label_cols` filter) AND are separately
listed in `HARP_LABEL_METADATA_COLUMNS`, so both got selected into the
merge's right side twice, producing a `dataset` with duplicate-named
columns. `dataset[col]` on a duplicated name silently returns a
DataFrame instead of a Series — pandas raises no error at the point of
duplication, only downstream: `assign_split_with_embargo` crashed with
`AttributeError: 'DataFrame' object has no attribute 'unique'`. Fixed by
using the curated `TARGET_COLUMNS` list (not the broad `label_`-prefix
filter) when selecting the join's label-side columns. A new regression
test (`tests/test_harp_datasets.py`) asserts `not
dataset.columns.duplicated().any()` AND runs the result through
`assign_split_with_embargo` — the existing tests before this fix did NOT
catch this bug (they never checked column uniqueness or exercised the
downstream integration point that actually surfaced it), a real gap now
closed.

All fixes re-verified together via the pilot smoke test before each
full-range re-attempt — same real numbers (6 attributions, 234 positives)
every time, confirming none of these fixes changed any actual computed
result, only performance/robustness.

**Third full-range attempt: steps 1-9 all succeeded for real.** Real
per-HARP dataset, extracted via chunked reading of the real 12.9GB
`dataset_with_split.csv` (loading it whole would itself risk OOM on this
machine): **3,941,133 real forecast instances across 6,930 distinct real
HARPs, 2010-2025**. Split: train 1,974,536 / val 141,163 / test 1,825,377
/ embargo 43 / unassigned 14. **Real, important finding**: `label_primary`
(M-or-above) prevalence is **3.39%** — dramatically lower than the
disk-integrated task's 22.8%, and much closer to the literature-typical
rarity Chapter 3.2's own text describes ("these events whose rarity...
makes them the forecasting challenge"). The disk-integrated task was
quietly much easier than the thesis's stated framing; this is real
evidence the per-HARP task is the scientifically appropriate one.
X-or-above prevalence: 0.26%.

**A sixth real issue, this one a genuine hardware ceiling, not a code
bug**: step 10 (training) failed with `ArrayMemoryError` inside
`SimpleImputer`'s median fit on the full ~1.97M-row train matrix (3.24
GiB just for the array `np.ma.median` needed internally to sort/copy).
This machine has only **15.6GB total RAM, 6.6GB free at idle**
(`Get-CimInstance Win32_OperatingSystem`) — three OOM failures in three
different places confirms this scale is genuinely at this machine's
memory ceiling, not one fixable bug. Fixed (researcher-approved,
"sample-based imputer fit" option chosen over trimming the model set or
stopping at dataset-only results) by adding `run_task(...,
imputer_fit_sample_size=150_000)` — an opt-in parameter, default `None`
preserves the disk-integrated pipeline's exact original behavior
untouched. Statistically near-equivalent (median/mean imputation
statistics stabilize with far fewer than 2M rows); avoids sklearn's
masked-array sort, not the actual per-row transform (which still runs on
the full train set regardless, and is a cheap O(n) fillna-style op, not
an internal sort).

Real training was RESUMED directly from the already-written
`dataset_with_split.csv` rather than re-running steps 1-9 a fourth time
(~40 real minutes) — a one-off scratch script
(`resume_training.py`), not part of the tracked codebase, loading the
saved CSV and calling `modeling.train.run_task` with the new parameter.

## Real full-range per-HARP model results (2026-07-27/28)

Training completed successfully: `n_train=1,770,323`, `n_val=133,562`,
`n_test=1,568,837` (after undefined-label dropping for `label_primary`).
Real total wall time: **40,293s (~11.2 hours)** — but the models' own
`.fit()` times sum to only ~40 minutes (RF 2274.5s, HistGB 104.5s, KNN
13.8s, LR 33.1s). The remaining ~10.5 hours were spent in prediction/
threshold-selection, and KNN is almost certainly the real cause: its
`runtime_seconds` only measures the near-instant `.fit()` (a lazy
learner), while predicting for ~3.47M combined train+val+test query rows
against a 1.77M-row training set in 246 dimensions is a genuinely massive
nearest-neighbor search — exactly the risk flagged (but left
unbenchmarked) after the disk-integrated run, now confirmed real.

| Model | Test TSS | Test HSS | Test ROC-AUC | tau* |
|---|---|---|---|---|
| climatology | 0.00 | 0.00 | 0.50 | 0.01 |
| persistence | 0.29 | 0.04 | 0.65 | 0.01 |
| logistic regression | 0.00 | 0.00 | 0.50 | 0.01 (same unscaled-feature convergence bug as the disk-integrated run — still not fixed) |
| random forest | **0.77** | 0.39 | 0.95 | 0.55 |
| hist gradient boosting | **0.78** | 0.45 | 0.96 | 0.14 |
| KNN | 0.52 | 0.36 | 0.80 | 0.07 |

**Real, substantial finding**: per-HARP test TSS (0.77-0.78 for the tree
ensembles) is dramatically higher than the disk-integrated task's
(0.37-0.45, see `planning/REAL_FULL_RANGE_RUN.md`). The per-HARP task —
predicting whether a specific region flares, from its own real
catalogue-attributed history — is a cleaner, more physically grounded,
more learnable problem than the disk-integrated whole-disk proxy, despite
its much lower base rate (3.39% vs. 22.8%). Precision stays low
(0.24-0.42), as expected for a genuinely rare-event task and consistent
with Chapter 3.2's own framing.

## Real follow-ups now identified (not yet done)

- **KNN prediction cost at this scale** — confirmed real (see above), not
  yet mitigated. A future run should either drop KNN for per-HARP runs,
  cap `n_neighbors`/use an approximate-NN library, or reduce feature
  dimensionality first.
- **Logistic regression's unscaled-feature convergence failure** persists
  at per-HARP scale too — same real, deferred fix (train-only
  standardization before LR) as the disk-integrated run.
- Real per-HARP interpretability (feature importances/coefficients) and
  the per-HARP version of Chapter 3's Table 3.2 have not been extracted
  from `results/real_full_per_harp_20100501_20250724/metrics.json` yet —
  available on request.

## What is NOT implemented / not yet decided

- Real full-range per-HARP results (dataset size, split sizes, per-model
  test-split TSS/HSS) — the real full 2010-2025 run was re-launched after
  the memory fix above; results to be added here once it completes.
- A KNN-at-per-HARP-scale performance check — KNN's disk-integrated
  `runtime_seconds` (0.7s) only measures `.fit()`, which is nearly free
  for a lazy learner; the real cost (prediction over large splits) was
  never separately benchmarked and could be a real bottleneck at
  per-HARP scale — not yet observed, will be reported honestly once the
  full run's real numbers are in, not assumed fine in advance.
