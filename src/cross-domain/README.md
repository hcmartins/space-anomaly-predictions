# Cross-Domain Fusion — Chapter 5 Pipeline

Initial code structure for Chapter 5 ("Cross-Domain Fusion Methodology"):
representation alignment and late-stage fusion between the solar
(`src/solar`) and Martian (`src/martian`) forecasting pipelines.

**Status:** core fusion mechanics (encoders, projections, alignment loss,
adaptive weighting, balance regularization, the combined model, the
training objective and loop) are real and test-passing, verified line by
line against Chapter 5's own Eq. 5.1-5.9 and Algorithm 5.1 text.

**Both domains' encoders are now real, and as of 2026-08-14 the martian side is
also trained with a genuinely non-empty validation split — the first run in this
package's history where that is true.** The martian LSTM encoder is promoted from
a checkpoint trained by `src/martian`'s `mars.run_all` against real EMIRS/MCD data
(`outputs/real/run_20260813T195013457439Z_seed42`, k=2/m=3 primary regime, real
Table 4.6 catalogue validation, chronological-split leakage fix + sentinel-row
exclusion + missing-data sensitivity selection — see
`src/martian/CHAPTER_4_PARAMETER_VALUES.md` Section 13). This supersedes the
2026-08-05 run, whose validation window fell entirely outside MY36's real data
coverage and was silently empty (`configs/encoders.yaml`'s `martian.input_size`
stayed **36** across this update — the feature manifest is column-for-column
identical between the two runs, so the checkpoint promotion is architecture-safe).
All simulation-only references, stale file paths, and hardcoded assumptions from
the old simulated-martian era remain removed from this package's code, tests, and
docs.

**A real, end-to-end fusion training run exists against real data on both sides,
with genuine validation-gated early stopping for the first time**
(`run_fusion.py`, `outputs/run_20260814T064609079205Z/`). Real martian sequence
counts (window=3, horizon=1): **n_train=37,977, n_val=11,360, n_test=35,288** —
val is now real and non-empty (49 positives), not the 0-row artifact of the
pre-2026-08-14 split bug (train/test prevalence 4.18%/0.71%; val prevalence
0.43%). **Result: solar fused TSS = 0.4698 vs. unimodal baseline TSS = 0.5552 —
some real negative transfer remains, but nowhere near the pre-fix collapse.**
Martian (fused) TSS = 0.4536 vs. unimodal baseline 0.3019 — real **positive**
transfer for martian. The pre-fix 2026-08-05 run (solar fused TSS = 0.0000, a
severe negative-transfer finding) is now understood to have been a downstream
artifact of training with no validation-based model selection at all (fixed
5-epoch patience, no best-checkpoint restore — see "Real-data provenance" below)
rather than an architecture-level property; see `CHAPTER_5_PARAMETER_VALUES.md`'s
Appendix for the full before/after. Reported plainly either way — see
`final_report.md` for full per-domain figures.

**All 7 A1-A7 ablation configurations re-run against real data on both sides,
with real validation-gated early stopping** (`run_ablation.py`,
`outputs/ablation_run_20260814T064653396282Z/`). Real TSS by config (solar/martian):
A1=0.5552/0.3019, A2=0.4704/0.4364, A3=0.5507/0.4482, A4=0.4698/0.4536,
A5=0.5429/0.4211, A6=0.2892/0.4239, A7=0.4698/0.4536. **The corrected real
finding supersedes the earlier "fusion always collapses solar" conclusion**: A6
(no alignment loss) has markedly lower solar TSS (0.2892) than every other
fusion-pathway config (A3/A4/A7 all ≈0.47-0.55), showing the moment-alignment term
(Eq. 5.4) is doing real, substantial work for solar stability under fusion — the
opposite emphasis from the pre-fix conclusion, which (incorrectly, as an artifact
of unvalidated training) attributed all the damage to the mixing step itself.
Martian TSS is stable and above its unimodal baseline across every fusion-pathway
config (0.42-0.45 vs. 0.30), a real, robust positive-transfer finding. The
A4≡A7 batch-mean-fusion equivalence (`fuse()` reduces the other domain to its
batch mean, so shuffling row order never changes the result) still holds exactly
under the corrected run (0.4697567705865804 solar / 0.4535760949735148 martian
for both) — this property is architecture-inherent, independent of the
validation-split fix.

**Section 5.6.1's uncertainty protocol (seed variation + block bootstrap + paired
comparison) is now real, built, and executed against 5 real seeds** (`uncertainty_analysis.py`,
`run_uncertainty_sweep.py`, `outputs/uncertainty_sweep_20260818T090949043479Z/`).
**The headline finding changes the single-seed reading above**: the configurations
that route through the real attention-mixing pathway (A3/A4/A7 — architecturally
identical per the A4≡A7 property) show far larger seed-to-seed TSS variance than any
non-mixing configuration, especially on martian (sd≈0.22 across 5 seeds, ranging from
≈0 to 0.45, versus ≤0.09 for A1/A2/A5/A6). The single-seed "martian positive
transfer" result above was real at `FUSION_SEED=44` but does not replicate at 3 of
the other 4 seeds. See `CHAPTER_5_PARAMETER_VALUES.md`'s Appendix for the full
numbers and reading — the mixing operator should now be read as *unstable*, not just
as harmful-on-average for solar.

**The negative-transfer/failure-case presentation layer is now real**
(`evaluation/presentation.py`, 2026-07-29 — figures + stamped markdown
tables, following `src/martian/data_sources/provenance.py`'s real
stamping convention). "Chapter 6/7" remains this package's own invented
label (no real thesis numbering scheme exists anywhere in this repo to
inherit) — the report says so plainly.

## Real-data provenance — read before citing any result from this package

- The **solar** encoder (`encoders/solar_encoder.py`) loads a REAL LSTM
  checkpoint
  (`src/solar/checkpoints/real_full_sequence_goes_sharp_20100501_20250724/lstm_label_primary.pt`)
  trained on the real GOES irradiance sequence AND the real 18-keyword
  SHARP magnetic-descriptor sequence (`input_size=20` — matching Chapter
  5's own description of $X_S$, closing the earlier GOES-only gap; see
  `src/solar/preprocessing/sharp_sequences.py`) under the real Cycle 24
  (train)/Cycle 25 (test) split. The earlier GOES-only checkpoint
  (`real_full_sequence_20100501_20250724/`, `input_size=2`) is left on
  disk, unmodified, for comparison — real test TSS improved from 0.179
  (GOES-only) to 0.555 (GOES+SHARP).
- The **martian** encoder (`encoders/martian_encoder.py`) loads a REAL LSTM
  checkpoint (`src/martian/checkpoints/task_storm_detection/lstm_w3_h1.pt`,
  promoted 2026-08-14 from `outputs/real/run_20260813T195013457439Z_seed42`)
  trained by `src/martian`'s `mars.run_all` against real EMIRS L3
  retrievals and a real Montabone et al. dust climatology — not simulated
  data (see `src/martian/README.md` and `CHAPTER_4_PARAMETER_VALUES.md`).
  `input_size=36` (real, dust-only feature set — down from the old
  simulated checkpoint's 61, which included a `surface_temperature_k`
  feature family no real climatology source supports). **RESOLVED
  (2026-08-14):** the earlier limitation here — martian's validation split
  having 0 sequence-feasible windows, because `src/martian`'s validation
  window fell entirely outside MY36's real data coverage — was a real bug
  in the upstream split, since fixed. Martian's validation split now has
  **11,360** sequence-feasible windows (49 positives) for this
  window/horizon, and Stage 2 fusion training now genuinely early-stops on
  real validation performance (`early_stopped_epoch_6` for the A7 run,
  varying per ablation config: 6/6/28/6/6/14/10) rather than always hitting
  the old fixed `early_stopped_epoch_4_no_validation_selection_available`
  floor. This is stated plainly in every report this package produces,
  never hidden.

## Why encoders are self-contained, not imported from solar/martian

`src/solar` and `src/martian` are each independently-runnable packages
using unprefixed intra-package imports (`from preprocessing.X import Y`)
and identically-named internal subpackages (`modeling/`, `preprocessing/`,
`data_sources/`). Adding both package roots to `sys.path` at once would
silently collide. `encoders/frozen_encoder.py` and its two subclasses
instead reconstruct just the minimal architecture needed to load each
real checkpoint's `state_dict` directly — a small, deliberate,
documented duplication of the relevant slice of each package's
`modeling/sequence_models.py::SequenceModel`, not a refactor of either
tested package. This also matches Chapter 5's own framing: encoders are
frozen, deterministic maps during fusion training — treating them as
opaque pre-trained artifacts is the right abstraction on its own merits.

## Quick start

```bash
cd src/cross-domain
pip install -r requirements.txt
python -m pytest -q

# Real, end-to-end fusion training run (A7 "full_model" -- exports real
# solar/martian sequence data automatically if not already present, then
# trains + evaluates for real):
python run_fusion.py

# Real A1-A7 ablation comparison (trains/loads all 7 configurations --
# takes several minutes, ~30s per fusion-model variant):
python run_ablation.py

# Section 5.6.1's uncertainty protocol: seed variation (>=5 seeds),
# block bootstrap (B=1000, domain-specific block length), paired
# comparison -- retrains the ablation across 5 seeds in run_ablation.py's
# light mode (no duplicate model checkpoints/figures per seed) and writes
# outputs/uncertainty_sweep_<id>/uncertainty_report.md. Takes roughly as
# long as 5x a single run_ablation.py call.
python run_uncertainty_sweep.py
```

`run_fusion.py` calls `src/solar/export_fusion_sequences.py` and
`src/martian/export_fusion_sequences.py` (each run as its own subprocess,
in its own package's directory) to regenerate the real sequence-tensor
arrays neither package persists on its own — see those scripts' own
docstrings for exactly which already-tested library functions they reuse.

## Layout

```
configs/      encoders.yaml, fusion.yaml, ablation.yaml -- every numeric
              choice Chapter 5 leaves as [INSERT]/[CONFIRM], documented
              as an explicit engineering decision
encoders/     frozen, checkpoint-loading wrappers producing h_S / h_M (Eq. 5.1)
fusion/       projections (Eq. 5.2), alignment loss (Eq. 5.3-5.4),
              adaptive weighting (Eq. 5.5-5.6), balance regularizer
              (Eq. 5.7), the combined CrossDomainFusionModel (with real
              A1-A7 ablation hooks -- use_fusion_pathway/
              other_domain_embedding), and unimodal.py's new
              CapacityMatchedUnimodalModel (A2)
training/     the total objective (Eq. 5.8), Algorithm 5.1's staged
              training loop (Section 5.5.1, train_fusion.py, now
              ablation-aware), and train_unimodal.py (A2's training loop)
evaluation/   negative-transfer analysis (Section 5.6.2), the A1-A7
              ablation registry (Section 5.6.1), failure-case
              cataloguing (Section 5.6.3), real unimodal baselines
              (unimodal_baseline.py), TSS-maximizing threshold selection
              (thresholds.py), the shared per-domain evaluation core
              (comparison.py, used by both run_fusion.py and
              run_ablation.py), and the real figure/table/report
              presentation layer (presentation.py)
fusion_data.py  shared real-data export/loading (both entry points)
run_fusion.py   the real, end-to-end A7 Stage-2 training + evaluation
                entry point (writes outputs/<run_id>/final_report.md +
                negative_transfer_and_failure_report.md)
run_ablation.py the real A1-A7 ablation comparison entry point (writes
                outputs/ablation_run_<id>/ablation_report.md; accepts
                --seed/--predictions-dir/--light for reuse by the
                uncertainty sweep below)
uncertainty_analysis.py   Section 5.6.1's protocol (seed variation, block
                bootstrap, paired comparison) over predictions_*.csv files
                -- no training, pure statistics; domain-specific block
                lengths (solar=120 rows=24h at 12min cadence, martian=1
                row=1 sol)
run_uncertainty_sweep.py  drives run_ablation.py across 5 real seeds in
                light mode, collects predictions into one shared
                outputs/uncertainty_sweep_<id>/, and runs
                uncertainty_analysis.py against them
tests/        hand-computed unit tests + real integration tests that
              load the real checkpoints and run real forward passes
```

## What is NOT (yet) implemented

- Only martian's `storm_detection` task is wired into fusion, not
  `dust_regression` (`src/martian`'s other real Chapter 4 task) — the
  martian encoder's checkpoint is specific to storm detection, window=3
  sols, horizon=1 sol.
- Figure 5.2 (mutual-information/partial-correlation diagnostics) does not
  exist — no such analysis has been built anywhere in this repository.
- Chapter 3.6's fuller solar stack (CNN-LSTM, FC-SHARP, Transformer,
  within-solar multimodal fusion) — the solar encoder here is a single
  LSTM over the concatenated GOES+SHARP sequence (see above), not the
  complete multi-branch Chapter 3.6 design.

Resolved gaps kept here only as historical pointers, not current limitations:
paired-bootstrap confidence intervals (Section 5.6.1) are now real, built, and
executed against 5 seeds — see `CHAPTER_5_PARAMETER_VALUES.md`'s Appendix, not
this list; `ablation_report.md` itself still reports single-seed point estimates
only (that's a separate, cheaper run from the uncertainty sweep, by design).
Martian's real validation split is no longer empty (see "Real-data provenance"
above, RESOLVED 2026-08-14) — an earlier version of this bullet list said
otherwise and was stale.
