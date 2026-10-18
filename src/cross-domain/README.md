# Cross-Domain Fusion — Chapter 5 Pipeline

Representation alignment and attention fusion between the frozen solar
(`src/solar`) and Martian (`src/martian`) forecasting encoders: projections
(Eq. 5.2), moment-alignment loss (Eq. 5.3-5.4), adaptive weighting
(Eq. 5.5-5.6), balance regularisation (Eq. 5.7), the total objective
(Eq. 5.8) and Algorithm 5.1's staged training, with the A1-A7 ablation
(Section 5.6.1), negative-transfer analysis (Section 5.6.2) and failure-case
cataloguing (Section 5.6.3).

**Canonical results:** the 5-seed uncertainty sweep
`outputs/uncertainty_sweep_20261004T092140396722Z/` (seeds 44-48, report in
`uncertainty_report.md`) and its per-seed ablation runs
`outputs/ablation_run_20261004T09*/`. The follow-up analyses (paired tests,
code audit, prototype/non-transductive re-scoring, attention recovery,
failure catalogue) are in the `outputs/*_20261004*` and `outputs/*_20261005*`
folders. See the root README for the order to run them in.

## Frozen encoders

- **Solar** (`encoders/solar_encoder.py`): the GOES+SHARP LSTM
  `src/solar/checkpoints/real_full_sequence_goes_sharp_20100501_20250724/lstm_label_primary.pt`
  (`input_size=20`: 2 GOES irradiance channels + 18 SHARP descriptors),
  trained under the Cycle 24 (train) / Cycle 25 (test) split; unimodal test
  TSS 0.555, ROC-AUC 0.854.
- **Martian** (`encoders/martian_encoder.py`): the storm-detection LSTM
  (window 3 sols, horizon 1 sol) from the canonical `mars.run_all` run
  `src/martian/outputs/real/run_20261004T080014968911Z_seed42`, copied to
  `src/martian/checkpoints/task_storm_detection/lstm_w3_h1.pt`
  (`input_size=36`, the dust-only storm-detection feature manifest).

Paths and architecture metadata are in `configs/encoders.yaml`; the
sequence-tensor exports both encoders consume are located by
`fusion_data.py`.

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
