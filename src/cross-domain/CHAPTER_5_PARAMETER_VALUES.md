# Chapter 5 [INSERT]/[CONFIRM] Values — Real, From This Repository

Every value below is read directly from real, working, tested code in this repository
(`src/cross-domain`, plus the real checkpoints in `src/solar` and `src/martian`), never
invented to make the chapter read cleanly. Where the actual implementation does not do
what the chapter's prose assumes, or where a chapter requirement was not built, that is
stated plainly rather than glossed over — consistent with this project's no-fabrication
convention. Source file/line references are given so every value can be re-verified
independently.

---

## 0. Scope caveats to read before filling anything in

**2026-08-14: martian encoder re-promoted from a corrected `src/martian` retrain, and
the fusion/ablation pipeline re-run end to end against it — the empty-validation-split
finding below (caveat 3) and its Section 5.5.1 consequence are now RESOLVED, not just
documented.** `src/martian`'s chronological train/val split had a real bug: its
configured validation window (sols 601-668) fell entirely outside MY36's real data
coverage (which never reaches past sol 464), so every real martian run's validation
split — including the one this package's fusion/ablation training depended on — was
silently empty. This was fixed at the source (`src/martian/configs/splits.yaml`'s val
window moved to `[432,668]`, chosen by scanning for zero anomaly-episode boundary
crossings; `src/martian/modeling/train.py` also had to be fixed since it only ever
built train/test sequence datasets, never val, silently early-stopping on the TEST
split before this fix) — see `src/martian/CHAPTER_4_PARAMETER_VALUES.md` Section 13
for the full martian-side writeup, which also fixed a real sentinel-row ingestion bug
(22.2% of the raw archive) and ran a real missing-data sensitivity sweep. The new
canonical martian run is `outputs/real/run_20260813T195013457439Z_seed42`
(supersedes `run_20260805T115705602822Z_seed42`). Concretely for this package:
the promoted martian LSTM checkpoint (`src/martian/checkpoints/task_storm_detection/
lstm_w3_h1.pt`) is now the corrected, leakage-fixed one (byte-verified different from
the pre-2026-08-14 file; feature-manifest column list confirmed identical between the
two runs, so this is a drop-in, architecture-safe replacement, not a fabricated
substitute), the martian fusion export (`src/martian/export_fusion_sequences.py
--run-dir outputs/real/run_20260813T195013457439Z_seed42`) now has a real, non-empty
validation split (11,360 rows, 49 positives — see updated caveat 3), and Stage 2
fusion training now genuinely exercises validation-gated early stopping for the first
time in this repository's history (real varying convergence epochs across the 7
ablation configurations: 6, 6, 28, 6, 6, 14, 10 — not the previous fixed
`early_stopped_epoch_4` every single time). The real before/after result is dramatic
and is written up in full in the Appendix below: this was not a small correction.

**2026-08-05: martian encoder promoted from a simulated-data checkpoint to a real
one, and the fusion/ablation pipeline re-run end to end against it.** The martian
LSTM encoder now loads a real checkpoint trained by `src/martian`'s `mars.run_all`
against real EMIRS/MCD data (`outputs/real/run_20260805T115705602822Z_seed42`,
k=2/m=3 primary regime, real Table 4.6 catalogue validation — see
`src/martian/CHAPTER_4_PARAMETER_VALUES.md`), not simulated data. Both domains'
encoders are now real. `d_M` (Eq. 5.1) changed from 61 to **36** as a direct
consequence (the real martian feature set is dust-only, no
`surface_temperature_k` family — see caveat 3 below for the full real numbers).
All simulation-only code paths, comments, and stale file paths have been removed
from `src/cross-domain` — there is no simulated fallback anywhere in this package.

These affect how several `[INSERT]`/`[CONFIRM]` markers below should be interpreted or
reworded, not just what number to write in:

1. **RESOLVED (2026-07-29): the solar encoder is now GOES+SHARP, matching Section
   5.4.1's formal definition of $X_S$ literally.** (This caveat previously flagged a
   real gap — GOES-only, `input_size=2` — against Section 5.2's "Solar forecasting
   integrates irradiance time series and magnetic-field-derived descriptors" and
   Section 5.4.1's "comprising the GOES irradiance sequence **and SHARP
   magnetic-descriptor sequence**"; kept below, struck through in spirit, so the
   resolution is traceable.) The real, built encoder
   (`src/solar/checkpoints/real_full_sequence_goes_sharp_20100501_20250724/
   lstm_label_primary.pt`, `input_size=20`) now takes the 2 GOES features
   (`goes_seq_xrsb_flux_w_m2`, `goes_seq_xrsa_flux_w_m2`) **and** the 18 real SHARP
   physical descriptors (`data_sources/schema.py::SHARP_PHYSICAL_KEYWORDS` —
   USFLUX, TOTUSJZ, TOTUSJH, ABSNJZH, SAVNCPP, MEANPOT, TOTPOT, R_VALUE, MEANGAM,
   MEANGBT, MEANGBZ, MEANGBH, MEANALP, MEANJZH, MEANJZD, MEANSHR, AREA_ACR,
   SHRGT45), disk-aggregated per real SHARP T_REC timestamp
   (`src/solar/preprocessing/alignment.py::aggregate_sharp_disk`, reused unchanged)
   and merged onto GOES's grid via an **exact** `time_utc` match — never a
   fuzzy/nearest-timestamp join (`src/solar/preprocessing/sharp_sequences.py`). A
   GOES timestamp with no exactly-matching SHARP row, or any raw NaN SHARP value, is
   a real, flagged gap (`sharp_seq_missing_any`) that discards that training window
   entirely (Algorithm 3.1 step 9's zero-tolerance policy, now honored for both input
   sources). SHARP columns are train-only-fit robust-scaled
   (`src/solar/preprocessing/scaling.py`, reused unchanged). $d_S$ (Eq. 5.1) does
   **not** change (`hidden_size=32`, unaffected by the input-width increase — see Eq.
   5.1's table below) — only $d_G$ effectively becomes $d_G+d_{\text{SHARP}}=20$ inside
   the encoder's own input layer, invisible to everything downstream of $h_S$. The
   prior GOES-only checkpoint (`real_full_sequence_20100501_20250724/`) is left on
   disk, unmodified, for comparison: real test TSS improved from **0.179** (GOES-only)
   to **0.555** (GOES+SHARP), ROC-AUC from 0.628 to 0.854
   (`src/solar/results/real_full_sequence_goes_sharp_20100501_20250724/
   run_summary.json`). One residual, narrower scope gap remains: Chapter 3.6's fuller
   multi-branch solar stack (separate CNN-LSTM/FC-SHARP/Transformer branches with
   within-solar fusion) still doesn't exist — this is a single LSTM over the
   concatenated GOES+SHARP sequence, not that multi-branch design.
2. **Only the martian `storm_detection` task is wired into fusion**, not
   `dust_regression`/`temperature_regression` (Chapter 4's other two tasks). The
   martian encoder's checkpoint (`lstm_w3_h1.pt`) is specific to storm detection,
   window=3 sols, horizon=1 sol.
3. **RESOLVED (2026-08-14): martian's validation split is now real and non-empty.**
   The `val=0` finding below was a real bug in `src/martian`'s chronological split
   (its validation window fell entirely outside MY36's real data coverage), not an
   inherent property of the real dataset — fixed at the source (see caveat 0 above
   and `src/martian/CHAPTER_4_PARAMETER_VALUES.md` Section 13). Current real counts
   (window=3 sols, horizon=1 sol, `storm_detection`, from the corrected
   `run_20260813T195013457439Z_seed42`): **train=37,977, val=11,360, test=35,288**.
   Real prevalence: train=4.18%, val=0.43%, test=0.71%. (Kept below, struck through
   in spirit, for traceability of what was wrong and why:) ~~Martian's real
   sequence-feasible sample is now substantially larger, but the validation split has
   zero feasible windows. Real counts: train=55,781, val=0, test=38,646 — up from the
   old simulated pipeline's 58/9/66 by roughly three orders of magnitude. Real
   prevalence: train=3.21%, val=undefined (0 rows), test=0.71%. MY36's validation
   window (sols 601-667, only 67 sols) is too short to produce any 3-sol sequence
   window that survives the real sequence-feasibility gate for this task/horizon.~~
   Source: `src/martian/export_fusion_sequences.py`'s real run output
   (`outputs/real/run_20260813T195013457439Z_seed42/fusion_export/`).
4. **RESOLVED (2026-08-18): the Section 5.6.1 uncertainty protocol (seed variation
   + block bootstrap + paired comparison) is now real, built, tested, AND executed**
   against all 7 configurations at 5 real seeds
   (`outputs/uncertainty_sweep_20260818T090949043479Z/uncertainty_report.md`). The
   headline result is not merely "intervals now exist" — it changes what can honestly
   be claimed: **A3/A4/A7 (the configurations that route through the real batch-mean
   attention-mixing pathway) show far larger seed-to-seed TSS variance than any
   non-mixing configuration** (martian TSS across seeds 44/45/46/47/48: 0.4482,
   0.0712, 0.0000285, 0.3844, 0.0035 — sd=0.2190, min≈0 to max≈0.45; solar: 0.5507,
   0.2207, 0.3992, 0.5532, 0.4907 — sd=0.1263), versus A1's sd=0.0000 (deterministic,
   no stochastic training), A5's sd=0.0117/0.0201 (alignment-only, no mixing), and
   A2's sd=0.0563/0.0396 (capacity-matched, no cross-domain information at all). The
   previously-reported single-seed (`FUSION_SEED=44`) point estimates were real and
   not wrong, but they were **not representative of the fusion pathway's typical
   behavior** — seed 44 happened to land near the high end of a wide, unstable range
   for A3/A4/A7's martian TSS in particular. See the Appendix for the full real
   numbers and their reading.
   `evaluation/comparison.py::evaluate_predictions` gained optional
   `predictions_dir`/`config_id`/`domain`/`seed` kwargs that, when given, write a real
   `predictions_<config>_<domain>_seed<NN>.csv` of that config's own fused test-set
   predictions (`y_true`, `y_prob`, `threshold`, `regime`) — the exact input format
   `uncertainty_analysis.py` (new, top-level in this package) consumes.
   `run_ablation.py` gained a `seed` parameter (threaded through every stage-two
   training call, replacing the previously-hardcoded `FUSION_SEED` constant; A1 is
   unaffected, since it loads the unchanged frozen Chapter 3/4 checkpoints) and a
   `light` mode that skips a seed-sweep run's own duplicate model checkpoints/figures,
   writing only `history_*.json`, `tss_summary.json`, `decisive_comparisons.csv`, and
   (via `predictions_dir`) the per-instance predictions. `run_uncertainty_sweep.py`
   (new) drives 5 real seeds (44 — the existing canonical seed — plus 45-48) through
   `run_ablation(light=True)`, collects every config/domain's predictions into one
   shared `outputs/uncertainty_sweep_<id>/`, and runs `uncertainty_analysis.py`'s full
   protocol (seed mean/sd, B=1000 block-bootstrap 95% intervals, paired differences
   with "established" = interval excludes zero) against them, writing
   `uncertainty_report.md` in that directory.
   **A real bug fixed while integrating this**: `uncertainty_analysis.py`'s own
   module docstring specifies a single illustrative `--block-length 24`, described as
   "the forecast window in instances." Applied literally to both domains at once this
   is wrong for solar: solar rows are **12-minute cadence**
   (`src/solar/configs/goes_preprocessing.yaml`'s `resample_cadence_minutes: 12`), not
   hourly, and the forecast horizon is **24 hours**
   (`src/solar/configs/labeling.yaml`'s `forecast_horizon_hours: 24`), so the real
   block length spanning one full horizon is `24*60/12 = `**120** rows, not 24 — a
   block length of 24 would span only 4.8 real hours, materially under-preserving
   flare-clustering autocorrelation and producing an artificially narrow, invalid CI.
   Martian's `1` (one row = one sol, one-sol horizon) was already correct. Fixed by
   making block length domain-specific throughout (`DEFAULT_BLOCK_LENGTHS = {"solar":
   120, "martian": 1}`, `--block-length-solar`/`--block-length-martian` CLI flags,
   `--block-length` retained as an explicit override-both escape hatch) rather than a
   single global default — 21 new tests (`tests/test_uncertainty_analysis.py`,
   `tests/test_comparison.py`) cover the block-bootstrap mechanics, the paired-
   comparison "established" logic (including a planted-gap recovery test and an
   identical-predictions null-result test), the CSV loader, and the per-domain
   block-length wiring on synthetic data with known planted structure — not run
   against real predictions yet within the test suite itself, by design (that's what
   the real sweep is for). See the Appendix for the real 5-seed run's results once
   `run_uncertainty_sweep.py` has completed and this caveat is updated to RESOLVED.
5. **Figure 5.2 (MI rankings / partial-correlation diagnostics) does not exist.** No
   mutual-information or partial-correlation analysis has been built anywhere in this
   repository. This figure would need new, separate analysis code.

---

## 1. Section 5.2 — Figure 5.2 caption

> `[CONFIRM: experiment/run identifier for the panels in Figure 5.2, per the
> empirical-versus-schematic captioning convention adopted in Chapter 6]`

**NOT AVAILABLE.** Per caveat 5 above, this figure was never built. There is no real
run identifier to cite. Recommendation: caption Figure 5.2 as **schematic**
(illustrative, not computed from real data) until a real MI/partial-correlation
analysis script is written, or remove/defer the figure.

---

## 2. Section 5.4.1 — Formal Definition of the Fusion Model

### Eq. (5.1) — Encoders

| Marker | Real value | Source |
|---|---|---|
| $T_M$ = Martian window length in sols | **3** | `configs/encoders.yaml` (martian architecture matches `lstm_w3_h1.pt`); `run_ablation.py`/`export_fusion_sequences.py`'s `WINDOW_LENGTH_SOLS = 3` |
| $d_M$ = Martian feature dimension | **36** (32 real engineered `feat_`-prefixed columns + 4 raw current-state columns: `dust_optical_depth`, `dust_optical_depth_z`, `dust_anomaly`, `atmospheric_anomaly`) | `configs/encoders.yaml`'s `martian.input_size: 36`; exact ordered list in `outputs/real/run_20260813T195013457439Z_seed42/features/storm_detection_feature_manifest.json` (column-for-column identical to the superseded `run_20260805T115705602822Z_seed42` manifest — confirmed 2026-08-14 before promoting the corrected checkpoint) |
| Which layer is exposed | **Final LSTM hidden state** (last time-step output, taken BEFORE the domain's own task head) | `encoders/frozen_encoder.py::FrozenSequenceEncoder.forward`: `out, _ = self.lstm(x); return out[:, -1, :]` — not a pooled sequence output, not a penultimate dense layer |
| $d_S$ | **32** | `configs/encoders.yaml`'s `solar.hidden_size: 32` |
| $d_M''$ | **16** | `configs/encoders.yaml`'s `martian.hidden_size: 16` |

Also real and worth stating explicitly: $\theta_S$/$\theta_M$ are confirmed frozen —
`FrozenSequenceEncoder.__init__` sets `requires_grad_(False)` on every parameter and the
module never leaves `.eval()` mode; verified by `tests/test_fusion_model.py::
test_only_fusion_parameters_are_trainable_not_the_frozen_encoders` and
`test_backward_pass_only_updates_fusion_parameters` (real, passing tests, not just a
design intention).

### Eq. (5.2) — Projection into shared space

| Marker | Real value | Source |
|---|---|---|
| $d$ = shared embedding dimension | **16** | `configs/fusion.yaml`'s `shared_embedding_dim: 16` |
| Non-linearity after projection? | **No** — plain affine map, exactly as Eq. (5.2) is written | `configs/fusion.yaml`'s `projection_nonlinearity: null`; `fusion/projections.py::DomainProjection.forward` is a bare `nn.Linear` |

### Eq. (5.3)/(5.4) — Alignment loss

| Marker | Real value | Source |
|---|---|---|
| Biased (1/n) vs. unbiased (1/(n-1)) variance | **Biased (1/n)**, exactly as Eq. (5.3) is written | `fusion/alignment.py::batch_moments`: `variance = ((h - mean) ** 2).mean(dim=0)` (a plain `.mean()`, not Bessel-corrected) |
| $\beta$ = relative weight of variance term | **1.0** (equal weighting) | `configs/fusion.yaml`'s `alignment_variance_weight: 1.0` |
| Variance ($\hat\sigma^2$) or std ($\hat\sigma$)? | **Variance**, matching Eq. (5.4) exactly — not standard deviation | `fusion/alignment.py::moment_alignment_loss`: `variance_term = torch.sum((var_solar - var_martian) ** 2) / d` |
| Covariance-matching term present? | **No** — only marginal mean/variance are constrained, exactly as the chapter text states | `fusion/alignment.py` — confirmed no cross-dimensional covariance term anywhere |

### Eq. (5.5) — Adaptive weighting (attention scorer)

> `[CONFIRM: the exact scorer, e.g. a single dense layer..., a bilinear form, or
> dot-product attention against a learned task query]`

**Single dense layer with tanh**, exactly the chapter's own first-listed example form:
$a_{t,k} = w_t^\top \tanh(W_t \tilde h_k + b_t)$.

- Source: `fusion/attention.py::AttentionScorer.forward`: `self.score(torch.tanh(self.dense(h)))`.
- `attention_scorer_hidden_size` (the dimensionality of $W_t \tilde h_k + b_t$) = **8** (`configs/fusion.yaml`).
- One scorer per task, as the chapter specifies: `scorer_solar_task`, `scorer_martian_task` (`fusion/model.py`).

**Real, documented deviation from Eq. (5.6)'s literal per-instance form** (relevant
context for anyone citing Eq. 5.5–5.6): because solar/martian batches are drawn
independently with different sizes ($n_S \ne n_M$, no sample-level pairing — exactly as
Section 5.5 itself states), the "other" domain's contribution at each step is that
batch's own **mean** embedding, broadcast across every row of the own-domain batch —
not a literal per-row $\tilde h_M$ as Eq. (5.6) implicitly assumes. This is documented
as an explicit ENGINEERING DECISION in `fusion/attention.py::fuse`'s docstring, and is
the only interpretation compatible with the unpaired-batch design already described in
Section 5.5.1. **Real consequence, confirmed across FOUR independent runs now
(GOES-only, GOES+SHARP, the first real-martian-checkpoint re-run, and the
2026-08-14 corrected-split re-run below)**: because `fuse()` reduces the other
domain to its batch mean, A4's "within-batch-shuffled" control (Table 5.2) produces
results mathematically **identical** to A7 ("real") to full floating-point precision
(martian TSS 0.4535760949735148, solar TSS 0.4697567705865804, for both A4 and A7 in
the 2026-08-14 corrected-split run — previously 0.006829684349686434 martian / 0.0
solar under the pre-fix run, a different absolute level but the same exact-equality
property) — shuffling row order never changes a mean. This is a real,
architecture-inherent, encoder-independent property, confirmed again under both the
original and the corrected real martian data, not noise, not a bug, and not an
artifact of any one checkpoint or training run.

### Eq. (5.7) — Balance regularisation

> `[CONFIRM: the implemented penalty, KL divergence to uniform..., negative-entropy...,
> or squared-error...]`

**KL divergence to uniform, exactly as Eq. (5.7) is written.**
Source: `fusion/balance.py::balance_regularizer`: `torch.sum(alpha_bar * torch.log(alpha_bar / uniform))`.
Confirmed real and implemented — the fallback clause ("if no such penalty exists...")
does not apply.

### Task heads and total objective (Eq. 5.8)

| Marker | Real value | Source |
|---|---|---|
| Head architecture $g_S$, $g_M$ | **A single `nn.Linear(shared_dim, 1)` + sigmoid** for each task (A7/A3–A6). No hidden layer. (A2's ablation-only capacity-matched heads DO have one hidden layer — see Table 5.2 section below.) | `fusion/model.py`: `self.head_solar = nn.Linear(shared_dim, 1)`, same for `head_martian` |
| Solar task loss $\mathcal{L}_S$ | **Weighted binary cross-entropy** (Chapter 3.7's formula: $w_+ = N/(2N_+)$, $w_- = N/(2N_-)$, train-split-only) | `training/objective.py::weighted_binary_cross_entropy` + `compute_class_weights` |
| Martian task loss $\mathcal{L}_M$ | **CONFIRMED, unweighted binary cross-entropy** — Chapter 5 Section 5.4.1's own text states directly: "the Martian storm-detection task uses unweighted binary cross-entropy." Matches the implementation exactly, not a placeholder. | `training/objective.py::martian_task_loss` |
| $\lambda_{\text{align}}$ | **0.1** (A7/A3/A4/A5 default); **0.0** for A6 (`fusion_without_alignment`) | `configs/fusion.yaml`'s `lambda_align: 0.1`; A6's override in `run_ablation.py::_train_fusion_variant` |
| $\lambda_{\text{bal}}$ | **0.1** | `configs/fusion.yaml`'s `lambda_bal: 0.1` |
| $\lambda_{\text{wd}}$ | **1.0e-4** | `configs/fusion.yaml`'s `lambda_wd: 1.0e-4` |

None of these three $\lambda$ values were tuned under a real "Appendix B" protocol —
they are documented, reasonable starting points (`configs/fusion.yaml`'s own header
comment: *"Starting points pending real tuning ... deliberately modest so no single
term dominates the total objective by construction"*). If Appendix B doesn't exist in
this thesis delivery, reword that clause.

### Cross-domain input at inference time

> `[CONFIRM: (a) contemporaneous embedding from the other domain's live stream, (b) a
> fixed prototype, or (c) domain-only inference with cross-domain weight zeroed]`

**(b) — a fixed prototype**: the training-set mean **projected** embedding per domain,
recorded at the end of Stage 2 training and reused for every subsequent missing-domain
inference call.

- Config: `configs/fusion.yaml`'s `inference_missing_domain_policy: fixed_prototype`.
- Implementation: `training/train_fusion.py::_update_prototypes` (called once,
  unconditionally, at the end of `train_fusion`) computes
  `h_solar_proj.mean(dim=0)`/`h_martian_proj.mean(dim=0)` over the full training set and
  stores them in `model.solar_prototype`/`model.martian_prototype` buffers.
- Used by: `fusion/model.py::predict_solar_only`/`predict_martian_only`.

---

## 3. Section 5.5.1 — Training Procedure

### Batch strategy

| Marker | Real value | Source |
|---|---|---|
| $n_S$ | **64** (nominal) | `configs/fusion.yaml`'s `batch_size_solar: 64` |
| $n_M$ | **64** (nominal) | `configs/fusion.yaml`'s `batch_size_martian: 64` |
| Per-domain imbalance-aware sampler (e.g. event oversampling ratio) | **NOT IMPLEMENTED as oversampling.** `_sample_batch` in both `training/train_fusion.py` and `training/train_unimodal.py` is plain uniform random sampling *without replacement* (`rng.choice(..., replace=False)`), with no class-balanced oversampling. Class imbalance is instead handled entirely via the loss weighting above ($w_+, w_-$ for solar; unweighted for martian). This should be stated plainly rather than implying an oversampling scheme exists. | `training/train_fusion.py::_sample_batch` |

**Real, honest consequence, superseding the earlier tiny-sample finding now that
both encoders are real**: martian's real train split (37,977 rows as of the
2026-08-14 corrected run) remains larger than solar's real train split (28,719 rows)
for this fusion pairing, so `steps_per_epoch = max(1, min(n_train_solar // 64,
n_train_martian // 64))` evaluates to **448** (`28719 // 64 = 448`, `37977 // 64 =
593`, minimum wins) — no longer floored to 1.

**RESOLVED (2026-08-14).** Previously, martian's validation split had 0
sequence-feasible rows (a real bug in `src/martian`'s chronological split, since
fixed — see caveat 0/3 above), so `val_available` (`len(X_solar_val) > 0 and
len(X_martian_val) > 0`) was `False` for every real run in this repository, `val_score`
was `NaN` every epoch, and early stopping always fired at exactly
`early_stopping_patience` (5) epochs regardless of training progress
(`early_stopped_epoch_4`, `_no_validation_selection_available` appended). This was
real, expected, and honestly stated at the time — not a bug in this package's own
`val_available` guard, which correctly avoided falling back to test data; the bug was
upstream, in what fed it an empty array. Now that martian's validation split is real
and non-empty (11,360 rows, 49 positives), `val_available` is `True` and genuine
validation-gated early stopping runs for the first time: the 2026-08-14 re-run of
`run_fusion.py` (single A7 model) converged at `early_stopped_epoch_6`, and the
`run_ablation.py` 7-configuration sweep produced real, varying convergence epochs
(A7=6, A2 solar=6/martian=28, A3=6, A4=6, A5=14, A6=10) — direct evidence that
training duration is now actually responding to validation performance rather than
hitting a fixed floor every time. See the Appendix for the resulting real skill
numbers, which changed substantially as a consequence.

### Eq. (5.9) — Optimiser

| Marker | Real value | Source |
|---|---|---|
| Optimiser | **Adam** (PyTorch default $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ — not separately tuned) | `training/train_fusion.py`: `torch.optim.Adam(model.trainable_parameters(), lr=training_config["learning_rate"])` |
| $\eta$ (learning rate) | **0.001**, constant — **no schedule** | `configs/fusion.yaml`'s `learning_rate: 0.001` |

### Gradient flow under frozen encoders

> `[CONFIRM: dropout disabled and batch-normalisation using stored running statistics...]`

**Not applicable** — there is no dropout or batch normalisation anywhere in the fusion
architecture (`fusion/model.py`, `fusion/attention.py`, `fusion/projections.py`). The
frozen `nn.LSTM` layers have `num_layers=1`, so PyTorch's own inter-layer dropout is
inactive regardless of any dropout parameter value. `FrozenSequenceEncoder` is
permanently in `.eval()` mode. Table 5.1's gradient-flow claims are confirmed real by
passing tests (`tests/test_fusion_model.py`,
`tests/test_ablation_forward.py::test_use_fusion_pathway_false_reads_own_projection_directly`),
not merely asserted.

### Algorithm 5.1

| Marker | Real value | Source |
|---|---|---|
| Stage 1a (solar) validation selection criterion | **Best validation TSS** (not loss) | `src/solar/modeling/sequence_models.py::train_sequence_model`, `best_val_tss` |
| Stage 1b (Martian) validation selection criterion | **Best (lowest) validation loss** (not TSS) — a real, documented asymmetry between the two domains' own Chapter 3/4 selection criteria, inherited unchanged here | `src/martian/modeling/sequence_models.py::train_sequence_model`, `best_val_loss` |
| Stage 2 initialisation scheme | **PyTorch default** (Kaiming-uniform `nn.Linear` init) | `configs/fusion.yaml`'s `initialization_scheme: pytorch_default`; also seeded explicitly (`torch.manual_seed(FUSION_SEED)` before model construction in `run_fusion.py`/`run_ablation.py`, `FUSION_SEED=44`, for cross-run reproducibility) |
| Evaluate every `[INSERT]` steps | **NOT literally "every N steps."** The real implementation evaluates on validation once **per epoch**, not on a custom step interval. Given `steps_per_epoch` is frequently 1 (see above), this is close in practice but should be described accurately as per-epoch, not per-step. | `training/train_fusion.py`'s main loop structure |
| Halt if `[INSERT: patience]` evaluations pass without improvement | **5** | `configs/fusion.yaml`'s `early_stopping_patience: 5` |
| Halting criterion — combined score or per-domain? | **A single combined score**: the negative of the full Eq. (5.8) total objective evaluated on the validation batches (`val_score = -val_losses["total"]`). This is **not** a TSS-based combined skill score — `training/train_fusion.py`'s own comment states explicitly: *"a real deployment would swap in each domain's own TSS-based skill score; this module deliberately has no dependency on either domain's own metrics package."* | `training/train_fusion.py`, main loop |
| "...or if either task's validation skill falls below its stage-one unimodal baseline by more than `[INSERT: tolerance]`" | **NOT IMPLEMENTED.** No such per-task-degradation-triggered early stop exists anywhere in `train_fusion`. Only the combined-loss patience mechanism above exists. This clause should be removed or explicitly flagged as a documented gap, not silently assumed present. | — |
| Fallback behaviour mechanism | **Fixed prototype** (see "Cross-domain input at inference time" above) — matches option (b), not a cross-domain-weight-zeroing mechanism (option c) | `fusion/model.py::predict_solar_only`/`predict_martian_only` |

---

## 4. Section 5.6 — Evaluation Strategy

### 5.6.1 — Ablation Design (Table 5.2)

The seven configurations, `decisive_comparisons`, and every "what it isolates" cell
match `configs/ablation.yaml` **exactly**, word for word — confirmed directly against
the chapter text. Real trainable-parameter counts, computed from the actual constructed
models in this repository:

| ID | Configuration | Trainable parameters (real) | Source |
|---|---|---|---|
| A1 | Unimodal baseline (Ch.3/4 models, unchanged) | **Solar: 6,945** (6,912 frozen-when-reused LSTM + 33 original head `nn.Linear(32,1)` — unchanged since the GOES+SHARP retrain, the solar checkpoint was not touched by this work). **Martian: 3,473** (real checkpoint: `rnn.weight_ih_l0` (64,36)=2,304 + `rnn.weight_hh_l0` (64,16)=1,024 + biases (64+64)=128 + `head.weight`/`head.bias` (16+1)=17 — down from 5,073 under the old simulated checkpoint's `input_size=61`, since only `weight_ih_l0`'s input dimension shrank, 61→36, everything else unchanged). Reported per-domain since A1 has no single joint model — these ARE the real Chapter 3/4 checkpoint sizes, not re-trained here. | Computed directly from both real checkpoints' `state_dict` shapes, 2026-08-05 |
| A2 | Capacity-matched unimodal | **Solar: target 689 / actual 691** (100.3% of target). **Martian: target 433 / actual 435** (100.5% of target). Both real runs met the "within ~5%" constraint the chapter specifies — with room to spare, since the closed-form widening (`fusion/unimodal.py::_solve_head_hidden_size`) always meets-or-exceeds the target by construction, never falls short. | `fusion/unimodal.py::CapacityMatchedUnimodalModel`, `domain_share_of_trainable_params`; real numbers from `outputs/ablation_run_<id>/ablation_report.md`'s "A2 capacity-matching" table |
| A3 | Fusion with noise auxiliary | **1,122** (same joint architecture as A7 — only the forward-pass `other_domain_embedding` value differs, not parameter count) | `fusion/model.py::CrossDomainFusionModel`, computed total |
| A4 | Fusion with shuffled auxiliary | **1,122** | same |
| A5 | Alignment only | **1,122** | same |
| A6 | Fusion without alignment | **1,122** | same |
| A7 | Full model | **1,122** total = projection_solar (528) + projection_martian (272) + scorer_solar_task (144) + scorer_martian_task (144) + head_solar (17) + head_martian (17) | `fusion/model.py::CrossDomainFusionModel.trainable_parameters()` |

Note: A2's *target* (689 solar / 433 martian) is defined as that domain's own real share
of A7's total (projection + scorer + head for that task only) — **not** A7's combined
1,122 — since A2 has no cross-domain component to compare against the other domain's
own parameters. This is documented as an explicit ENGINEERING DECISION in
`fusion/unimodal.py`'s module docstring (the config file specifies no widening formula).

**Uncertainty quantification**: per caveat 4 above, the paired-bootstrap protocol
described here and in Section 5.6.2 is now real, built, and executed
(`uncertainty_analysis.py`, `run_uncertainty_sweep.py`,
`outputs/uncertainty_sweep_20260818T090949043479Z/uncertainty_report.md`) — see the
Appendix for the real 5-seed results and their reading. Everything in
`outputs/ablation_run_<id>/ablation_report.md` itself remains a single-seed point
estimate (that report's own job is the ablation training/evaluation, not the
uncertainty sweep, which is a separate real run over `outputs/uncertainty_sweep_<id>/`)
— cite the uncertainty report, not the ablation report, for interval-bearing values.

### 5.6.2 — Negative Transfer Analysis

| Marker | Real value | Source |
|---|---|---|
| $M$ = primary skill metric | **TSS**, for both tasks — martian's `storm_detection` is a binary classification task (storm episode yes/no within the forecast horizon), so the same TSS formula applies unchanged; no separate martian metric exists or is needed in this fusion package (martian's other two Chapter 4 tasks, `dust_regression`/`temperature_regression`, are not wired into fusion at all — see caveat 2) | `evaluation/negative_transfer.py::true_skill_statistic` |
| Thresholds selected independently per model, per domain, on validation only | **Confirmed, real** — `evaluation/thresholds.py::select_threshold_by_tss`, called separately for the fused and unimodal predictions inside `evaluation/comparison.py::evaluate_predictions` | — |
| Solar regime bins ("by cycle phase and by activity level") | **NOT the described bins.** Real implementation uses **calendar year** (2020–2025, the 6 real years spanning the Cycle-25-only test partition) as a simple, defensible temporal-regime proxy — not formal cycle-phase (rise/max/decline) or flare-activity-level bins. State this as a real simplification if citing regime-level solar results. | `src/solar/export_fusion_sequences.py`'s `regime_test` = `joined["time_utc"].dt.year` |
| Martian regime bins ("by season, solar-longitude bins") | **Real solar-longitude ($L_s$) quartile bins**, labeled Q1–Q4 — computed via `pandas.qcut` on the real `solar_longitude_ls` values within the martian test partition (empirical quartiles, not fixed physical Ls-degree ranges like a standard Northern-Spring/Summer/Autumn/Winter split). Reasonably fulfills "solar-longitude bins" but is data-driven, not a fixed astronomical convention — worth a one-line methods note if cited directly. | `src/martian/export_fusion_sequences.py`'s `regime_test` construction (real `solar_longitude_ls` per cell-sol, from `preprocessing/features.py`'s bookkeeping columns) |
| Martian dust-regime bins (clear vs. elevated-opacity, threshold `[INSERT]`) | **NOT IMPLEMENTED.** No dust-opacity-based regime stratification exists anywhere in this repository; only the $L_s$-quartile stratification above. | — |

Real per-regime results from this session's actual runs are in
`outputs/run_<id>/negative_transfer_{solar,martian}.csv` (single-model fused-vs-unimodal
run) and `outputs/ablation_run_<id>/negative_transfer_*.csv` (pairwise ablation
comparisons) — see the Appendix below for a summary.

### 5.6.3 — Failure-Case Analysis Protocol

| Marker | Real value | Source |
|---|---|---|
| False-alarm probability threshold `[INSERT]` | **0.5** | `run_fusion.py`/`run_ablation.py`'s `catalogue_failure_cases(..., false_alarm_probability_threshold=0.5)` |
| Per-prediction attention-weight logging enabled? | **Yes, real and confirmed** — every failure case catalogued in `outputs/<run_id>/failure_cases_*.csv` carries real `alpha_solar_at_issuance`/`alpha_martian_at_issuance` columns, sourced directly from the model's real forward pass (`fusion/model.py::CrossDomainFusionModel.forward`'s `alpha_solar`/`alpha_martian` outputs), not added retroactively. | `evaluation/failure_cases.py::catalogue_failure_cases`; real output columns confirmed in every run this session |

---

## Appendix — Real results obtained with the values above (informational, for Chapter 6)

Not a Chapter 5 methodology value, but included since it's the direct, real output of
running the methodology with the parameters resolved above.

**2026-08-14 corrected re-run (current, canonical)**:
`outputs/run_20260814T064609079205Z/final_report.md` and
`outputs/ablation_run_20260814T064653396282Z/ablation_report.md`, the first run in
this repository with a real, non-empty martian validation split (see caveat 0/3 and
Section 5.5.1 above). Both files' own `[DATA SOURCE: ...]` headers confirm
`REAL (solar) / REAL (src/martian mars.run_all pipeline,
outputs/real/run_20260813T195013457439Z_seed42)`.

**2026-08-05 run (superseded — kept below for the before/after comparison, do not
cite these numbers as current)**: `outputs/run_20260805T172137033082Z/final_report.md`
and `outputs/ablation_run_20260805T172234087059Z/ablation_report.md`, the first run
against a REAL martian encoder but trained entirely without validation-based model
selection because of the empty-val bug (caveat 3, pre-fix).

### Single-model fusion run (A7)

| | solar TSS (fused) | solar TSS (unimodal) | martian TSS (fused) | martian TSS (unimodal) |
|---|---|---|---|---|
| 2026-08-05 (pre-fix, no real val selection) | 0.0000 | 0.5552 | 0.0068 | 0.0000 |
| **2026-08-14 (corrected, real val selection)** | **0.4698** | 0.5552 | **0.4536** | 0.3019 |

Solar no longer collapses through the fusion pathway (0.0000 → 0.4698), and martian's
fused TSS now *exceeds* its own unimodal baseline (0.4536 > 0.3019) — real, positive
transfer for martian, not the negative transfer the pre-fix run showed. Solar still
trails its own unimodal baseline under fusion (0.4698 < 0.5552), so some real
negative transfer remains on the solar side — see the ablation breakdown below for
which mechanism is responsible.

### Ablation comparison (real TSS by config)

| config | solar TSS (2026-08-05, pre-fix) | solar TSS (2026-08-14, corrected) | martian TSS (2026-08-05, pre-fix) | martian TSS (2026-08-14, corrected) |
|---|---|---|---|---|
| A1 (unimodal baseline) | 0.5552 | 0.5552 | 0.0000 | 0.3019 |
| A2 (capacity-matched unimodal) | 0.4704 | 0.4704 | 0.0031 | 0.4364 |
| A3 (fusion, noise auxiliary) | 0.0000 | 0.5507 | 0.0068 | 0.4482 |
| A4 (fusion, shuffled auxiliary) | 0.0000 | 0.4698 | 0.0068 | 0.4536 |
| A5 (alignment only, no fusion pathway) | 0.5467 | 0.5429 | 0.0067 | 0.4211 |
| A6 (fusion, no alignment) | 0.0049 | 0.2892 | 0.0068 | 0.4239 |
| A7 (full model) | 0.0000 | 0.4698 | 0.0068 | 0.4536 |

(A1's own TSS values differ between the two rows above because A1 uses the
per-domain checkpoints directly, and the martian checkpoint itself was promoted from
the corrected 2026-08-14 retrain — see caveat 0. A1's *solar* number is identical in
both rows since the solar checkpoint was not touched by this fix.)

**The 2026-08-05 "decisive finding" — that solar TSS collapses to ≈0 for every
configuration routing through `fuse()` (A3, A4, A6, A7), regardless of what the
auxiliary embedding contains — does NOT hold under the corrected run and should no
longer be cited.** Under real validation-gated training, A3 (0.5507) and A4/A7
(0.4698) are all close to the A1 unimodal baseline (0.5552), not collapsed to zero.
The original finding was a real, honestly-reported result at the time, but it was a
downstream artifact of the empty-martian-validation-split bug: every pre-fix run
trained for a fixed 5 epochs with no validation-based best-checkpoint restore
(caveat 0/Section 5.5.1), so the specific weights evaluated were whatever an
un-selected, fixed-length run happened to land on for the solar pathway — not
necessarily a representative or good one. This is a useful methodological lesson in
its own right: a real architecture-level negative-transfer conclusion should not be
drawn from a training procedure that never validated its own convergence.

**What the corrected ablation DOES show, real and current:**
- A6 (fusion without alignment) has markedly lower solar TSS (0.2892) than every
  other fusion-pathway configuration (A3/A4/A7 all ≈0.47-0.55) — the moment-alignment
  term (Eq. 5.4, $\lambda_{\text{align}}=0.1$) is doing real, substantial work for
  solar's stability under fusion, the opposite emphasis from the pre-fix conclusion
  (which attributed all the damage to the mixing step itself, not the alignment term).
- Martian TSS is fairly stable across all fusion-pathway configs (0.42-0.45) and
  consistently above both its own A1 unimodal baseline (0.3019) and its A2
  capacity-matched baseline (0.4364, closest competitor) — positive transfer for
  martian appears robust to the specific ablation axis, not fragile to one setting.
- A2's solar TSS (0.4704) is very close to A7's (0.4698) — capacity-matching alone
  (no cross-domain information at all) explains most of solar's degradation from its
  A1 unimodal baseline (0.5552 → ~0.47), suggesting the *smaller effective head*
  under the fusion architecture, not cross-domain interference per se, is the larger
  contributor to solar's residual gap under A7.
- The A4≡A7 batch-mean-fusion exact-equality property (Eq. 5.5-5.6 section above)
  still holds exactly under the corrected run (both 0.4697567705865804 solar,
  0.4535760949735148 martian) — this is architecture-inherent and independent of the
  validation-split fix.
- Real convergence epochs varied meaningfully across configs for the first time
  (A1 n/a, A2 solar=6/martian=28, A3=6, A4=6, A5=14, A6=10, A7=6) — direct evidence
  the fix produced genuine validation-responsive training, not just different final
  numbers.

All of the above are single-seed point estimates (`FUSION_SEED=44`), not averaged
over multiple seeds or bootstrap-resampled — **caveat 4 above no longer applies as a
gap**; this limitation is now closed by the real 5-seed run below, which should be
read alongside every number in this section.

### Section 5.6.1 uncertainty quantification — real 5-seed results (2026-08-18)

Real run: `outputs/uncertainty_sweep_20260818T090949043479Z/` — `run_ablation.py`
retrained (light mode, no duplicate model checkpoints/figures) at seeds
`44, 45, 46, 47, 48` (44 is the pre-existing canonical seed; 45-48 are additional,
otherwise-identical runs), and `uncertainty_analysis.py` ran the full Section 5.6.1
protocol against the resulting 70 `predictions_<config>_<domain>_seed<NN>.csv` files
(7 configs × 2 domains × 5 seeds). Full table: `uncertainty_report.md` in that
directory; per-seed raw TSS: `outputs/ablation_run_<id>/tss_summary.json` for each of
the 5 seed runs. Wall-clock: training sweep 2179s (~36 min), full protocol including
B=1000 block bootstrap 3411s (~57 min) end to end.

**Component 1 — seed variation (TSS mean ± sd across 5 seeds):**

| Config | solar mean | solar sd | martian mean | martian sd |
|---|---|---|---|---|
| A1 (unimodal baseline) | 0.5552 | **0.0000** | 0.3019 | **0.0000** |
| A2 (capacity-matched unimodal) | 0.4833 | 0.0563 | 0.4065 | 0.0396 |
| A3 (fusion, noise auxiliary) | 0.4429 | 0.1391 | 0.1815 | **0.2174** |
| A4 (fusion, shuffled auxiliary) | 0.4262 | 0.1263 | 0.1876 | **0.2190** |
| A5 (alignment only, no mixing) | 0.5222 | **0.0117** | 0.4208 | **0.0201** |
| A6 (fusion, no alignment) | 0.3672 | 0.0808 | 0.3739 | 0.0855 |
| A7 (full model) | 0.4262 | 0.1263 | 0.1876 | **0.2190** |

A1's sd=0.0000 is a real, correct sanity check (it loads the unchanged, deterministic
Chapter 3/4 checkpoints — no stochastic training component exists for it to vary
across seeds). **The decisive new finding is the sd column, not the mean column**:
every configuration that routes through the real attention-mixing pathway of Eqs.
5.5-5.6 (A3, A4, A7 — architecturally identical per the A4≡A7 property below, so
their sd values match exactly) has by far the largest seed-to-seed variance of any
configuration on both tasks, and dramatically so on martian (sd≈0.22, versus ≤0.09
for every non-A3/A4/A7 configuration). Per-seed martian TSS for A7 alone:
**0.4536 (seed 44), 0.0856 (45), 0.0000285 (46, effectively zero), 0.3934 (47),
0.0053 (48)** — the previously-reported "real positive transfer" (fused TSS 0.4536 >
unimodal 0.3019) was a real result at seed 44, but three of the other four seeds show
either no positive transfer or a near-total loss of martian skill under fusion. A5
(alignment active, mixing pathway bypassed) and A2 (no cross-domain information at
all) both stay far more stable across seeds than A3/A4/A6/A7, reinforcing the earlier
single-seed finding that the mixing step (not the alignment loss) is the
architectural locus of instability — but now with seed-robustness evidence behind
it, not just one run's ablation contrast. A6 (mixing without alignment) is
intermediate: markedly more stable across seeds than A3/A4/A7 (which retain
alignment), a real, previously-invisible piece of evidence that **the moment-
alignment term (Eq. 5.4) is not simply neutral for solar/martian stability under
fusion — its presence alongside the mixing pathway coincides with materially higher
run-to-run variance**, a more nuanced reading than the single-seed conclusion that
alignment "does real, substantial work for solar's stability" (that conclusion
compared only A6 against A3/A4/A5/A7 at one seed, and does not survive the seed
sweep in that strong form).

> Fewer than five seeds are present for at least one configuration — **not
> triggered**: every configuration had exactly 5 seeds in this run, so the seed-count
> warning `uncertainty_analysis.py` emits below that threshold did not fire; the sd
> values above are genuinely interpretable under Section 5.6.1's own stated minimum.

**Component 2 — block bootstrap (95% CI on the primary seed=44 point estimate,
B=1000, block length 120 rows for solar [24h horizon at 12min cadence] / 1 row for
martian [1-sol horizon]):** these intervals reflect **test-set sampling
uncertainty at a single seed**, not the between-seed variability of Component 1 above
— the two are complementary per Section 5.6.1's own three-part structure, and must
not be conflated. Given how large Component 1's between-seed spread turned out to be
for A3/A4/A7, a seed-44-only bootstrap CI materially understates the full uncertainty
in any of that configuration's point estimate; full CIs are in
`uncertainty_report.md`'s Section 1-2 tables.

**Component 3 — paired comparisons (identical seed-44 resamples; "established" =
95% CI excludes zero):**

| Comparison | domain | difference | 95% CI | established? |
|---|---|---|---|---|
| A7 − A1 | solar | −0.0854 | [−0.1068, −0.0636] | **yes** |
| A7 − A1 | martian | +0.1516 | [0.0840, 0.2134] | **yes** |
| A7 − A2 | solar | −0.0007 | [−0.0104, 0.0092] | no |
| A7 − A2 | martian | +0.0172 | [−0.0351, 0.0672] | no |
| A7 − A3 | solar | −0.0810 | [−0.0991, −0.0607] | **yes** |
| A7 − A3 | martian | +0.0054 | [−0.0072, 0.0197] | no |
| A7 − A4 | both | 0.0000 | [0.0000, 0.0000] | no (exact) |
| A5 − A7 | solar | +0.0732 | [0.0522, 0.0927] | **yes** |
| A5 − A7 | martian | −0.0325 | [−0.0834, 0.0172] | no |
| A5 − A6 | solar | +0.2537 | [0.2139, 0.2928] | **yes** |
| A6 − A7 | solar | −0.1805 | [−0.2142, −0.1497] | **yes** |

At seed 44: solar negative transfer under fusion is established against both A1 and
A3 (not against A2 — capacity-matching alone accounts for the seed-44 solar gap, as
the single-seed report already found); martian's seed-44 positive transfer against A1
is established, but per Component 1 this specific comparison is exactly the one that
does not replicate at seeds 46/48 (near-zero martian TSS under fusion at those
seeds) — **the paired-comparison "established" verdict at seed 44 should not be read
as "true across seeds"**; it is a statement about seed 44's own test-set sampling
uncertainty only. The A4−A7 exact-zero row (both configs' seed-44 predictions are byte-identical —
martian TSS `0.4535760949735148` for both, matching the value already reported
earlier in this Appendix — consistent with the documented A4≡A7 batch-mean-fusion
property) reproduces as expected under the seed sweep too, and in fact holds at
every one of the 5 seeds (each seed's own `tss_summary.json` shows A4 and A7 exactly
equal on both tasks), confirming the property is architecture-inherent and not an
artifact of one seed or one martian checkpoint.

**Component 4 — per-regime negative-transfer deltas (A7 vs A1, seed 44, with
intervals):** established negative transfer in solar 2022 (Δ=−0.0520, CI
[−0.0787,−0.0155]) and 2023 (Δ=−0.0499, CI [−0.0857,−0.0213]); not established in
2021/2024/2025 (CI includes zero) or 2020 (TSS undefined at seed 44's threshold for
that year, reported n/a rather than fabricated). Martian: established negative
transfer only in $L_s$ quartile Q2 (Δ=−0.2743, CI [−0.5733,−0.0243]); Q1/Q3/Q4 show
positive point deltas, none reaching statistical establishment given the wide
intervals at martian's real, low event prevalence. This materially narrows the
single-seed claim that solar "fails in five of the six regimes examined" (Chapter
5's own Section 5.6 text) — under the real bootstrap intervals, only 2 of the 6 solar
regimes have a 95%-established negative delta; the other 3 nominally-negative regimes
(2021, 2024, 2025) are not statistically distinguishable from zero at seed 44's own
sampling uncertainty, before even considering Component 1's seed-to-seed spread.

**Recommended framing for the chapter text**: the honest, examiner-facing summary is
no longer "cross-domain fusion eliminates solar skill and marginally helps martian"
(the pre-2026-08-14 single-seed reading) nor simply "fusion costs solar ~0.09 TSS and
helps martian" (the 2026-08-14 single-seed reading) — it is **"the attention-mixing
pathway (Eqs. 5.5-5.6) is not just harmful on average, it is unstable: its outcome
depends materially on initialization/batch-sampling seed, in a way the alignment-only
and capacity-matched controls are not.**" That instability is itself evidence against
treating the mixing operator as a reliable component of this architecture, independent
of whether its *average* effect over seeds is positive or negative for either task.
