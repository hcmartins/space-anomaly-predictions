<!-- AUTO-MAINTAINED (by hand). Update when a seed value or its source
     config changes anywhere in the pipeline. -->

# Seed Strategy

Every stochastic component of this pipeline is seeded explicitly and
read from a config file — never a bare hardcoded literal inside a function
body, and never left to numpy/Python's unseeded global RNG state.

**Real-only pipeline: no data-generation randomness left to seed.**
`RealEMIRSObservationProvider.load(seed=...)` accepts a `seed` parameter for
interface compatibility with the (removed) simulated provider it replaced, but
does not use it — real data has no generation randomness. The independent
documented-catalogue provider and its separate ground-truth seed
(`seed_offset`) no longer exist (Section 4.4.1 was descoped entirely, see
`CHAPTER_4_PARAMETER_VALUES.md` Section 7).

## 1. Top-level run seed

`configs/run.yaml`'s `seed` field (`42` in the primary run) is recorded in
every run's manifest (`seed_strategy.global_generation_seed`) for provenance,
even though it has no generative effect on real data — kept as the single
top-level seed field for continuity with the pipeline's config-driven-seed
convention, and so the seed-recording mechanism itself doesn't need special-
casing for a real-only pipeline.

## 2. Per-task model seeds

Each of the **two** modeling tasks has its OWN seed, fixed in that task's own
config (`configs/task_*.yaml`) — never a shared global default:

| Task | Seed | Used for |
|---|---|---|
| `dust_regression` | 42 | `RandomForestRegressor`/`GradientBoostingRegressor` `random_state`; `torch.manual_seed` for LSTM/GRU |
| `storm_detection` | 44 | `RandomForestClassifier`/`GradientBoostingClassifier`/`LogisticRegression` `random_state`; `torch.manual_seed` for LSTM/GRU |

There is no seed 43 / `temperature_regression` anywhere in this repository —
that task was removed entirely, not left at its old seed unused.

`KNeighborsClassifier` has no `random_state` parameter (its predictions are
deterministic given its data and hyperparameters) — the seed is still
threaded through the registry call for interface uniformity but is
intentionally unused (`modeling/classical_models.py::make_knn_classifier`).

Baselines (climatology, persistence, seasonal_mean) are non-stochastic —
they have no seed at all, by construction (see
`planning/MODELING_DOCUMENTATION.md` §2).

## 3. Bootstrap confidence-interval seed

`configs/evaluation.yaml`'s `bootstrap.seed` (`42`) seeds the
`numpy.random.default_rng` used for block-by-cell resampling in
`evaluation/confidence_intervals.py`. Fixed independently of the per-task
model seeds, so re-running evaluation with a different model seed does not
change which cells get resampled into which bootstrap replicate.

## 4. Where seeds are recorded per run

Every `mars.run_all` run writes its full seed strategy into
`manifests/run_manifest.json`'s `seed_strategy` field (see
`mars/run_manifest.py`) — the top-level run seed, the two task seeds, and a
pointer to the bootstrap seed's config location. Real example, from the
completed real run:

```json
{
  "global_generation_seed": 42,
  "task_seeds": {"dust_regression": 42, "storm_detection": 44},
  "bootstrap_seed": "see configs/evaluation.yaml bootstrap.seed",
  "note": "see planning/SEED_STRATEGY.md for the full rationale"
}
```

Written once per run and never silently changes mid-run.

## 5. What is NOT seeded (and why that's fine)

`SimpleImputer(strategy="median")` and the manual sequence-window imputer
(`modeling/train.py::_impute_sequence_train_only`) are deterministic given
their input data — no RNG involved, so nothing to seed.
`sklearn.ensemble.GradientBoosting*`'s internal subsampling is NOT enabled
in this pipeline's hyperparameters (`subsample=1.0`, the default, in every
`configs/task_*.yaml`), so no additional seed is needed there either — if a
future config change enables subsampling, that config's `random_state` (the
per-task model seed) already covers it, since scikit-learn's `random_state`
parameter governs both tree randomness AND subsampling randomness for
`GradientBoostingRegressor`/`Classifier`.
