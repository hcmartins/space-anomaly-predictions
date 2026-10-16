"""Solar flare forecasting package (Chapter 3: "Solar Flare Forecasting
Methodology").

This package implements the solar-domain counterpart to `martian/`, using the
SAME engineering conventions established there: config-driven parameters
(never hardcoded), explicit provenance stamping on every artefact, honest
labelling of anything not yet implemented or not yet verified, and no
fabricated results.

Current scope (this delivery): real data ingestion, quality control,
disk-level alignment, causal gap-filling, feature engineering,
forward-looking flare-occurrence labeling, leak-audited chronological
splitting, and baseline/classical-model training.

    cd src/solar
    python -m ingest --config configs/solar_ingest.yaml

fetches real GOES X-Ray Sensor (XRS) soft X-ray irradiance data from NOAA's
public archives and real SHARP (Space-weather HMI Active Region Patch) data
from Stanford's JSOC, and writes them to a local, checksummed, provenancestamped cache. Unlike the Martian chapter (where no real EMIRS/MCD provider
could be implemented in this environment), both GOES and SHARP are genuinely
public, freely downloadable datasets, so this ingestion is REAL, not
simulated.

`preprocessing/` (quality control -> disk-level alignment -> causal
gap-filling -> feature engineering), `labeling/` (forward-looking
flare-occurrence targets), `modeling/datasets.py` (the features+labels
join, chained into leak-audited splitting via
`assign_split_with_embargo`), `splitting/` (leak-audited, embargoed
chronological train/val/test splitting), and `modeling/{metrics,baselines,
classical_models,registry,train}.py` (baseline and classical-model
fitting/evaluation) are implemented and unit-tested — see
`planning/PREPROCESSING_DOCUMENTATION.md`,
`planning/LABELING_DOCUMENTATION.md`, `planning/DATASETS_DOCUMENTATION.md`,
`planning/SPLITTING_DOCUMENTATION.md`, and
`planning/MODELING_DOCUMENTATION.md`. None has been run end-to-end against
a real, full-scale ingested dataset yet (only against small synthetic
DataFrames in tests, including an end-to-end synthetic-data run of the
full alignment -> missing-data -> features -> labeling -> dataset-join ->
split -> train chain). Sequence models, an `evaluation/` package
(confidence intervals, figures, failure analysis), and a full
provenance-record schema (Chapter 3 §3.4's remaining pieces / Chapter 4)
are NOT part of this delivery and are not implemented yet — see those five
documents' "What is NOT implemented" sections for the precise boundary. No
chapter text should cite results from this package until those later
stages exist and have actually been run.
"""
