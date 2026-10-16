"""One reproducible entry point for the whole Martian Chapter 4 pipeline:

    python -m mars.run_all --config configs/run.yaml

Orchestrates, in order: input validation, run-manifest writing, QC +
gridding, cell-sol aggregation, causal missing-data handling, causal
filtering, climatological deviation, label/episode construction, causal
feature engineering, chronological splitting, runtime leakage checks,
baseline+model training, per-task/per-horizon evaluation, uncertainty
intervals, figures/tables, a final report, and checksums for every output
file. Every step reuses the already-tested library functions from
data_sources/, preprocessing/, labeling/, splitting/, modeling/, and
evaluation/ -- this package adds orchestration, run isolation, and
reproducibility bookkeeping, not new scientific logic. Real data only --
requires real downloaded EMIRS FITS files (see
configs/real_emirs_ingestion.yaml).
"""
