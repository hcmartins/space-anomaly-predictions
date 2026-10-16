"""Preprocessing for real GOES + SHARP data: quality control
(`quality_control.py`), disk-level alignment (`alignment.py` --
satellite/active-region aggregation onto a common timeline), causal
missing-data handling (`missing_data.py`), and feature engineering
(`features.py`). Operates on the normalized DataFrames produced by
`data_sources/goes_ingestion.py` and `data_sources/sharp_ingestion.py` --
see `planning/PREPROCESSING_DOCUMENTATION.md` for the engineering decisions
made here that are NOT extracted from the thesis text (there is no real
per-active-region flare attribution catalogue ingested yet, so this
operates on DISK-INTEGRATED series -- see that document for the full
rationale) and for this pass's bug-fix log.

Every module in this package is strictly causal (no value at time t ever
depends on an observation after t). Forward-looking flare-occurrence labels
are a separate, deliberate exception living in `labeling/`, never imported
here -- see `planning/LABELING_DOCUMENTATION.md`.
"""
