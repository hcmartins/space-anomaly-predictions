"""Modeling for the solar flare-forecasting pipeline: dataset construction
(`datasets.py` -- joins `preprocessing/features.py`'s feature columns with
`labeling/flare_labels.py`'s label columns, then chains into leak-audited
splitting via `assign_split_with_embargo`) and baseline/classical-model
fitting and evaluation (`metrics.py`, `baselines.py`,
`classical_models.py`, `registry.py`, `train.py`). See
`planning/DATASETS_DOCUMENTATION.md` and
`planning/MODELING_DOCUMENTATION.md` for the full design rationale and the
precise "What is NOT implemented" boundary (no sequence models, no
`evaluation/` package, no full provenance-record schema yet -- this is a
deliberately scoped-down counterpart to `martian/modeling/`).
"""
