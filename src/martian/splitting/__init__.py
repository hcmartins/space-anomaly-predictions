"""Chronological train/val/test splitting for Phase 5 modeling.

Unlike labeling/reporting.py's `assign_split` (explicitly scoped there as
"reporting-only, not leak-audited"), this package IS the leak-audited split:
it asserts no storm episode spans a split boundary, and provides
train-only standardisation-statistic fitting so no val/test information
ever leaks into a fitted scaler.
"""
