"""TSS-maximizing threshold selection, reused for both the fused and
unimodal models -- Section 5.6.2's explicit requirement that both models'
thresholds be "selected independently on their own validation data (never
a shared threshold)". Reuses `negative_transfer.true_skill_statistic`
rather than duplicating the TSS formula a third time in this package.
"""
from __future__ import annotations

import numpy as np

from .negative_transfer import true_skill_statistic


def select_threshold_by_tss(
    y_true, y_prob, *, grid_min: float = 0.01, grid_max: float = 0.99, grid_steps: int = 99
) -> tuple[float, str]:
    """Returns (threshold, source). Falls back to a fixed 0.5, honestly
    labeled as such, when the validation data has only one class present
    (POD or FAR -- and therefore TSS -- is undefined for every candidate
    threshold in that case, never silently fabricated)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5, "fixed_fallback (validation split has a single class present -- TSS is undefined for every threshold)"

    candidates = np.linspace(grid_min, grid_max, grid_steps)
    scores = np.array([true_skill_statistic(y_true, y_prob, t) for t in candidates], dtype=float)
    if np.all(np.isnan(scores)):
        return 0.5, "fixed_fallback (TSS undefined at every candidate threshold)"
    best_idx = int(np.nanargmax(scores))
    return float(candidates[best_idx]), f"validation_selected (maximizing tss, best={scores[best_idx]:.4f})"
