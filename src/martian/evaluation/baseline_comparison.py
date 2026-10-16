"""Compares every learned (classical or sequence) model against its task's
"appropriate baseline" (configs/evaluation.yaml's `primary_baseline_by_task`
-- persistence for the two regression tasks, climatology for storm
detection), producing a side-by-side table with an explicit skill-score
column so "does the model actually beat the reference" is answered directly
rather than left for a reader to compute by eye.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Metrics where a LOWER value is better -- skill score flips sign for these
# so "positive skill" always means "model is better than baseline" for
# every metric in the table, regression or classification.
_LOWER_IS_BETTER = {"mae", "rmse", "mse", "brier", "log_score", "far"}


def compare_to_baseline(model_metrics: dict, baseline_metrics: dict) -> pd.DataFrame:
    rows = []
    for key in model_metrics:
        if key not in baseline_metrics:
            continue
        model_value = model_metrics[key]
        baseline_value = baseline_metrics[key]
        if not isinstance(model_value, (int, float)) or not isinstance(baseline_value, (int, float)):
            continue
        if isinstance(model_value, bool) or np.isnan(model_value) or np.isnan(baseline_value):
            skill = float("nan")
        elif key in _LOWER_IS_BETTER:
            skill = (baseline_value - model_value) / abs(baseline_value) if baseline_value != 0 else float("nan")
        else:
            skill = (model_value - baseline_value) / abs(baseline_value) if baseline_value != 0 else float("nan")
        rows.append(
            {
                "metric": key,
                "model": model_value,
                "baseline": baseline_value,
                "relative_skill": skill,
                "model_beats_baseline": (
                    (model_value < baseline_value if key in _LOWER_IS_BETTER else model_value > baseline_value)
                    if not (isinstance(model_value, float) and np.isnan(model_value)) else None
                ),
            }
        )
    return pd.DataFrame(rows)
