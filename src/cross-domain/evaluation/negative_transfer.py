"""Negative-transfer analysis (Section 5.6.2):

Delta_t(r) = M_t^fused(r) - M_t^unimodal(r)

reported per regime r, where M is TSS (the primary skill metric for both
this project's real classification tasks -- solar's flare forecasting and
martian's storm detection already use TSS as their headline metric).
Negative transfer is declared when Delta_t(r) < 0. Both models use
thresholds selected independently on their own validation data (never a
shared threshold) -- Section 5.6.2's own explicit requirement.

TSS is reproduced locally (four lines from confusion counts) rather than
imported from either domain package -- see README.md's decoupling
decision; this is small enough that duplicating it is cheaper and safer
than the sys.path collision importing either package's metrics module
would risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_skill_statistic(y_true, y_prob, threshold: float) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    pod = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    far = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    if np.isnan(pod) or np.isnan(far):
        return float("nan")
    return pod - far


def negative_transfer_by_regime(
    y_true,
    y_prob_fused,
    y_prob_unimodal,
    regime,
    *,
    threshold_fused: float,
    threshold_unimodal: float,
) -> pd.DataFrame:
    """One row per distinct value of `regime` (e.g. solar cycle
    phase/activity level, or martian season/dust regime): n, both models'
    TSS, Delta_t(r), and whether that regime shows negative transfer.
    Never silently skips a regime with too little data -- TSS/Delta are
    NaN (not fabricated) when either class is entirely absent."""
    df = pd.DataFrame(
        {"y_true": y_true, "y_prob_fused": y_prob_fused, "y_prob_unimodal": y_prob_unimodal, "regime": regime}
    )
    rows = []
    for regime_value, group in df.groupby("regime"):
        tss_fused = true_skill_statistic(group["y_true"], group["y_prob_fused"], threshold_fused)
        tss_unimodal = true_skill_statistic(group["y_true"], group["y_prob_unimodal"], threshold_unimodal)
        delta = tss_fused - tss_unimodal if not (np.isnan(tss_fused) or np.isnan(tss_unimodal)) else float("nan")
        rows.append(
            {
                "regime": regime_value,
                "n": len(group),
                "tss_fused": tss_fused,
                "tss_unimodal": tss_unimodal,
                "delta": delta,
                "negative_transfer": bool(delta < 0) if not np.isnan(delta) else None,
            }
        )
    return pd.DataFrame(rows)
