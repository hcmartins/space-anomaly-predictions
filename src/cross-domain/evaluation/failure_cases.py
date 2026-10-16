"""Failure-case cataloguing (Section 5.6.3): a fixed selection rule --
every missed event at the operational threshold, and every false alarm
above a configured probability -- catalogued with both the unimodal and
fused predictions plus the attention weights at issuance. Answers the two
questions Section 5.6.3 structures failure analysis around directly (per
row, via `shared_with_unimodal`): a shared miss/false-alarm indicates a
representation problem, not a fusion problem; a fusion-only one is a
candidate negative-transfer case (Section 5.6.2).

The Chapter 6/7 presentation/reporting layer is explicitly out of scope
for this delivery (see README.md) -- this module produces the real
catalogue, not the narrative around it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def catalogue_failure_cases(
    y_true,
    y_prob_fused,
    y_prob_unimodal,
    alpha_fused,
    *,
    threshold_fused: float,
    threshold_unimodal: float,
    false_alarm_probability_threshold: float,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """alpha_fused: (n, 2) attention weights [alpha_solar, alpha_martian]
    at issuance for this task's own scorer (fusion/attention.py::fuse's
    second return value). metadata (optional): per-instance identifying
    columns (run id, date/sol, regime) carried through into the
    catalogue, per Section 5.6.3's "catalogued with its run identifier,
    date or sol, and regime" requirement."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob_fused = np.asarray(y_prob_fused, dtype=float)
    y_prob_unimodal = np.asarray(y_prob_unimodal, dtype=float)

    pred_fused = (y_prob_fused >= threshold_fused).astype(int)
    pred_unimodal = (y_prob_unimodal >= threshold_unimodal).astype(int)

    missed_event = (y_true == 1) & (pred_fused == 0)
    false_alarm = (y_true == 0) & (y_prob_fused > false_alarm_probability_threshold)
    selected = missed_event | false_alarm

    unimodal_missed = (y_true == 1) & (pred_unimodal == 0)
    unimodal_false_alarm = (y_true == 0) & (y_prob_unimodal > false_alarm_probability_threshold)
    shared_with_unimodal = np.where(missed_event, unimodal_missed, unimodal_false_alarm)

    catalogue = pd.DataFrame(
        {
            "failure_type": np.where(missed_event, "missed_event", "false_alarm")[selected],
            "y_true": y_true[selected],
            "y_prob_fused": y_prob_fused[selected],
            "y_prob_unimodal": y_prob_unimodal[selected],
            "shared_with_unimodal": shared_with_unimodal[selected],
            "alpha_solar_at_issuance": np.asarray(alpha_fused)[selected, 0],
            "alpha_martian_at_issuance": np.asarray(alpha_fused)[selected, 1],
        }
    )
    if metadata is not None:
        meta_selected = metadata.reset_index(drop=True).loc[selected].reset_index(drop=True)
        catalogue = pd.concat([meta_selected, catalogue], axis=1)
    return catalogue
