"""Shared per-domain evaluation core: threshold selection, TSS, negative-
transfer, and failure-case cataloguing, given real prediction ARRAYS
directly (not a model object) -- usable by both run_fusion.py (fused vs.
unimodal-baseline comparison) and run_ablation.py (any of A1-A7's
predictions vs. the real A1 unimodal baseline), so this logic exists
exactly once. Originally private to run_fusion.py::_evaluate_domain;
factored out here unchanged in behavior."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .failure_cases import catalogue_failure_cases
from .negative_transfer import negative_transfer_by_regime, true_skill_statistic
from .thresholds import select_threshold_by_tss

_EMPTY_FAILURE_CASES_COLUMNS = [
    "failure_type", "y_true", "y_prob_fused", "y_prob_unimodal", "shared_with_unimodal",
    "alpha_solar_at_issuance", "alpha_martian_at_issuance",
]


def evaluate_predictions(
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
    y_prob_fused_val: np.ndarray, y_prob_fused_test: np.ndarray,
    y_prob_unimodal_val: np.ndarray, y_prob_unimodal_test: np.ndarray,
    regime_test: np.ndarray,
    *,
    alpha_fused_test: np.ndarray | None = None,
    predictions_dir: Path | None = None, config_id: str | None = None,
    domain: str | None = None, seed: int | None = None,
) -> dict:
    """`alpha_fused_test`: (n_test, 2) `[alpha_solar, alpha_martian]` at
    issuance, or None for a configuration with no attention weights (A1,
    A2, A5's use_fusion_pathway=False) -- failure-case cataloguing is then
    skipped (an empty, clearly-labeled table), never fabricated with
    placeholder attention values.

    `predictions_dir`/`config_id`/`domain`/`seed`: when all four are given,
    also emits a `predictions_<config_id>_<domain>_seed<seed>.csv` of this
    config's own fused test-set predictions, in the exact format
    `uncertainty_analysis.py`'s module docstring specifies -- the real
    per-instance record the Section 5.6.1 uncertainty protocol (seed
    variation, block bootstrap, paired comparison) is computed from. This
    writes the FUSED prediction (`y_prob_fused_test`), not the unimodal
    one, since the caller already treats "this config's own predictions"
    as what's fused for that config (A1 itself is a config too, and its
    own "fused" slot is simply its unimodal head's output)."""
    threshold_fused, threshold_fused_source = select_threshold_by_tss(y_val, y_prob_fused_val)
    threshold_unimodal, threshold_unimodal_source = select_threshold_by_tss(y_val, y_prob_unimodal_val)

    tss_fused = true_skill_statistic(y_test, y_prob_fused_test, threshold_fused)
    tss_unimodal = true_skill_statistic(y_test, y_prob_unimodal_test, threshold_unimodal)

    negative_transfer_table = negative_transfer_by_regime(
        y_test, y_prob_fused_test, y_prob_unimodal_test, regime_test,
        threshold_fused=threshold_fused, threshold_unimodal=threshold_unimodal,
    )

    if alpha_fused_test is None:
        failure_cases = pd.DataFrame(columns=_EMPTY_FAILURE_CASES_COLUMNS)
        failure_cases_note = "not applicable (no attention weights in this configuration)"
    else:
        failure_cases = catalogue_failure_cases(
            y_test, y_prob_fused_test, y_prob_unimodal_test, alpha_fused_test,
            threshold_fused=threshold_fused, threshold_unimodal=threshold_unimodal,
            false_alarm_probability_threshold=0.5,
        )
        failure_cases_note = None

    if predictions_dir is not None and config_id is not None and domain is not None and seed is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"y_true": y_test, "y_prob": y_prob_fused_test,
                      "threshold": threshold_fused, "regime": regime_test}
                     ).to_csv(predictions_dir / f"predictions_{config_id}_{domain}_seed{seed:02d}.csv",
                              index=False)

    return {
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
        "train_prevalence": float(np.mean(y_train)) if len(y_train) else float("nan"),
        "val_prevalence": float(np.mean(y_val)) if len(y_val) else float("nan"),
        "test_prevalence": float(np.mean(y_test)) if len(y_test) else float("nan"),
        "threshold_fused": threshold_fused, "threshold_fused_source": threshold_fused_source,
        "threshold_unimodal": threshold_unimodal, "threshold_unimodal_source": threshold_unimodal_source,
        "tss_fused": tss_fused, "tss_unimodal": tss_unimodal,
        "n_negative_transfer_regimes": int((negative_transfer_table["negative_transfer"] == True).sum()),  # noqa: E712
        "n_regimes_evaluated": len(negative_transfer_table),
        "n_failure_cases": len(failure_cases),
        "failure_cases_note": failure_cases_note,
        "negative_transfer_table": negative_transfer_table,
        "failure_cases": failure_cases,
    }
