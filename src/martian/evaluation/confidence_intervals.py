"""Bootstrap confidence intervals for regression and classification metrics.

Block-by-cell rather than a plain i.i.d. row bootstrap: nearby cell-sols are
spatially/temporally correlated, so resampling whole cells (with
replacement) is a closer approximation to independent resampling than
resampling individual rows would be -- see configs/evaluation.yaml's header
comment for the adaptation this makes from the originally-planned single-
time-series temporal-block bootstrap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR
from modeling.metrics import classification_metrics, regression_metrics


def load_evaluation_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "evaluation.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _resample_indices(n: int, block_ids: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    if block_ids is None:
        return rng.integers(0, n, size=n)
    unique_blocks = np.unique(block_ids)
    sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
    index_by_block = {b: np.where(block_ids == b)[0] for b in unique_blocks}
    return np.concatenate([index_by_block[b] for b in sampled_blocks])


def bootstrap_ci_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    block_ids: np.ndarray | None = None,
    config: dict | None = None,
) -> dict[str, tuple[float, float]]:
    """Returns {metric_name: (ci_low, ci_high)} for every regression metric."""
    config = config or load_evaluation_config()
    boot_config = config["bootstrap"]
    rng = np.random.default_rng(boot_config["seed"])
    n = len(y_true)

    replicate_metrics: list[dict] = []
    for _ in range(boot_config["n_replicates"]):
        idx = _resample_indices(n, block_ids, rng)
        replicate_metrics.append(regression_metrics(y_true[idx], y_pred[idx]))

    return _percentile_ci(replicate_metrics, boot_config["ci_level"])


def bootstrap_ci_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    block_ids: np.ndarray | None = None,
    config: dict | None = None,
) -> dict[str, tuple[float, float]]:
    """Returns {metric_name: (ci_low, ci_high)} for every classification
    metric. A replicate with only one class present yields NaN for AUC-based
    metrics (same convention as modeling.metrics), excluded from that
    metric's percentile calculation."""
    config = config or load_evaluation_config()
    boot_config = config["bootstrap"]
    rng = np.random.default_rng(boot_config["seed"])
    n = len(y_true)

    replicate_metrics: list[dict] = []
    for _ in range(boot_config["n_replicates"]):
        idx = _resample_indices(n, block_ids, rng)
        replicate_metrics.append(classification_metrics(y_true[idx], y_prob[idx], threshold=threshold))

    return _percentile_ci(replicate_metrics, boot_config["ci_level"])


def _percentile_ci(replicate_metrics: list[dict], ci_level: float) -> dict[str, tuple[float, float]]:
    if not replicate_metrics:
        return {}
    alpha = 1.0 - ci_level
    lower_q, upper_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)

    out: dict[str, tuple[float, float]] = {}
    for key in replicate_metrics[0]:
        values = np.array([r[key] for r in replicate_metrics], dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            out[key] = (float("nan"), float("nan"))
        else:
            out[key] = (float(np.percentile(values, lower_q)), float(np.percentile(values, upper_q)))
    return out
