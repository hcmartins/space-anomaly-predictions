"""SHARP robust-scaling preprocessing (Chapter 3.4.2, Algorithm 3.2):
Winsorize each SHARP-derived feature column to train-only percentile bounds,
then apply median/IQR ("robust") scaling. Both statistics are fit on the
TRAIN split ONLY and applied unchanged to val/test -- the same train-only-fit
discipline modeling/train.py's SimpleImputer already follows..

Motivation (real, observed bug): logistic regression has failed to converge
(TSS=0, ROC-AUC=0.5 exactly, all-zero coefficients) in every real run so far
(see planning/MODELING_DOCUMENTATION.md and
planning/FLARE_CATALOGUE_DOCUMENTATION.md) -- SHARP physical indices span
wildly different raw magnitudes (USFLUX ~1e22 vs R_VALUE ~0-5) with nothing
scaling them before fitting. Scope is SHARP-derived feature columns only (see
sharp_feature_columns) -- matching Algorithm 3.2's scope in Chapter 3.4.2.
GOES gets its own, separately-tracked preprocessing chain (Algorithm 3.1:
log-transform/despiking/smoothing/background-normalization), not yet
implemented.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR, SHARP_PHYSICAL_KEYWORDS

from .features import FEATURE_PREFIX, load_features_config


def load_scaling_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "scaling.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def sharp_feature_columns(feature_manifest: list[str], features_config: dict | None = None) -> list[str]:
    """Every column in feature_manifest whose underlying source variable is
    one of the 18 SHARP physical indices -- both the feat_-prefixed
    lag/trailing/diff/slope derivatives (found by regenerating the exact
    column names preprocessing/features.py::build_features would have
    produced for each such variable, not by regex-parsing feature names,
    which would be ambiguous given underscore-containing variable names
    like goes_xrsb_flux_trailing_max_w_m2) AND the bare current-state
    column itself (e.g. "USFLUX", "R_VALUE").

    The bare names matter, not just a completeness nicety: modeling/
    datasets.py::build_flare_forecast_dataset deliberately includes each
    configured variable's raw CURRENT value in feature_manifest alongside
    its derived features (`raw_current_state_cols`) -- the model's reading
    of the series at t0 itself, not just its lagged/trailing history. A
    real full-range run (2026-07-28) showed a SHARP record with
    SAVNCPP ~= 2.18e38 (suspiciously close to float32's max -- plausibly an
    unflagged JSOC fill-value artifact that passes the non_finite_value QC
    rule, which only rejects literal +/-inf, not implausible finite
    magnitudes -- see preprocessing/quality_control.py's
    _check_non_finite_value docstring, which is itself a known, not yet
    tightened gap) reaching straight into logistic regression completely
    unscaled via this bare column, even after the feat_-derived columns
    were correctly scaled -- LR failed identically. Winsorizing the bare
    column too clips this kind of rare extreme artifact the same way it
    already handles the derived columns."""
    features_config = features_config or load_features_config()
    window = features_config["trailing_window_steps"]
    lags = features_config["lags_steps"]
    expected: list[str] = list(SHARP_PHYSICAL_KEYWORDS)
    for var in SHARP_PHYSICAL_KEYWORDS:
        if features_config.get("enable_lags"):
            expected += [f"{FEATURE_PREFIX}{var}_lag{lag}" for lag in lags]
        if features_config.get("enable_trailing_stats"):
            expected += [f"{FEATURE_PREFIX}{var}_trailing{window}_{stat}" for stat in ("mean", "std", "max", "min")]
        if features_config.get("enable_first_difference_and_slope"):
            expected += [f"{FEATURE_PREFIX}{var}_diff1", f"{FEATURE_PREFIX}{var}_slope{window}"]
    manifest_set = set(feature_manifest)
    return [c for c in expected if c in manifest_set]


def fit_robust_scaling(X_train: pd.DataFrame, columns: list[str], config: dict | None = None) -> dict:
    """Fits per-column Winsorize bounds + median/IQR on TRAIN data only.
    NaNs (real, unfilled gaps) are ignored via nanpercentile/nanmedian, never
    fabricated -- a column that is entirely NaN in train is left out of the
    returned params (apply_robust_scaling then leaves it untouched). A
    column with IQR == 0 after clipping (constant in-train) gets scale = 1.0
    -- a documented divide-by-zero guard, not a fabricated spread."""
    config = config or load_scaling_config()
    lower_pct = config["winsorize_lower_percentile"]
    upper_pct = config["winsorize_upper_percentile"]

    params: dict[str, dict[str, float]] = {}
    for col in columns:
        if col not in X_train.columns:
            continue
        values = X_train[col].to_numpy(dtype=float)
        if np.all(np.isnan(values)):
            continue
        lower, upper = np.nanpercentile(values, [lower_pct, upper_pct])
        clipped = np.clip(values, lower, upper)
        median = float(np.nanmedian(clipped))
        q25, q75 = np.nanpercentile(clipped, [25, 75])
        iqr = float(q75 - q25)
        params[col] = {
            "lower": float(lower),
            "upper": float(upper),
            "median": median,
            "scale": iqr if iqr > 0 else 1.0,
        }
    return params


def apply_robust_scaling(X: pd.DataFrame, params: dict, *, copy: bool = True) -> pd.DataFrame:
    """Clips each column present in params to its fitted [lower, upper]
    bounds, then applies (x - median) / scale. Columns not in params (not
    SHARP-derived, or all-NaN in train) pass through unchanged.

    copy=True (default) never mutates the caller's frame -- safe for
    general callers, including every existing test. copy=False mutates X
    in place and returns it: a real fix for a real ArrayMemoryError hit at
    the per-HARP pipeline's ~1.97M-row train split (2026-07-28) -- an
    unconditional X.copy() here forces pandas 3.0's copy-on-write to
    materialize ALL ~257 feature columns upfront, on top of the already
    ~13GB-resident per-HARP dataset_with_split at this machine's 15.6GB RAM
    ceiling. modeling/train.py uses copy=False only where it already owns a
    disposable frame (X_train_raw / a per-split slice, never read again
    afterward) -- pandas' CoW then only copies the specific column blocks
    actually being modified, incrementally, instead of the whole frame at
    once."""
    out = X.copy() if copy else X
    for col, p in params.items():
        if col not in out.columns:
            continue
        clipped = out[col].clip(lower=p["lower"], upper=p["upper"])
        out[col] = (clipped - p["median"]) / p["scale"]
    return out
