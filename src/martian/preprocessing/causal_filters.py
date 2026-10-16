"""Causal (trailing) and centred (descriptive-only) Savitzky-Golay filtering.

The causal filter is the ONLY variant permitted to feed a model: at sol s it
uses only observations at sol <= s (trailing coefficient position), so a
change to any future observation can never alter a past causal output --
enforced structurally by pandas' trailing (non-centred) rolling window, and
proven by tests/test_causal_filters.py's perturbation test.

The centred filter exists ONLY for descriptive, side-by-side comparison
figures (e.g. Figure 4.3) and must NEVER be used as a model feature or
label. This module keeps the two implementations structurally separate
(distinct functions, distinct output-column suffix) and provides
`assert_feature_safe` so any code assembling a model feature matrix can
reject a centred column outright rather than relying on convention alone.

Both variants require an unbroken, sol-contiguous window (no internal gap,
no NaN) to produce a value; otherwise the output is NaN, per the same
"never fabricate a defensible value from insufficient data" principle used
in preprocessing/missing_data.py and preprocessing/quality_control.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs

from data_sources.schema import CONFIG_DIR

CAUSAL_COLUMN_SUFFIX = "_sg_causal"
CENTERED_COLUMN_SUFFIX = "_sg_centered_DESCRIPTIVE_ONLY"


class DescriptiveOnlyColumnError(ValueError):
    """Raised when a column produced by the descriptive-only centred filter
    is passed to code assembling model features/labels."""


def load_causal_filter_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "causal_filter.yaml"
    import yaml

    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def assert_feature_safe(columns) -> None:
    """Raise DescriptiveOnlyColumnError if any column name in `columns` is a
    descriptive-only (centred) filter output. Call this on the final feature
    (and label) column list before fitting/evaluating any model."""
    bad = [c for c in columns if CENTERED_COLUMN_SUFFIX in c]
    if bad:
        raise DescriptiveOnlyColumnError(
            f"Descriptive-only centred-filter column(s) {bad!r} must never be "
            f"used as a model feature or label -- see "
            f"preprocessing/causal_filters.py."
        )


def _rolling_dot(values: np.ndarray, coeffs: np.ndarray) -> float:
    if np.any(np.isnan(values)):
        return np.nan
    return float(np.dot(coeffs, values))


def _contiguity_mask(sol: pd.Series, window_length: int, *, center: bool) -> pd.Series:
    span = sol.rolling(window_length, center=center).apply(lambda w: w[-1] - w[0], raw=True)
    return span == (window_length - 1)


def _filter_one_series(
    group: pd.DataFrame,
    *,
    sol_col: str,
    value_col: str,
    window_length: int,
    polyorder: int,
    center: bool,
    output_col: str,
) -> pd.Series:
    group = group.sort_values(sol_col)
    pos = None if center else window_length - 1
    coeffs = savgol_coeffs(window_length, polyorder, pos=pos, use="dot")

    raw = group[value_col].rolling(window_length, center=center).apply(
        lambda w: _rolling_dot(w, coeffs), raw=True
    )
    valid = _contiguity_mask(group[sol_col], window_length, center=center)
    return raw.where(valid, np.nan).reindex(group.index)


def apply_causal_savgol(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("cell_id", "mars_year"),
    sol_col: str = "sol",
    value_col: str | None = None,
    window_length: int | None = None,
    polyorder: int | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Trailing-position causal Savitzky-Golay filter. Output column:
    `<value_col>_sg_causal`. Uses only observations at or before the current
    sol -- no future observation is ever read, per group, per the trailing
    (non-centred) rolling window and coefficient position."""
    config = config or load_causal_filter_config()
    value_col = value_col or config["variable"]
    window_length = window_length or config["window_length"]
    polyorder = polyorder or config["polyorder"]

    out = df.copy()
    output_col = f"{value_col}{CAUSAL_COLUMN_SUFFIX}"
    out[output_col] = np.nan
    for _, group in df.groupby(list(group_cols), sort=False):
        series = _filter_one_series(
            group,
            sol_col=sol_col,
            value_col=value_col,
            window_length=window_length,
            polyorder=polyorder,
            center=False,
            output_col=output_col,
        )
        out.loc[series.index, output_col] = series
    return out


def apply_centered_savgol_for_descriptive_comparison_only(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("cell_id", "mars_year"),
    sol_col: str = "sol",
    value_col: str | None = None,
    window_length: int | None = None,
    polyorder: int | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Centred Savitzky-Golay filter -- DESCRIPTIVE COMPARISON ONLY. Reads
    observations both before AND after the current sol. Output column name
    (`<value_col>_sg_centered_DESCRIPTIVE_ONLY`) is deliberately verbose so
    misuse as a feature is loud; call `assert_feature_safe` on any feature
    column list to enforce this structurally rather than by convention."""
    config = config or load_causal_filter_config()
    value_col = value_col or config["variable"]
    window_length = window_length or config["window_length"]
    polyorder = polyorder or config["polyorder"]

    out = df.copy()
    output_col = f"{value_col}{CENTERED_COLUMN_SUFFIX}"
    out[output_col] = np.nan
    for _, group in df.groupby(list(group_cols), sort=False):
        series = _filter_one_series(
            group,
            sol_col=sol_col,
            value_col=value_col,
            window_length=window_length,
            polyorder=polyorder,
            center=True,
            output_col=output_col,
        )
        out.loc[series.index, output_col] = series
    return out
