"""Forward-looking flare-occurrence labeling target(s), built from the
FINE-GRAINED (1-min, disk-aggregated) GOES series -- NOT the SHARP-coarsened
aligned series -- to avoid missing a flare peak that falls strictly between
two SHARP timestamps. See configs/labeling.yaml's header for why this looks
into the future (a deliberate, label-only exception to the strictly-causal
rule preprocessing/{alignment,missing_data,features}.py otherwise enforce
everywhere) and why an under-covered horizon is labeled NaN rather than
fabricated as a negative..

Column naming: every label column is prefixed `label_`, mirroring
preprocessing/features.py's `feat_` convention, so feature and label columns
can never be confused by a downstream dataset-builder that filters on
prefix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR

LABEL_PREFIX = "label_"
_CLASS_ORDER = ["B", "C", "M", "X"]  # ascending severity


def load_labeling_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "labeling.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def classify_flux(peak_xrsb_flux_w_m2, thresholds: dict) -> str | None:
    """Returns the highest flare class (X > M > C > B) whose threshold
    peak_xrsb_flux_w_m2 meets or exceeds, or None if it never reaches even
    the B threshold (or peak_xrsb_flux_w_m2 is NaN/None)."""
    if peak_xrsb_flux_w_m2 is None or (isinstance(peak_xrsb_flux_w_m2, float) and np.isnan(peak_xrsb_flux_w_m2)):
        return None
    for cls in reversed(_CLASS_ORDER):
        if peak_xrsb_flux_w_m2 >= thresholds[cls]:
            return cls
    return None


def _class_at_or_above(cls: str | None, target: str) -> float:
    if cls is None:
        return 0.0
    return 1.0 if _CLASS_ORDER.index(cls) >= _CLASS_ORDER.index(target) else 0.0


def build_labels(
    aligned_df: pd.DataFrame,
    goes_disk_df: pd.DataFrame,
    *,
    horizon_hours: float | None = None,
    config: dict | None = None,
    time_col: str = "time_utc",
) -> pd.DataFrame:
    """For each row of aligned_df (SHARP-cadence timestamps), looks FORWARD
    over (t, t+horizon] in the fine-grained goes_disk_df and derives:
      - label_max_flare_class_in_horizon: "X"/"M"/"C"/"B"/None (None means
        the max flux never reached even the B threshold)
      - label_flare_{cls}_or_above for cls in b/c/m/x: 1.0/0.0/NaN
      - label_coverage_fraction: fraction of expected 1-min GOES steps in
        the horizon that were actually observed
    A row's max-class/binary labels are left UNDEFINED (NaN / None) whenever
    label_coverage_fraction < 1 - max_allowed_missing_coverage_fraction --
    see configs/labeling.yaml.
    """
    config = config or load_labeling_config()
    thresholds = config["flare_class_thresholds_w_m2"]
    horizon = pd.Timedelta(hours=horizon_hours if horizon_hours is not None else config["forecast_horizon_hours"])
    min_coverage = 1.0 - config["max_allowed_missing_coverage_fraction"]
    fine_cadence = pd.Timedelta(minutes=config["goes_fine_cadence_minutes"])

    goes_sorted = goes_disk_df.sort_values(time_col).reset_index(drop=True)
    goes_times = pd.to_datetime(goes_sorted[time_col], utc=True).to_numpy()
    xrsb = goes_sorted["xrsb_flux_w_m2"].to_numpy(dtype=float)

    out = aligned_df.sort_values(time_col).reset_index(drop=True).copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True)
    t0 = out[time_col].to_numpy()
    t1 = (out[time_col] + horizon).to_numpy()

    lo_idx = np.searchsorted(goes_times, t0, side="right")  # strictly AFTER t0
    hi_idx = np.searchsorted(goes_times, t1, side="right")  # up to and including t1

    expected_steps = max(int(horizon / fine_cadence), 1)

    n = len(out)
    max_class = np.full(n, None, dtype=object)
    coverage = np.zeros(n)
    binary = {cls: np.full(n, np.nan) for cls in _CLASS_ORDER}

    for i in range(n):
        lo, hi = lo_idx[i], hi_idx[i]
        n_obs = hi - lo
        coverage[i] = min(n_obs / expected_steps, 1.0)
        if coverage[i] < min_coverage:
            continue  # UNDEFINED: max_class stays None, binary stays NaN
        peak = float(np.nanmax(xrsb[lo:hi])) if n_obs > 0 else np.nan
        cls = classify_flux(peak, thresholds)
        max_class[i] = cls
        for target in _CLASS_ORDER:
            binary[target][i] = _class_at_or_above(cls, target)

    out[f"{LABEL_PREFIX}max_flare_class_in_horizon"] = max_class
    out[f"{LABEL_PREFIX}coverage_fraction"] = coverage
    for cls in _CLASS_ORDER:
        out[f"{LABEL_PREFIX}flare_{cls.lower()}_or_above"] = binary[cls]
    out[f"{LABEL_PREFIX}horizon_hours"] = float(horizon / pd.Timedelta(hours=1))
    return out


def build_primary_label(
    aligned_df: pd.DataFrame,
    goes_disk_df: pd.DataFrame,
    *,
    config: dict | None = None,
) -> pd.DataFrame:
    config = config or load_labeling_config()
    out = build_labels(aligned_df, goes_disk_df, config=config)
    primary = config["primary_target_class"]
    out[f"{LABEL_PREFIX}primary"] = out[f"{LABEL_PREFIX}flare_{primary.lower()}_or_above"]
    return out


def run_horizon_sensitivity(
    aligned_df: pd.DataFrame,
    goes_disk_df: pd.DataFrame,
    *,
    config: dict | None = None,
) -> pd.DataFrame:
    """Sweeps configs/labeling.yaml's sensitivity_horizons_hours, reporting
    per-class positive-rate and coverage stats at each horizon."""
    config = config or load_labeling_config()
    rows = []
    for h in config["sensitivity_horizons_hours"]:
        labeled = build_labels(aligned_df, goes_disk_df, horizon_hours=h, config=config)
        for cls in _CLASS_ORDER:
            col = f"{LABEL_PREFIX}flare_{cls.lower()}_or_above"
            defined = labeled[col].notna()
            rows.append(
                {
                    "horizon_hours": h,
                    "class": cls,
                    "n_total": len(labeled),
                    "n_defined": int(defined.sum()),
                    "n_positive": int(labeled.loc[defined, col].sum()) if defined.any() else 0,
                    "positive_rate": float(labeled.loc[defined, col].mean()) if defined.any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)
