"""Per-retrieval harmonisation of EMIRS dust optical depth to the Montabone
reference convention (configs/harmonisation.yaml):

    tau*_i = (tau_ext,i / f) * (610 Pa / p_s,i)

f = 1.3 is the 9.3-um extinction-to-absorption ratio from Montabone et al.
(2015, Icarus 251, 65-95, Sec. 2.3.4; doi:10.1016/j.icarus.2014.12.034), as
documented on the LMD Mars dust climatology product page; the 610 Pa
normalisation follows the same source (local OD = cdod610 x p_s/610).

Applied to QC-accepted retrievals BEFORE cell-sol aggregation, so the
aggregated dust_optical_depth is in the same units as the climatology
(mu_clim/sigma_clim from Montabone cdod610) it is standardised against.
Fails loudly (never silently skips or imputes) if any dust-bearing retrieval
lacks a positive surface pressure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR


def load_harmonisation_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "harmonisation.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def harmonisation_factor(pressure: np.ndarray, config: dict) -> np.ndarray:
    p_pa = np.asarray(pressure, dtype=float) * float(config["pressure_column_to_pa"])
    return (1.0 / float(config["extinction_to_absorption_ratio"])) * (float(config["reference_pressure_pa"]) / p_pa)


def harmonise_dust_to_reference(df: pd.DataFrame, config: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (harmonised copy, summary). Identity (with summary
    {"enabled": False}) when the config disables harmonisation."""
    config = config or load_harmonisation_config()
    if not config.get("enabled", False):
        return df.copy(), {"enabled": False}
    var, unc, pcol = config["variable"], config["uncertainty_variable"], config["pressure_column"]
    for col in (var, unc, pcol):
        if col not in df.columns:
            raise KeyError(f"harmonisation requires column {col!r}")
    has_dust = df[var].notna().to_numpy()
    p = df[pcol].to_numpy(dtype=float)
    bad = has_dust & ~(np.isfinite(p) & (p > 0))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} dust-bearing retrievals have a missing or non-positive {pcol!r}; "
                         "cannot harmonise them and will not impute a pressure")
    out = df.copy()
    factor = np.full(len(out), np.nan)
    factor[has_dust] = harmonisation_factor(p[has_dust], config)
    out[var] = out[var].to_numpy(dtype=float) * np.where(has_dust, factor, 1.0)
    out[unc] = out[unc].to_numpy(dtype=float) * np.where(has_dust, factor, 1.0)
    f = factor[has_dust]
    summary = {
        "enabled": True, "n_retrievals_harmonised": int(has_dust.sum()),
        "extinction_to_absorption_ratio": float(config["extinction_to_absorption_ratio"]),
        "reference_pressure_pa": float(config["reference_pressure_pa"]),
        "factor_p05_p50_p95": [float(x) for x in np.percentile(f, [5, 50, 95])] if len(f) else None,
        "pressure_mbar_min_median_max": [float(np.min(p[has_dust])), float(np.median(p[has_dust])), float(np.max(p[has_dust]))] if len(f) else None,
    }
    return out, summary
