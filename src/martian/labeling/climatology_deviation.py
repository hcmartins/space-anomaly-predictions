"""Climatological deviation z-score (Eq. 4.1).

    z_v(c,s) = (x_v(c,s) - mu_v(c,s)) / max(sigma_v(c,s), epsilon)

`mu_v`/`sigma_v` come from an `MCDClimatologyProvider.query()` call
(`data_sources/real_mcd_provider.py`, built from real Montabone et al. dust
climatology data) evaluated at each cell-sol's own (latitude, longitude,
solar_longitude_ls, local_true_solar_time). The `max(sigma, epsilon)` floor
prevents division blow-up if a climatology model ever returns a near-zero
sigma. Dust only -- no real climatological source exists for
surface_temperature_k.

Operates ONLY on directly observed (non-interpolated) cell-sols by
convention of the caller -- this module itself does not know or care
whether a row was trailing-filled; passing filled rows through would flag
our own carried-forward values as "anomalous," which
preprocessing/missing_data.py's `interpolation_flag` column exists
specifically to let callers exclude (see labeling/anomaly_labels.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_z_scores(
    df: pd.DataFrame,
    mcd_provider,
    *,
    variables: tuple[str, ...] = ("dust_optical_depth",),
    epsilon: float = 1.0e-6,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    ls_col: str = "solar_longitude_ls",
    ltst_col: str = "local_true_solar_time",
) -> pd.DataFrame:
    """Attach `<variable>_mu`, `<variable>_sigma`, and `<variable>_z` columns
    for each variable in `variables`, computed via a single vectorised
    `mcd_provider.query()` call per variable (never per-row)."""
    out = df.copy()
    for var in variables:
        query_result = mcd_provider.query(
            latitude=out[lat_col].to_numpy(dtype=float),
            longitude=out[lon_col].to_numpy(dtype=float),
            solar_longitude_ls=out[ls_col].to_numpy(dtype=float),
            local_true_solar_time=out[ltst_col].to_numpy(dtype=float),
            variable=var,
        )
        mu = query_result["mu"].to_numpy(dtype=float)
        sigma = query_result["sigma"].to_numpy(dtype=float)
        denom = np.maximum(sigma, epsilon)

        out[f"{var}_mu"] = mu
        out[f"{var}_sigma"] = sigma
        out[f"{var}_z"] = (out[var].to_numpy(dtype=float) - mu) / denom
    return out
