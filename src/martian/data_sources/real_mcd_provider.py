"""Real MCD climatology provider -- replaces data_sources/real_stubs.py's
RealMCDClimatologyProvider stub (removed there; see that module).

Reads a PRECOMPUTED climatology netCDF file (mu_clim/sigma_clim on a
(Ls, LTST, lat, lon) grid) built OFFLINE by build_mcd_climatology.py against
a real, locally-installed Mars Climate Database (via aymeric-spiga/mcd-python's
f2py wrapper around the MCD's Fortran call_mcd routine -- see that script's
own header for the full build prerequisites: a compiled `fmcd` module + a
real, separately-downloaded MCD installation, neither of which this
repository can provide on its own).

This provider does NOT call the MCD live at query time -- it interpolates the
precomputed grid, LINEARLY in (Ls, local time, latitude, longitude), matching
the MCD's own documented native query behaviour (see configs/mcd.yaml).

If the netCDF file configured at configs/mcd.yaml's real.climatology_netcdf_path
does not exist, this provider raises a clear, actionable FileNotFoundError --
it never falls back to a fabricated or empty value.

WIRED INTO mars.run_all (Step 7, data_source: dust_optical_depth), but the
full pipeline still cannot run end-to-end: RealEMIRSObservationProvider
(the other real provider it is paired with) has no real downloaded EMIRS
FITS files yet -- see configs/real_emirs_ingestion.yaml. No simulated
EMIRS/MCD provider exists anywhere in this repository as a fallback.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.interpolate import RegularGridInterpolator

from .base import MCDClimatologyProvider
from .provenance import ProviderIdentity, attach_batch_provenance
from .schema import CONFIG_DIR


class RealMCDClimatologyProvider(MCDClimatologyProvider):
    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or CONFIG_DIR / "mcd.yaml"
        self._config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self._real_config = self._config["real"]
        self._netcdf_path = Path(self._real_config["climatology_netcdf_path"])
        self._variables = self._real_config["variables"]
        self._dataset: xr.Dataset | None = None
        self._interpolators: dict[str, RegularGridInterpolator] = {}

    def _load_dataset(self) -> xr.Dataset:
        if self._dataset is not None:
            return self._dataset
        if not self._netcdf_path.exists():
            raise FileNotFoundError(
                f"Real MCD climatology file not found at {self._netcdf_path}. This "
                "file is NOT generated automatically by this pipeline -- run "
                "build_mcd_climatology.py (src/martian root) against a compiled "
                "`fmcd` module (https://github.com/aymeric-spiga/mcd-python) and a "
                "real, separately-downloaded MCD installation first. See "
                "configs/mcd.yaml's `real` section and that script's own header "
                "for the full prerequisites (gfortran + netCDF dev libraries + a "
                "Linux build environment, plus the MCD data files themselves)."
            )
        self._dataset = xr.load_dataset(self._netcdf_path)
        return self._dataset

    @staticmethod
    def _wrap_pad(coord: np.ndarray, values: np.ndarray, *, axis: int, period: float) -> tuple[np.ndarray, np.ndarray]:
        """Appends a boundary point equal to the first point, `period` later, so
        linear interpolation wraps correctly across a periodic seam (e.g. a
        query at Ls=359 interpolates toward the real Ls=0/360 value, rather
        than extrapolating past the last real bin at ~Ls=330).

        ENGINEERING DECISION: the MCD's own grid is periodic in Ls, local
        time, and longitude; a plain non-periodic RegularGridInterpolator
        would otherwise mishandle values near each axis's upper edge. This
        wrap-and-pad is the standard, defensible technique for that -- not
        independently verified against the real MCD wrapper's own behaviour,
        since no live `fmcd` module is available in this environment to
        cross-check against.
        """
        wrapped_coord = np.concatenate([coord, [coord[0] + period]])
        first_slice = np.take(values, [0], axis=axis)
        wrapped_values = np.concatenate([values, first_slice], axis=axis)
        return wrapped_coord, wrapped_values

    def _interpolator_for(self, variable: str, field: str) -> RegularGridInterpolator:
        """field: 'mu' or 'sigma'."""
        key = f"{variable}_{field}"
        if key in self._interpolators:
            return self._interpolators[key]
        ds = self._load_dataset()
        column = f"{variable}_{field}"
        if column not in ds:
            raise ValueError(
                f"Real MCD climatology file {self._netcdf_path} has no {column!r} "
                f"variable -- available: {list(ds.data_vars)}. Was it built with "
                f"configs/mcd.yaml's real.variables including {variable!r}?"
            )
        da = ds[column]
        ls = np.asarray(da["ls"].values, dtype=float)
        ltst = np.asarray(da["ltst"].values, dtype=float)
        lat = np.asarray(da["lat"].values, dtype=float)
        lon = np.asarray(da["lon"].values, dtype=float)
        values = np.asarray(da.values, dtype=float)  # dims: (ls, ltst, lat, lon)

        ls_w, values = self._wrap_pad(ls, values, axis=0, period=360.0)
        ltst_w, values = self._wrap_pad(ltst, values, axis=1, period=24.0)
        lon_w, values = self._wrap_pad(lon, values, axis=3, period=360.0)

        interpolator = RegularGridInterpolator(
            (ls_w, ltst_w, lat, lon_w), values,
            method="linear", bounds_error=False, fill_value=None,
        )
        self._interpolators[key] = interpolator
        return interpolator

    def describe(self) -> dict:
        # PREFER the loaded file's OWN embedded provenance attrs when the
        # file exists -- different build scripts (build_mcd_climatology.py
        # via call_mcd vs build_mcd_climatology_from_montabone.py via real
        # downloaded Montabone CDOD files) produce genuinely different real
        # products with different citations/versions/verification status.
        # Reading configs/mcd.yaml's generic MCD-proper fields unconditionally
        # would misattribute a Montabone-built file to "MCD v6.1 / Millour et
        # al." when it was never built from a live or precomputed call_mcd
        # query at all. Fall back to the generic config only when no file
        # exists yet (still useful to document intent before it's built).
        if self._netcdf_path.exists():
            ds = self._load_dataset()
            attrs = ds.attrs
            return {
                "data_source": attrs.get("data_source", "REAL"),
                "product_id": attrs.get("product_id", "REAL-MCD-CLIMATOLOGY"),
                "processing_level": "N/A (precomputed climatology grid, not a live retrieval)",
                "version": attrs.get("version", self._config["mcd_version"]),
                "scenario": attrs.get("dust_scenario", self._config["scenario"]),
                "interpolation": attrs.get("interpolation", self._config["interpolation"]),
                "verification_status": (
                    "RESEARCHER VERIFICATION REQUIRED (this file's own build method and "
                    "statistics are documented in its netCDF attrs, but have not been "
                    "independently cross-checked against the cited papers/manual in this "
                    "environment)"
                ),
                "citation": attrs.get("citation", self._config["citation"]),
                "licence": attrs.get("licence", self._config["licence"]),
                "build_method": attrs.get("method", attrs.get("note", "unknown")),
                "source_files": attrs.get("source_files"),
            }
        return {
            "data_source": "REAL",
            "product_id": "REAL-MCD-CLIMATOLOGY",
            "processing_level": "N/A (precomputed climatology grid, not a live retrieval)",
            "version": self._config["mcd_version"],
            "scenario": self._config["scenario"],
            "interpolation": self._config["interpolation"],
            "verification_status": (
                "NOT YET BUILT -- no precomputed climatology file exists at "
                f"{self._netcdf_path}. The fields above describe intent (the "
                "MCD-proper target), not a built, checkable product."
            ),
            "citation": self._config["citation"],
            "licence": self._config["licence"],
        }

    def _identity(self) -> ProviderIdentity:
        d = self.describe()
        return ProviderIdentity(
            data_source=d["data_source"],
            product_id=d["product_id"],
            processing_level=d["processing_level"],
            version=d["version"],
            verification_status=d["verification_status"],
        )

    def query(
        self,
        *,
        latitude,
        longitude,
        solar_longitude_ls,
        local_true_solar_time,
        variable: str,
        pressure_level_pa: float | None = None,
    ) -> pd.DataFrame:
        del pressure_level_pa  # only surface/column variables precomputed so far -- see configs/mcd.yaml real.variables
        if variable not in self._variables:
            raise ValueError(
                f"No real MCD climatology precomputed for variable={variable!r}. "
                f"Configured: {self._variables} (configs/mcd.yaml real.variables)."
            )
        lat = np.atleast_1d(np.asarray(latitude, dtype=float))
        lon = np.atleast_1d(np.asarray(longitude, dtype=float)) % 360.0
        ls = np.atleast_1d(np.asarray(solar_longitude_ls, dtype=float)) % 360.0
        ltst = np.atleast_1d(np.asarray(local_true_solar_time, dtype=float)) % 24.0

        points = np.stack([ls, ltst, lat, lon], axis=-1)
        mu = self._interpolator_for(variable, "mu")(points)
        sigma = self._interpolator_for(variable, "sigma")(points)

        df = pd.DataFrame(
            {
                "latitude": lat,
                "longitude": lon,
                "solar_longitude_ls": ls,
                "local_true_solar_time": ltst,
                "variable": variable,
                "mu": mu,
                "sigma": sigma,
            }
        )
        df.attrs["units"] = {"latitude": "degrees", "longitude": "degrees_east_0_360"}

        identity = self._identity()
        n = len(df)
        df = attach_batch_provenance(
            df,
            identity,
            original_filenames=pd.Series([self._netcdf_path.name] * n),
            quality_flags=pd.Series([0] * n),  # precomputed-grid queries are never "flagged"
            retrieval_timestamps=pd.Series([pd.Timestamp.now(tz="UTC")] * n),
            checksum_source_columns=[
                "latitude", "longitude", "solar_longitude_ls",
                "local_true_solar_time", "variable", "mu", "sigma",
            ],
        )
        return self._enforce_provenance_contract(df, caller="RealMCDClimatologyProvider.query")
