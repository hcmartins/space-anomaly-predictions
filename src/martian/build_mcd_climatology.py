#!/usr/bin/env python3
"""Builds the real MCD climatology fields (mu_clim, sigma_clim) consumed by
data_sources/real_mcd_provider.py::RealMCDClimatologyProvider, replacing the
simulated parametric stand-in (data_sources/simulated_mcd.py) used by every
result reported so far in CHAPTER_4_PARAMETER_VALUES.md.

It loops over the pipeline's own 5x5-degree latitude-longitude grid
(configs/grid.yaml), over a set of solar-longitude (Ls) bins and local-true-
solar-time (LTST) bins (configs/mcd.yaml's `real` section), and for each
(cell, Ls, LTST) queries the *full, locally installed* Mars Climate Database
to obtain:

    mu_clim    : the MCD mean field                        (perturkey=1)
    sigma_clim : the day-to-day (RMS) variability field     (perturkey=2,
                 sampled N times with different seeds; std across draws)

for the label variables of Eq. 4.2 -- column dust optical depth and surface
temperature (configs/mcd.yaml real.variables).

Output is written as netCDF with dims (ls, ltst, lat, lon), variables named
"{variable}_mu" / "{variable}_sigma" -- the exact shape
RealMCDClimatologyProvider expects.

-------------------------------------------------------------------------------
PREREQUISITES (this script does NOT run against a web interface, and cannot
be run end-to-end inside this repository's own dev environment)
-------------------------------------------------------------------------------
1. Request and download the FULL MCD (data files + access software) from the
   LMD/IPSL team directly: forget@lmd.jussieu.fr, millour@lmd.jussieu.fr.
   The wrapper's source code (step 2) does NOT include the climate database
   itself -- this is a separate request/download, gated by the licence terms
   below.
2. Build the Python bindings: https://github.com/aymeric-spiga/mcd-python
   (f2py over the MCD's Fortran sources; needs gfortran + netCDF dev
   libraries + numpy -- a Linux build environment. The compile step is the
   fiddly part, not the querying itself.) This produces the importable
   module `fmcd`, exposing `fmcd.call_mcd(...)`.
3. Set the MCD data directory below (--mcd-data / configs/mcd.yaml) to the
   directory holding the downloaded MCD .nc data files.

CITATION / LICENCE: free for research use; the MCD must be cited (Millour et
al., MCD v6.1) and the LMD/IPSL team kept informed of usage; no commercial
use. See configs/mcd.yaml.

VERIFY-BEFORE-TRUSTING: the `extvar` output indices and the `perturkey`/
`scena` codes below follow the MCD v6.1 user manual as documented in
configs/mcd.yaml -- every index marked CONFIRM has NOT been independently
verified against that manual in this environment (no compiled `fmcd` module
or downloaded MCD installation is available here to cross-check). Confirm
each one against your own build's manual before trusting the numbers this
script produces.

Run (from src/martian, on a machine with steps 1-2 completed):
    python build_mcd_climatology.py --mcd-data /path/to/MCD_DATA
-------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import yaml

MARTIAN_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = MARTIAN_ROOT / "configs"

# extvar indices (1-based, per the MCD v6.1 user manual). CONFIRM each one
# against your own build's manual before relying on them -- not independently
# verified in this environment.
EXTVAR_SURFACE_TEMP = 38  # surface temperature [K]                    # CONFIRM
EXTVAR_DUST_OD = 40  # column dust visible optical depth                # CONFIRM
_EXTVAR_INDEX_BY_VARIABLE = {
    "surface_temperature_k": EXTVAR_SURFACE_TEMP,
    "dust_optical_depth": EXTVAR_DUST_OD,
}


def load_mcd_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "mcd.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def load_grid_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "grid.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def build_axes(mcd_config: dict, grid_config: dict) -> dict[str, np.ndarray]:
    real = mcd_config["real"]
    res = grid_config["resolution_deg"]
    if res != real["grid_resolution_deg"]:
        raise ValueError(
            f"configs/grid.yaml resolution_deg={res} does not match "
            f"configs/mcd.yaml real.grid_resolution_deg={real['grid_resolution_deg']} "
            "-- keep these in sync."
        )
    lats = np.arange(-90 + res / 2, 90, res)  # cell centres, e.g. -87.5 ... 87.5
    # Longitude MUST use the pipeline's own 0-360 deg-east convention
    # (configs/grid.yaml lon_convention: "0_360_east", preprocessing/grid.py)
    # -- NOT the -180/180 convention some MCD examples use -- so that
    # RealMCDClimatologyProvider's interpolation domain matches every other
    # longitude value flowing through this pipeline.
    if grid_config["lon_convention"] != "0_360_east":
        raise ValueError(
            f"configs/grid.yaml lon_convention={grid_config['lon_convention']!r} "
            "is not '0_360_east' -- build_mcd_climatology.py's longitude axis "
            "assumes 0-360 deg east and would need updating."
        )
    lons = np.arange(0 + res / 2, 360, res)  # cell centres, e.g. 2.5 ... 357.5
    ls_bins = np.arange(0, 360, real["ls_bin_deg"])
    ltst_bins = np.arange(0, 24, real["ltst_bin_hours"])
    return {"lat": lats, "lon": lons, "ls": ls_bins, "ltst": ltst_bins}


def query_mcd(fmcd_module, *, lat: float, lon: float, ls: float, ltst: float, perturkey: int, seed: int, mcd_data: str, dust_scenario_code: int):
    """One MCD point query. Returns (meanvar, extvar).

    zkey=3     : vertical coord = altitude above the local surface
    xz=0       : at the surface (irrelevant for column/surface quantities)
    datekey=1  : Mars date, so xdate carries Ls and `localtime` is used
    hrkey=1    : high-resolution mode ON
    perturkey  : 1 = mean field ; 2 = + large-scale day-to-day perturbation
    """
    extvarkeys = np.zeros(100, dtype=int)
    for idx in _EXTVAR_INDEX_BY_VARIABLE.values():
        extvarkeys[idx - 1] = 1

    (pres, dens, temp, zonwind, meridwind, meanvar, extvar, seedout, ier) = fmcd_module.call_mcd(
        zkey=3, xz=0.0, xlon=float(lon), xlat=float(lat),
        hrkey=1, datekey=1, xdate=float(ls), localtime=float(ltst),
        dset=mcd_data, scena=dust_scenario_code,
        perturkey=perturkey, seedin=int(seed), gwlength=0.0,
        extvarkeys=extvarkeys,
    )
    if ier != 0:
        raise RuntimeError(f"MCD error code {ier} at lat={lat}, lon={lon}, Ls={ls}, LTST={ltst}")
    return meanvar, extvar


def read_target_vars(extvar, variables: list[str]) -> dict[str, float]:
    return {v: extvar[_EXTVAR_INDEX_BY_VARIABLE[v] - 1] for v in variables}


def mu_sigma_at(fmcd_module, *, lat: float, lon: float, ls: float, ltst: float, variables: list[str], n_sigma: int, mcd_data: str, dust_scenario_code: int) -> tuple[dict, dict]:
    """mu_clim from the mean field; sigma_clim from N perturbed draws."""
    _, extvar_mean = query_mcd(fmcd_module, lat=lat, lon=lon, ls=ls, ltst=ltst, perturkey=1, seed=1, mcd_data=mcd_data, dust_scenario_code=dust_scenario_code)
    mu = read_target_vars(extvar_mean, variables)

    draws: dict[str, list[float]] = {v: [] for v in variables}
    for i in range(n_sigma):
        _, extvar_pert = query_mcd(fmcd_module, lat=lat, lon=lon, ls=ls, ltst=ltst, perturkey=2, seed=i + 1, mcd_data=mcd_data, dust_scenario_code=dust_scenario_code)
        vals = read_target_vars(extvar_pert, variables)
        for v, val in vals.items():
            draws[v].append(val)
    sigma = {v: float(np.std(draws[v], ddof=1)) for v in variables}
    # NOTE: the MCD's large-scale perturbations capture day-to-day
    # variability for temperature/density/winds robustly; for DUST optical
    # depth the built-in perturbation spread is documented as weaker, so
    # consider deriving sigma for dust from the inter-scenario spread
    # instead, and state explicitly which one was used when citing this
    # output (see configs/mcd.yaml, real.n_sigma_draws).
    return mu, sigma


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mcd-data", required=True, help="Directory holding the downloaded MCD .nc data files (see this script's header, prerequisite 1).")
    parser.add_argument("--config", default=None, help="Path to configs/mcd.yaml (default: src/martian/configs/mcd.yaml).")
    parser.add_argument("--grid-config", default=None, help="Path to configs/grid.yaml (default: src/martian/configs/grid.yaml).")
    parser.add_argument("--output", default=None, help="Output netCDF path (default: configs/mcd.yaml's real.climatology_netcdf_path).")
    args = parser.parse_args(argv)

    try:
        import fmcd  # f2py-compiled MCD interface (aymeric-spiga/mcd-python)
    except ImportError as exc:
        raise SystemExit(
            "The `fmcd` module was not found. Build mcd-python "
            "(https://github.com/aymeric-spiga/mcd-python) against a real MCD "
            "installation first -- see this script's module docstring for the "
            "full prerequisites. This step requires gfortran + netCDF dev "
            "libraries (a Linux build environment) and cannot be completed "
            "inside this repository's own dev environment."
        ) from exc
    import xarray as xr

    mcd_config = load_mcd_config(Path(args.config) if args.config else None)
    grid_config = load_grid_config(Path(args.grid_config) if args.grid_config else None)
    real = mcd_config["real"]
    variables: list[str] = real["variables"]
    n_sigma: int = real["n_sigma_draws"]
    dust_scenario_code: int = real["dust_scenario_code"]
    output_path = Path(args.output) if args.output else Path(real["climatology_netcdf_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    axes = build_axes(mcd_config, grid_config)
    ls_bins, ltst_bins, lats, lons = axes["ls"], axes["ltst"], axes["lat"], axes["lon"]

    shape = (len(ls_bins), len(ltst_bins), len(lats), len(lons))
    mu_out = {v: np.full(shape, np.nan) for v in variables}
    sig_out = {v: np.full(shape, np.nan) for v in variables}

    index_grid = list(itertools.product(range(len(ls_bins)), range(len(ltst_bins)), range(len(lats)), range(len(lons))))
    total = len(index_grid)
    for n, (a, b, c, d) in enumerate(index_grid):
        mu, sigma = mu_sigma_at(
            fmcd, lat=lats[c], lon=lons[d], ls=ls_bins[a], ltst=ltst_bins[b],
            variables=variables, n_sigma=n_sigma, mcd_data=args.mcd_data,
            dust_scenario_code=dust_scenario_code,
        )
        for v in variables:
            mu_out[v][a, b, c, d] = mu[v]
            sig_out[v][a, b, c, d] = sigma[v]
        if n % 5000 == 0:
            print(f"{n}/{total} queries done", flush=True)

    data = {}
    for v in variables:
        data[f"{v}_mu"] = (("ls", "ltst", "lat", "lon"), mu_out[v])
        data[f"{v}_sigma"] = (("ls", "ltst", "lat", "lon"), sig_out[v])
    ds = xr.Dataset(data, coords={"ls": ls_bins, "ltst": ltst_bins, "lat": lats, "lon": lons})
    ds.attrs.update(
        product_id="REAL-MCD-CLIMATOLOGY",
        data_source="REAL",
        version=mcd_config["mcd_version"],
        source=f"Mars Climate Database ({mcd_config['mcd_version']})",
        dust_scenario=mcd_config["scenario"],
        interpolation=mcd_config["interpolation"],
        citation=mcd_config["citation"],
        licence=mcd_config["licence"],
        note="mu = MCD mean field (perturkey=1); sigma = std over N large-scale day-to-day perturbation draws (perturkey=2)",
    )
    ds.to_netcdf(output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
