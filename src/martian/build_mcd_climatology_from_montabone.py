#!/usr/bin/env python3
"""Builds the real dust climatology fields (mu_clim, sigma_clim for
dust_optical_depth) consumed by
data_sources/real_mcd_provider.py::RealMCDClimatologyProvider, from the
REAL, downloaded Montabone et al. gridded column-dust-optical-depth (CDOD)
climatology files (data_cache/griddedcdod_MY{34,35,36}_v2-5.nc) -- no
compiled `fmcd` module and no separate MCD installation download are needed
for this path (unlike build_mcd_climatology.py, which calls the MCD's own
Fortran `call_mcd` routine and remains blocked on external prerequisites).

WHAT THIS DATA IS: real, gridded, per-sol column dust optical depth,
reconstructed from real Mars Reconnaissance Orbiter/Mars Climate Sounder
observations via the Iterative Weighted Binning (IWB) method (Montabone et
al., 2015; 2019). This is the SAME dust-climatology product the MCD's own
"climatology"/dust-scenario input is built from -- not the MCD's GCM output
itself, but the real observational reconstruction underpinning it. See each
file's own embedded citation/licence metadata (`ds.attrs`), reproduced in
the output file's attributes below.

WHAT THIS DOES NOT COVER: only column dust optical depth (tau_d). No real
source for surface_temperature_k exists yet -- RealMCDClimatologyProvider
will (correctly) raise a clear error if queried for that variable against
this file; see configs/mcd.yaml's real.variables, which lists only
dust_optical_depth until a real temperature climatology source is found.

METHOD: for each of the 3 downloaded Mars Years, cdod610 (CDOD normalized to
the 610 Pa reference pressure -- the standard, citable convention for this
quantity, matching Chapter 4's tau_d) is pooled by real per-sol Ls (not the
`360*sol/668` linear approximation used by the simulated generator) into
Ls-degree-wide bins (configs/mcd.yaml real.ls_bin_deg), per native grid cell
(5 deg lat x 6 deg lon -- Montabone's own grid, NOT regridded to this
pipeline's 5x5 grid; RealMCDClimatologyProvider interpolates at query time,
so exact grid alignment is not required). mu_clim is the pooled mean across
all 3 years' sols falling in that (Ls-bin, lat, lon) cell; sigma_clim is the
pooled standard deviation of that SAME pool -- capturing both real
within-season (day-to-day) and real year-to-year (interannual) dust
variability at once. ENGINEERING DECISION: this is a defensible, real,
NaN-aware statistic (missing satellite coverage in a given cell/sol is
excluded from the pool, never imputed) -- it is NOT the same statistical
quantity as the MCD's own internal "day-to-day RMS variability field" (that
would require the GCM path in build_mcd_climatology.py), and this file's
netCDF attributes say so explicitly.

The diurnally-averaged nature of this dataset (each file's own title states
"diurnal averages... provided as a separate dataset") means it carries NO
local-time resolution. The output's `ltst` axis therefore has 2 bins with
IDENTICAL values (0.0h and 12.0h) -- a degenerate axis, not a real diurnal
climatology. Do not read local-time structure into dust_optical_depth_mu/
sigma from this file.

CITATION / LICENCE (reproduced from the source files' own attrs, and written
into the output netCDF's attrs): Montabone, L., Spiga, A., Kass, D. M.,
Kleinbohl, A., Forget, F., Millour, E., "Martian Year 34 Column Dust
Climatology from Mars Climate Sounder Observations: Reconstructed Maps and
Model Simulations", J. Geophys. Res. - Planets (2019). Data are CC BY-SA 3.0
licensed (free to copy/redistribute/modify with attribution).

Run (from src/martian, once the files above are present in data_cache/):
    python build_mcd_climatology_from_montabone.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

MARTIAN_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = MARTIAN_ROOT / "configs"

_SOURCE_VARIABLE = "cdod610"  # CDOD normalized to 610 Pa -- see module docstring
_OUTPUT_VARIABLE = "dust_optical_depth"
_LTST_BINS = np.array([0.0, 12.0])  # degenerate: source data has no local-time resolution


def load_mcd_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "mcd.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _load_one_year(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (ls_per_sol, lat_ascending, lon_0_360_ascending, cdod610) with
    cdod610 reindexed to match the ascending lat/lon axes returned."""
    ds = xr.load_dataset(path)
    if _SOURCE_VARIABLE not in ds:
        raise ValueError(f"{path} has no {_SOURCE_VARIABLE!r} variable -- unexpected Montabone file format.")

    ls = np.asarray(ds["Ls"].values, dtype=float)
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon_raw = np.asarray(ds["longitude"].values, dtype=float)
    cdod = np.asarray(ds[_SOURCE_VARIABLE].values, dtype=float)  # (Time, latitude, longitude)

    lat_order = np.argsort(lat)
    lon_0_360 = lon_raw % 360.0
    lon_order = np.argsort(lon_0_360)

    lat_sorted = lat[lat_order]
    lon_sorted = lon_0_360[lon_order]
    cdod_sorted = cdod[:, lat_order, :][:, :, lon_order]

    ds.close()
    return ls, lat_sorted, lon_sorted, cdod_sorted


def build_climatology(data_dir: Path, mars_years: list[int], ls_bin_deg: float) -> xr.Dataset:
    all_ls: list[np.ndarray] = []
    all_cdod: list[np.ndarray] = []
    lat_ref: np.ndarray | None = None
    lon_ref: np.ndarray | None = None
    source_files: list[str] = []

    for my in mars_years:
        path = data_dir / f"griddedcdod_MY{my}_v2-5.nc"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected real Montabone gridded-CDOD file not found: {path}. "
                f"Download griddedcdod_MY{my}_v2-5.nc into {data_dir} first."
            )
        ls, lat, lon, cdod = _load_one_year(path)
        if lat_ref is None:
            lat_ref, lon_ref = lat, lon
        elif not (np.allclose(lat, lat_ref) and np.allclose(lon, lon_ref)):
            raise ValueError(f"{path} has a different lat/lon grid from the other Mars Years -- cannot pool directly.")
        all_ls.append(ls)
        all_cdod.append(cdod)
        source_files.append(path.name)

    assert lat_ref is not None and lon_ref is not None
    n_ls_bins = int(np.ceil(360.0 / ls_bin_deg))
    ls_bin_centers = (np.arange(n_ls_bins) + 0.5) * ls_bin_deg

    n_lat, n_lon = len(lat_ref), len(lon_ref)
    mu = np.full((n_ls_bins, len(_LTST_BINS), n_lat, n_lon), np.nan)
    sigma = np.full((n_ls_bins, len(_LTST_BINS), n_lat, n_lon), np.nan)
    n_pool = np.zeros((n_ls_bins,), dtype=int)

    for bin_idx in range(n_ls_bins):
        lo, hi = bin_idx * ls_bin_deg, (bin_idx + 1) * ls_bin_deg
        pooled_slices = []
        for ls, cdod in zip(all_ls, all_cdod):
            sol_mask = (ls >= lo) & (ls < hi)
            if sol_mask.any():
                pooled_slices.append(cdod[sol_mask])  # (n_sols_in_bin, lat, lon)
        if not pooled_slices:
            continue
        pooled = np.concatenate(pooled_slices, axis=0)  # (n_pooled_sols, lat, lon)
        n_pool[bin_idx] = pooled.shape[0]
        with np.errstate(invalid="ignore"):
            bin_mu = np.nanmean(pooled, axis=0)
            bin_sigma = np.nanstd(pooled, axis=0, ddof=1) if pooled.shape[0] > 1 else np.full((n_lat, n_lon), np.nan)
        for ltst_idx in range(len(_LTST_BINS)):  # degenerate LTST axis -- see module docstring
            mu[bin_idx, ltst_idx] = bin_mu
            sigma[bin_idx, ltst_idx] = bin_sigma

    ds_out = xr.Dataset(
        {
            f"{_OUTPUT_VARIABLE}_mu": (("ls", "ltst", "lat", "lon"), mu),
            f"{_OUTPUT_VARIABLE}_sigma": (("ls", "ltst", "lat", "lon"), sigma),
        },
        coords={"ls": ls_bin_centers, "ltst": _LTST_BINS, "lat": lat_ref, "lon": lon_ref},
    )
    ds_out.attrs.update(
        product_id="REAL-MONTABONE-CDOD-CLIMATOLOGY",
        data_source="REAL",
        version="Montabone et al. v2-5 (gridded CDOD, IWB method)",
        source="Montabone et al. gridded column dust optical depth (CDOD) climatology, real MRO/MCS observations",
        source_files=", ".join(source_files),
        source_mars_years=", ".join(str(my) for my in mars_years),
        source_variable=_SOURCE_VARIABLE,
        method=(
            "mu = NaN-aware pooled mean, sigma = NaN-aware pooled std (ddof=1), across all sols of all "
            f"listed Mars Years falling in each {ls_bin_deg}-deg Ls bin, per native (lat, lon) grid cell. "
            "LTST axis is degenerate (source data is diurnally averaged, no local-time resolution)."
        ),
        n_pooled_sols_per_ls_bin=", ".join(str(int(n)) for n in n_pool),
        citation=(
            "Montabone, L., Spiga, A., Kass, D. M., Kleinboehl, A., Forget, F., Millour, E., "
            "Martian Year 34 Column Dust Climatology from Mars Climate Sounder Observations: "
            "Reconstructed Maps and Model Simulations, J. Geophys. Res. - Planets (2019), "
            "doi: 10.1029/2019JE006111; Montabone et al., Eight-year Climatology of Dust Optical "
            "Depth on Mars, Icarus (2015), 251, 65-95, doi: 10.1016/j.icarus.2014.12.034."
        ),
        licence="CC BY-SA 3.0 Unported (free to copy/redistribute/modify with attribution) -- see source files' own attrs.",
        not_covered="No real climatology for surface_temperature_k -- see configs/mcd.yaml real.variables.",
    )
    return ds_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="Directory holding griddedcdod_MY*_v2-5.nc (default: configs/mcd.yaml real.montabone.data_directory).")
    parser.add_argument("--mars-years", nargs="+", type=int, default=None, help="Mars Years to pool (default: configs/mcd.yaml real.montabone.mars_years).")
    parser.add_argument("--output", default=None, help="Output netCDF path (default: configs/mcd.yaml real.climatology_netcdf_path).")
    parser.add_argument("--config", default=None, help="Path to configs/mcd.yaml (default: src/martian/configs/mcd.yaml).")
    args = parser.parse_args(argv)

    mcd_config = load_mcd_config(Path(args.config) if args.config else None)
    real = mcd_config["real"]
    montabone = real["montabone"]

    data_dir = Path(args.data_dir) if args.data_dir else MARTIAN_ROOT / montabone["data_directory"]
    mars_years = args.mars_years if args.mars_years else montabone["mars_years"]
    ls_bin_deg = real["ls_bin_deg"]
    output_path = Path(args.output) if args.output else MARTIAN_ROOT / real["climatology_netcdf_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds_out = build_climatology(data_dir, mars_years, ls_bin_deg)
    ds_out.to_netcdf(output_path)
    print(f"Wrote {output_path}")
    print(f"  ls bins: {ds_out.sizes['ls']} (width {ls_bin_deg} deg), lat: {ds_out.sizes['lat']}, lon: {ds_out.sizes['lon']}")
    print(f"  n_pooled_sols_per_ls_bin: {ds_out.attrs['n_pooled_sols_per_ls_bin']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
