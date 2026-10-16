"""Tests for build_mcd_climatology_from_montabone.py: real-Montabone-format
netCDF loading, longitude convention conversion (-180/180 -> 0-360,
re-sorted), Ls-binned pooling across Mars Years, NaN-aware mean/std, and the
degenerate LTST axis.

Uses small, hand-built netCDF fixtures matching the real Montabone gridded-
CDOD file schema (Time/latitude/longitude dims, Ls/cdod610 variables) since
the real downloaded files in data_cache/ are large and specific to this
machine -- matching test_real_emirs_provider.py's existing fixture
convention.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from martian.build_mcd_climatology_from_montabone import build_climatology


def _write_fixture_year(path: Path, *, ls_values: list[float], cdod_values: list[list[list[float]]]) -> None:
    """lat = [-45, 45] (descending, real-file convention), lon = [-90, 90]
    (real -180/180-style convention, deliberately unsorted after mod-360)."""
    n = len(ls_values)
    ds = xr.Dataset(
        {
            "SOY": ("Time", np.arange(n, dtype=np.int16)),
            "Ls": ("Time", np.array(ls_values, dtype=np.float32)),
            "cdod610": (("Time", "latitude", "longitude"), np.array(cdod_values, dtype=np.float32)),
        },
        coords={
            "Time": np.arange(n, dtype=np.float32) + 0.5,
            "latitude": np.array([45.0, -45.0], dtype=np.float32),  # descending, like the real files
            "longitude": np.array([-90.0, 90.0], dtype=np.float32),
        },
    )
    ds.to_netcdf(path)


def test_pools_across_years_and_bins_by_real_ls(tmp_path):
    # MY_A: one sol at Ls=10 (bin 0, width 30deg -> [0,30)), value 0.10 everywhere.
    _write_fixture_year(
        tmp_path / "griddedcdod_MY34_v2-5.nc",
        ls_values=[10.0],
        cdod_values=[[[0.10, 0.10], [0.10, 0.10]]],
    )
    # MY_B: one sol at Ls=20 (same bin 0), value 0.30 everywhere.
    _write_fixture_year(
        tmp_path / "griddedcdod_MY35_v2-5.nc",
        ls_values=[20.0],
        cdod_values=[[[0.30, 0.30], [0.30, 0.30]]],
    )
    ds_out = build_climatology(tmp_path, [34, 35], ls_bin_deg=30.0)

    mu = ds_out["dust_optical_depth_mu"]
    # Bin 0 (ls center = 15.0) should pool BOTH sols -> mean 0.20.
    assert float(mu.sel(ls=15.0, ltst=0.0, lat=-45.0, lon=90.0)) == pytest.approx(0.20)
    sigma = ds_out["dust_optical_depth_sigma"]
    assert float(sigma.sel(ls=15.0, ltst=0.0, lat=-45.0, lon=90.0)) == pytest.approx(np.std([0.10, 0.30], ddof=1))


def test_longitude_converted_to_0_360_and_sorted(tmp_path):
    _write_fixture_year(
        tmp_path / "griddedcdod_MY34_v2-5.nc",
        ls_values=[10.0],
        cdod_values=[[[0.10, 0.20], [0.10, 0.20]]],  # lon=-90 -> 0.10, lon=+90 -> 0.20
    )
    ds_out = build_climatology(tmp_path, [34], ls_bin_deg=30.0)

    lon = ds_out["lon"].values
    assert list(lon) == sorted(lon)  # ascending
    assert set(np.round(lon).astype(int)) == {90, 270}  # -90 -> 270, +90 -> 90 under mod 360


def test_degenerate_ltst_axis_has_two_identical_bins(tmp_path):
    _write_fixture_year(
        tmp_path / "griddedcdod_MY34_v2-5.nc",
        ls_values=[10.0],
        cdod_values=[[[0.10, 0.10], [0.10, 0.10]]],
    )
    ds_out = build_climatology(tmp_path, [34], ls_bin_deg=30.0)
    assert list(ds_out["ltst"].values) == [0.0, 12.0]
    mu = ds_out["dust_optical_depth_mu"]
    assert float(mu.isel(ls=0, ltst=0, lat=0, lon=0)) == float(mu.isel(ls=0, ltst=1, lat=0, lon=0))


def test_nan_only_bin_stays_nan_never_fabricated(tmp_path):
    # Only one sol, at Ls=10 -> only bin 0 is populated; every other Ls bin
    # must stay NaN, never interpolated/fabricated into a value.
    _write_fixture_year(
        tmp_path / "griddedcdod_MY34_v2-5.nc",
        ls_values=[10.0],
        cdod_values=[[[0.10, 0.10], [0.10, 0.10]]],
    )
    ds_out = build_climatology(tmp_path, [34], ls_bin_deg=30.0)
    mu = ds_out["dust_optical_depth_mu"]
    # bin 1 (ls center = 45.0, covering Ls [30,60)) had zero pooled sols.
    assert bool(np.isnan(mu.sel(ls=45.0, ltst=0.0, lat=-45.0, lon=90.0)))


def test_missing_year_file_raises_clear_error(tmp_path):
    _write_fixture_year(
        tmp_path / "griddedcdod_MY34_v2-5.nc",
        ls_values=[10.0],
        cdod_values=[[[0.10, 0.10], [0.10, 0.10]]],
    )
    with pytest.raises(FileNotFoundError, match="griddedcdod_MY35_v2-5.nc"):
        build_climatology(tmp_path, [34, 35], ls_bin_deg=30.0)


def test_mismatched_grid_across_years_raises(tmp_path):
    _write_fixture_year(
        tmp_path / "griddedcdod_MY34_v2-5.nc",
        ls_values=[10.0],
        cdod_values=[[[0.10, 0.10], [0.10, 0.10]]],
    )
    # MY35 fixture with a DIFFERENT latitude grid.
    n = 1
    ds = xr.Dataset(
        {
            "SOY": ("Time", np.arange(n, dtype=np.int16)),
            "Ls": ("Time", np.array([10.0], dtype=np.float32)),
            "cdod610": (("Time", "latitude", "longitude"), np.array([[[0.1, 0.1], [0.1, 0.1]]], dtype=np.float32)),
        },
        coords={
            "Time": np.array([0.5], dtype=np.float32),
            "latitude": np.array([30.0, -30.0], dtype=np.float32),  # different from MY34's [45, -45]
            "longitude": np.array([-90.0, 90.0], dtype=np.float32),
        },
    )
    ds.to_netcdf(tmp_path / "griddedcdod_MY35_v2-5.nc")

    with pytest.raises(ValueError, match="different lat/lon grid"):
        build_climatology(tmp_path, [34, 35], ls_bin_deg=30.0)


def test_output_attrs_carry_real_provenance(tmp_path):
    _write_fixture_year(
        tmp_path / "griddedcdod_MY34_v2-5.nc",
        ls_values=[10.0],
        cdod_values=[[[0.10, 0.10], [0.10, 0.10]]],
    )
    ds_out = build_climatology(tmp_path, [34], ls_bin_deg=30.0)
    assert ds_out.attrs["data_source"] == "REAL"
    assert ds_out.attrs["product_id"] == "REAL-MONTABONE-CDOD-CLIMATOLOGY"
    assert "Montabone" in ds_out.attrs["citation"]
    assert "CC BY-SA" in ds_out.attrs["licence"]
    assert "surface_temperature_k" in ds_out.attrs["not_covered"]
