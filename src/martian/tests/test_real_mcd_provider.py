"""Tests for RealMCDClimatologyProvider: netCDF loading + linear
interpolation (including periodic wraparound in Ls/local-time/longitude),
provenance contract, describe(), and the missing-file error path.

Uses a small, hand-built netCDF fixture (via xarray) since no live real MCD
data or compiled `fmcd` module is available at test time -- matching
test_real_emirs_provider.py's existing convention for the real EMIRS
provider.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from martian.data_sources.provenance import REQUIRED_PROVENANCE_COLUMNS
from martian.data_sources.real_mcd_provider import RealMCDClimatologyProvider

# Deliberately coarse: 2 Ls bins, 2 LTST bins, 2 lat bins, 2 lon bins --
# enough to test exact-grid-point lookup, mid-point linear interpolation on
# one axis at a time, and periodic wraparound near each axis's upper edge.
_LS = np.array([0.0, 180.0])
_LTST = np.array([0.0, 12.0])
_LAT = np.array([-45.0, 45.0])
_LON = np.array([90.0, 270.0])

# surface_temperature_k_mu varies only with ls: 200 at ls=0, 220 at ls=180.
_TEMP_MU = np.full((2, 2, 2, 2), 200.0)
_TEMP_MU[1, :, :, :] = 220.0
_TEMP_SIGMA = np.full((2, 2, 2, 2), 4.0)

# dust_optical_depth_mu varies only with lon: 0.1 at lon=90, 0.3 at lon=270.
_DUST_MU = np.full((2, 2, 2, 2), 0.1)
_DUST_MU[:, :, :, 1] = 0.3
_DUST_SIGMA = np.full((2, 2, 2, 2), 0.02)


def _write_fixture_netcdf(path: Path) -> None:
    ds = xr.Dataset(
        {
            "surface_temperature_k_mu": (("ls", "ltst", "lat", "lon"), _TEMP_MU),
            "surface_temperature_k_sigma": (("ls", "ltst", "lat", "lon"), _TEMP_SIGMA),
            "dust_optical_depth_mu": (("ls", "ltst", "lat", "lon"), _DUST_MU),
            "dust_optical_depth_sigma": (("ls", "ltst", "lat", "lon"), _DUST_SIGMA),
        },
        coords={"ls": _LS, "ltst": _LTST, "lat": _LAT, "lon": _LON},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _make_provider(tmp_path: Path, *, populate: bool = True) -> RealMCDClimatologyProvider:
    netcdf_path = tmp_path / "mcd_climatology_5deg.nc"
    if populate:
        _write_fixture_netcdf(netcdf_path)

    config_path = tmp_path / "mcd.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "mcd_version": "MCD v6.1 (test fixture)",
                "scenario": "climatology (average solar/EUV conditions)",
                "interpolation": "Linear interpolation in (Ls, local time, latitude, longitude) from the MCD's native grid",
                "citation": "Millour et al. (test fixture)",
                "licence": "Free for research use (test fixture)",
                "real": {
                    "climatology_netcdf_path": netcdf_path.as_posix(),
                    "variables": ["surface_temperature_k", "dust_optical_depth"],
                },
            }
        ),
        encoding="utf-8",
    )
    return RealMCDClimatologyProvider(config_path=config_path)


def test_query_at_exact_grid_point_returns_stored_value(tmp_path):
    provider = _make_provider(tmp_path)
    df = provider.query(
        latitude=-45.0, longitude=90.0, solar_longitude_ls=0.0,
        local_true_solar_time=0.0, variable="surface_temperature_k",
    )
    assert df["mu"].iloc[0] == pytest.approx(200.0)
    assert df["sigma"].iloc[0] == pytest.approx(4.0)


def test_query_interpolates_linearly_along_ls(tmp_path):
    provider = _make_provider(tmp_path)
    df = provider.query(
        latitude=-45.0, longitude=90.0, solar_longitude_ls=90.0,  # halfway between ls=0 (200) and ls=180 (220)
        local_true_solar_time=0.0, variable="surface_temperature_k",
    )
    assert df["mu"].iloc[0] == pytest.approx(210.0)


def test_query_interpolates_linearly_along_longitude(tmp_path):
    provider = _make_provider(tmp_path)
    df = provider.query(
        latitude=-45.0, longitude=180.0, solar_longitude_ls=0.0,  # halfway between lon=90 (0.1) and lon=270 (0.3)
        local_true_solar_time=0.0, variable="dust_optical_depth",
    )
    assert df["mu"].iloc[0] == pytest.approx(0.2)


def test_query_wraps_across_ls_seam_instead_of_extrapolating(tmp_path):
    provider = _make_provider(tmp_path)
    # ls=350 is between the real ls=180 (220) bin and the wrapped ls=360==0
    # (200) bin -- must land strictly between 200 and 220, not below 200
    # (which naive extrapolation past ls=180 would produce, since mu
    # increases with ls up to 180).
    df = provider.query(
        latitude=-45.0, longitude=90.0, solar_longitude_ls=350.0,
        local_true_solar_time=0.0, variable="surface_temperature_k",
    )
    mu = df["mu"].iloc[0]
    assert 200.0 <= mu <= 220.0


def test_query_wraps_across_longitude_seam(tmp_path):
    provider = _make_provider(tmp_path)
    df = provider.query(
        latitude=-45.0, longitude=350.0, solar_longitude_ls=0.0,
        local_true_solar_time=0.0, variable="dust_optical_depth",
    )
    mu = df["mu"].iloc[0]
    assert 0.1 <= mu <= 0.3


def test_query_supports_vectorized_inputs(tmp_path):
    provider = _make_provider(tmp_path)
    df = provider.query(
        latitude=np.array([-45.0, 45.0]),
        longitude=np.array([90.0, 270.0]),
        solar_longitude_ls=np.array([0.0, 180.0]),
        local_true_solar_time=np.array([0.0, 12.0]),
        variable="surface_temperature_k",
    )
    assert len(df) == 2
    assert df["mu"].iloc[0] == pytest.approx(200.0)
    assert df["mu"].iloc[1] == pytest.approx(220.0)


def test_query_attaches_full_provenance_and_real_data_source(tmp_path):
    provider = _make_provider(tmp_path)
    df = provider.query(
        latitude=0.0, longitude=90.0, solar_longitude_ls=0.0,
        local_true_solar_time=0.0, variable="surface_temperature_k",
    )
    for col in REQUIRED_PROVENANCE_COLUMNS:
        assert col in df.columns, f"missing required provenance column {col!r}"
    assert (df["data_source"] == "REAL").all()


def test_query_unsupported_variable_raises_clear_error(tmp_path):
    provider = _make_provider(tmp_path)
    with pytest.raises(ValueError, match="No real MCD climatology precomputed"):
        provider.query(
            latitude=0.0, longitude=0.0, solar_longitude_ls=0.0,
            local_true_solar_time=0.0, variable="not_a_real_variable",
        )


def test_query_raises_clear_error_when_netcdf_missing(tmp_path):
    provider = _make_provider(tmp_path, populate=False)
    with pytest.raises(FileNotFoundError, match="build_mcd_climatology_from_montabone.py"):
        provider.query(
            latitude=0.0, longitude=90.0, solar_longitude_ls=0.0,
            local_true_solar_time=0.0, variable="surface_temperature_k",
        )


def test_describe_reports_real_data_source_and_verification_flag(tmp_path):
    provider = _make_provider(tmp_path)
    desc = provider.describe()
    assert desc["data_source"] == "REAL"
    assert "RESEARCHER VERIFICATION REQUIRED" in desc["verification_status"]
    assert "citation" in desc
    assert "licence" in desc
    assert desc["interpolation"].startswith("Linear interpolation")


def test_describe_reports_not_yet_built_when_file_missing(tmp_path):
    provider = _make_provider(tmp_path, populate=False)
    desc = provider.describe()
    assert "NOT YET BUILT" in desc["verification_status"]


def test_describe_prefers_the_loaded_files_own_provenance_over_generic_mcd_config(tmp_path):
    # A file built by build_mcd_climatology_from_montabone.py (or any other
    # real source) carries its OWN citation/version/product_id in its netCDF
    # attrs -- describe() must report those, not silently substitute the
    # generic "MCD v6.1 / Millour et al." config values, which would
    # misattribute a non-MCD real product to the MCD proper.
    netcdf_path = tmp_path / "mcd_climatology_5deg.nc"
    _write_fixture_netcdf(netcdf_path)
    ds = xr.load_dataset(netcdf_path)
    ds.attrs.update(
        data_source="REAL",
        product_id="REAL-MONTABONE-CDOD-CLIMATOLOGY",
        version="Montabone et al. v2-5 (test fixture)",
        citation="Montabone et al. (test fixture citation)",
        source_files="griddedcdod_MY34_v2-5.nc",
    )
    ds.to_netcdf(netcdf_path)

    config_path = tmp_path / "mcd.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "mcd_version": "MCD v6.1 (generic config value -- must NOT appear in describe() below)",
                "scenario": "climatology (average solar/EUV conditions)",
                "interpolation": "Linear interpolation in (Ls, local time, latitude, longitude) from the MCD's native grid",
                "citation": "Millour et al. (generic MCD citation -- must NOT appear in describe() below)",
                "licence": "Free for research use (test fixture)",
                "real": {
                    "climatology_netcdf_path": netcdf_path.as_posix(),
                    "variables": ["surface_temperature_k", "dust_optical_depth"],
                },
            }
        ),
        encoding="utf-8",
    )
    provider = RealMCDClimatologyProvider(config_path=config_path)
    desc = provider.describe()
    assert desc["product_id"] == "REAL-MONTABONE-CDOD-CLIMATOLOGY"
    assert desc["version"] == "Montabone et al. v2-5 (test fixture)"
    assert desc["citation"] == "Montabone et al. (test fixture citation)"
    assert "Millour" not in desc["citation"]
    assert desc["source_files"] == "griddedcdod_MY34_v2-5.nc"
