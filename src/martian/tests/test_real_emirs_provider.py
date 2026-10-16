"""Tests for RealEMIRSObservationProvider: l3atm+l3emiss merge, provenance
contract, describe(), mars_year_range filtering, and the missing/empty
local-directory error path.

Uses hand-built small FITS fixtures (via astropy.io.fits directly, matching
the real column structure from configs/real_emirs_schema.yaml) since no live
real EMIRS data is available at test time -- see the approved plan.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from astropy.io import fits

from martian.data_sources.provenance import REQUIRED_PROVENANCE_COLUMNS
from martian.data_sources.real_emirs_provider import RealEMIRSObservationProvider

_UTC_VALUES = ["2021-02-09T00:00:00.000", "2021-02-10T00:00:00.000", "2021-02-11T00:00:00.000"]
_LON_VALUES = [10.0, 20.0, 30.0]
_LAT_VALUES = [-5.0, 0.0, 5.0]


def _l3atm_hdu() -> fits.BinTableHDU:
    n = len(_UTC_VALUES)
    temp = np.arange(n * 19, dtype=float).reshape(n, 19)
    columns = [
        fits.Column(name="utc", format="23A", array=np.array(_UTC_VALUES)),
        fits.Column(name="longitude", format="D", array=np.array(_LON_VALUES)),
        fits.Column(name="latitude", format="D", array=np.array(_LAT_VALUES)),
        fits.Column(name="qls", format="E", array=np.array([6.4, 6.5, 6.6], dtype="float32")),
        fits.Column(name="qlt", format="E", array=np.array([14.0, 15.0, 16.0], dtype="float32")),
        fits.Column(name="chi2temp", format="D", array=np.array([1.1, 1.2, 1.3])),
        fits.Column(name="zh2o", format="D", array=np.array([0.1, 0.2, 0.3])),
        fits.Column(name="pres0", format="D", array=np.array([600.0, 610.0, 620.0])),
        fits.Column(name="tauice", format="D", array=np.array([0.01, 0.02, 0.03])),
        fits.Column(name="chi2a", format="D", array=np.array([0.5, 0.6, 0.7])),
        fits.Column(name="iceuncert", format="D", array=np.array([0.001] * n)),
        fits.Column(name="temp", format="19D", array=temp),
        fits.Column(name="taudust", format="D", array=np.array([0.1] * n)),
        fits.Column(name="zdust", format="D", array=np.array([20.0] * n)),
        fits.Column(name="dustuncert", format="D", array=np.array([0.01] * n)),
        fits.Column(name="wateruncert", format="D", array=np.array([0.01] * n)),
        fits.Column(name="prum", format="D", array=np.array([10.0] * n)),
    ]
    return fits.BinTableHDU.from_columns(columns)


def _l3emiss_hdu() -> fits.BinTableHDU:
    n = len(_UTC_VALUES)
    columns = [
        fits.Column(name="utc", format="23A", array=np.array(_UTC_VALUES)),
        fits.Column(name="longitude", format="D", array=np.array(_LON_VALUES)),
        fits.Column(name="latitude", format="D", array=np.array(_LAT_VALUES)),
        fits.Column(name="kinetic_temp", format="K", array=np.array([210, 215, 220])),
        fits.Column(name="kinetic_temp_uncertainty", format="D", array=np.array([1.0, 1.0, 1.0])),
        fits.Column(name="kinetic_temp_wave", format="K", array=np.array([825, 825, 825])),
    ]
    return fits.BinTableHDU.from_columns(columns)


def _write_fits(path: Path, hdu: fits.BinTableHDU) -> None:
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)


def _make_provider(tmp_path: Path, *, populate: bool = True) -> RealEMIRSObservationProvider:
    fits_dir = tmp_path / "fits"
    if populate:
        fits_dir.mkdir(parents=True, exist_ok=True)
        _write_fits(fits_dir / "l3atm_0001.fits", _l3atm_hdu())
        _write_fits(fits_dir / "l3emiss_0001.fits", _l3emiss_hdu())

    ingestion_config_path = tmp_path / "real_emirs_ingestion.yaml"
    ingestion_config_path.write_text(
        yaml.safe_dump(
            {
                "local_fits_directory": fits_dir.as_posix(),
                "products": ["l3atm", "l3emiss"],
                "fits_extensions": [".fits", ".fit"],
            }
        ),
        encoding="utf-8",
    )
    return RealEMIRSObservationProvider(ingestion_config_path=ingestion_config_path)


def test_load_merges_l3atm_and_l3emiss_and_sets_mars_year(tmp_path):
    provider = _make_provider(tmp_path)
    df = provider.load(mars_year_range=(36, 36), seed=0)

    assert len(df) == 3
    # Real fields NOT covered by pipeline_column_mapping stay as-is (extras).
    for col in ["chi2temp"]:
        assert col in df.columns
    # Real fields covered by pipeline_column_mapping are renamed onto the
    # pipeline's canonical column names -- the real names no longer appear.
    for col in ["dust_optical_depth", "surface_temperature_k", "surface_temperature_k_uncertainty", "utc_timestamp"]:
        assert col in df.columns
    for col in ["taudust", "kinetic_temp", "kinetic_temp_uncertainty", "utc"]:
        assert col not in df.columns
    assert (df["mars_year"] == 36).all()
    assert sorted(df["sol"].tolist()) == [1, 2, 3]


def test_load_passes_through_real_solar_longitude_ls_and_local_true_solar_time(tmp_path):
    """qls/qlt are real, native l3atm columns (not computed) -- see
    configs/real_emirs_schema.yaml's qls/qlt entries -- so the loaded values
    must match the fixture's real input exactly, not just be plausible."""
    provider = _make_provider(tmp_path)
    df = provider.load(mars_year_range=(36, 36), seed=0)

    assert "solar_longitude_ls" in df.columns
    assert "local_true_solar_time" in df.columns
    assert sorted(df["solar_longitude_ls"].tolist()) == pytest.approx([6.4, 6.5, 6.6], abs=1e-4)
    assert sorted(df["local_true_solar_time"].tolist()) == pytest.approx([14.0, 15.0, 16.0], abs=1e-4)


def test_load_attaches_full_provenance_and_real_data_source(tmp_path):
    provider = _make_provider(tmp_path)
    df = provider.load(mars_year_range=(36, 36), seed=0)

    for col in REQUIRED_PROVENANCE_COLUMNS:
        assert col in df.columns, f"missing required provenance column {col!r}"
    assert (df["data_source"] == "REAL").all()
    assert (df["quality_flag"] == 0).all()


def test_load_filters_out_of_range_mars_years(tmp_path):
    provider = _make_provider(tmp_path)
    df = provider.load(mars_year_range=(1, 1), seed=0)
    assert len(df) == 0


def test_describe_reports_real_data_source_and_verification_flag(tmp_path):
    provider = _make_provider(tmp_path)
    desc = provider.describe()
    assert desc["data_source"] == "REAL"
    assert desc["processing_level"] == "L3"
    assert "RESEARCHER VERIFICATION REQUIRED" in desc["verification_status"]
    assert "citation" in desc


def test_load_raises_clear_error_when_directory_missing(tmp_path):
    provider = _make_provider(tmp_path, populate=False)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        provider.load(mars_year_range=(36, 36), seed=0)


def test_load_raises_clear_error_when_directory_empty(tmp_path):
    (tmp_path / "fits").mkdir(parents=True, exist_ok=True)
    provider = _make_provider(tmp_path, populate=False)
    with pytest.raises(FileNotFoundError, match="No real l3atm FITS files found"):
        provider.load(mars_year_range=(36, 36), seed=0)
