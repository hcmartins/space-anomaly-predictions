"""Tests for real_emirs_fits.py: real FITS parsing, temp-profile collapse,
product identification, and directory-loading error handling.

Uses hand-built small FITS fixtures (via astropy.io.fits directly, matching
the real column structure from configs/real_emirs_schema.yaml) since no live
real EMIRS data is available at test time -- see the approved plan.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from martian.data_sources.real_emirs_fits import (
    apply_pipeline_column_mapping,
    collapse_temp_profile,
    identify_product,
    load_real_emirs_directory,
    load_real_emirs_schema,
    parse_fits_file,
)


def _l3atm_hdu(n_rows: int = 3) -> fits.BinTableHDU:
    utc = np.array([f"2021-02-{9 + i:02d}T00:00:00.000" for i in range(n_rows)])
    temp = np.arange(n_rows * 19, dtype=float).reshape(n_rows, 19)
    columns = [
        fits.Column(name="utc", format="23A", array=utc),
        fits.Column(name="longitude", format="D", array=np.array([10.0, 20.0, 30.0][:n_rows])),
        fits.Column(name="latitude", format="D", array=np.array([-5.0, 0.0, 5.0][:n_rows])),
        fits.Column(name="chi2temp", format="D", array=np.array([1.1, 1.2, 1.3][:n_rows])),
        fits.Column(name="zh2o", format="D", array=np.array([0.1, 0.2, 0.3][:n_rows])),
        fits.Column(name="pres0", format="D", array=np.array([600.0, 610.0, 620.0][:n_rows])),
        fits.Column(name="tauice", format="D", array=np.array([0.01, 0.02, 0.03][:n_rows])),
        fits.Column(name="chi2a", format="D", array=np.array([0.5, 0.6, 0.7][:n_rows])),
        fits.Column(name="iceuncert", format="D", array=np.array([0.001] * n_rows)),
        fits.Column(name="temp", format="19D", array=temp),
        fits.Column(name="taudust", format="D", array=np.array([0.1] * n_rows)),
        fits.Column(name="zdust", format="D", array=np.array([20.0] * n_rows)),
        fits.Column(name="dustuncert", format="D", array=np.array([0.01] * n_rows)),
        fits.Column(name="wateruncert", format="D", array=np.array([0.01] * n_rows)),
        fits.Column(name="prum", format="D", array=np.array([10.0] * n_rows)),
    ]
    return fits.BinTableHDU.from_columns(columns)


def _l3emiss_hdu(n_rows: int = 3) -> fits.BinTableHDU:
    utc = np.array([f"2021-02-{9 + i:02d}T00:00:00.000" for i in range(n_rows)])
    columns = [
        fits.Column(name="utc", format="23A", array=utc),
        fits.Column(name="longitude", format="D", array=np.array([10.0, 20.0, 30.0][:n_rows])),
        fits.Column(name="latitude", format="D", array=np.array([-5.0, 0.0, 5.0][:n_rows])),
        fits.Column(name="kinetic_temp", format="K", array=np.array([210, 215, 220][:n_rows])),
        fits.Column(name="kinetic_temp_uncertainty", format="D", array=np.array([1.0, 1.0, 1.0][:n_rows])),
        fits.Column(name="kinetic_temp_wave", format="K", array=np.array([825, 825, 825][:n_rows])),
    ]
    return fits.BinTableHDU.from_columns(columns)


def _write_fits(path: Path, hdu: fits.BinTableHDU) -> None:
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)


def test_parse_fits_file_extracts_real_columns(tmp_path):
    path = tmp_path / "l3atm_0001.fits"
    _write_fits(path, _l3atm_hdu())
    df = parse_fits_file(path)
    for col in ["utc", "longitude", "latitude", "chi2temp", "temp", "taudust"]:
        assert col in df.columns
    assert len(df) == 3


def test_parse_fits_file_rejects_image_extension(tmp_path):
    path = tmp_path / "not_a_table.fits"
    image_hdu = fits.ImageHDU(data=np.zeros((4, 4)))
    fits.HDUList([fits.PrimaryHDU(), image_hdu]).writeto(path)
    with pytest.raises(ValueError, match="IMAGE extension"):
        parse_fits_file(path)


def test_collapse_temp_profile_matches_hand_computation():
    values = [list(range(19)), list(range(10, 29)), None]
    series = pd.Series(values)
    collapsed = collapse_temp_profile(series)
    assert collapsed.iloc[0] == pytest.approx(np.mean(range(19)))
    assert collapsed.iloc[1] == pytest.approx(np.mean(range(10, 29)))
    assert np.isnan(collapsed.iloc[2])


def test_identify_product_distinguishes_l3atm_and_l3emiss(tmp_path):
    schema = load_real_emirs_schema()
    atm_path = tmp_path / "atm.fits"
    emiss_path = tmp_path / "emiss.fits"
    _write_fits(atm_path, _l3atm_hdu())
    _write_fits(emiss_path, _l3emiss_hdu())
    assert identify_product(parse_fits_file(atm_path), schema) == "l3atm"
    assert identify_product(parse_fits_file(emiss_path), schema) == "l3emiss"


def test_load_real_emirs_directory_missing_directory_raises():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_real_emirs_directory(Path("does/not/exist/anywhere"), product="l3atm")


def test_load_real_emirs_directory_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No real l3atm FITS files found"):
        load_real_emirs_directory(tmp_path, product="l3atm")


def test_load_real_emirs_directory_parses_mixed_fractional_second_utc_formats(tmp_path):
    """Real utc strings mix a '.mmm' fractional-second suffix (present when
    milliseconds != 0) with plain 'HH:MM:SS' (when milliseconds == 0) WITHIN
    THE SAME FILE -- confirmed against real downloaded data. A naive
    pd.to_datetime call silently turns the non-matching rows into NaT."""
    utc = np.array(["2021-02-20T11:30:10.998", "2021-02-20T11:30:17", "2021-02-20T11:30:19"])
    columns = [
        fits.Column(name="utc", format="23A", array=utc),
        fits.Column(name="longitude", format="D", array=np.array([10.0, 20.0, 30.0])),
        fits.Column(name="latitude", format="D", array=np.array([-5.0, 0.0, 5.0])),
        fits.Column(name="taudust", format="D", array=np.array([0.1, 0.2, 0.3])),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    _write_fits(tmp_path / "l3atm_0001.fits", hdu)

    df = load_real_emirs_directory(tmp_path, product="l3atm")
    assert df["utc"].isna().sum() == 0


def test_load_real_emirs_directory_drops_unparseable_utc_fill_rows(tmp_path):
    """Real files contain genuine fill/sentinel rows (empty utc string,
    never-geolocated footprints, latitude=longitude=0) -- confirmed against
    real downloaded data, up to ~37% of rows in some files. These must be
    dropped at load time: a row with no timestamp is unusable downstream
    regardless, and EVERY such row shares the identical (NaT, 0, 0) merge
    key -- left in, RealEMIRSObservationProvider's l3atm/l3emiss inner merge
    produces a many-to-many cartesian product for that one key (pandas'
    merge() treats NaT as equal to NaT), which is what caused a real 120 GiB
    allocation failure on the full downloaded dataset (2026-08-04)."""
    utc = np.array(["", "2021-02-20T11:30:17", ""])
    columns = [
        fits.Column(name="utc", format="23A", array=utc),
        fits.Column(name="longitude", format="D", array=np.array([0.0, 20.0, 0.0])),
        fits.Column(name="latitude", format="D", array=np.array([0.0, 5.0, 0.0])),
        fits.Column(name="taudust", format="D", array=np.array([0.1, 0.2, 0.3])),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    _write_fits(tmp_path / "l3atm_0001.fits", hdu)

    df = load_real_emirs_directory(tmp_path, product="l3atm")
    assert len(df) == 1
    assert df["utc"].isna().sum() == 0


def test_load_real_emirs_directory_drops_chi2_null_sentinel_rows(tmp_path):
    """Real l3atm files also contain a SECOND, distinct kind of null row from
    the empty-utc case above: a genuine, unique, parseable utc timestamp, but
    chi2temp AND chi2a (the real per-retrieval goodness-of-fit statistics)
    BOTH exactly 0.0, alongside every other l3atm physical field -- confirmed
    against the real downloaded archive (2026-08-13): 308,186 of 1,385,897
    raw l3atm rows (22.2%). These are not unparseable-utc (so the filter
    above never catches them) and (0,0) is within any sane QC coordinate
    range (so preprocessing/quality_control.py never catches them either) --
    see CHAPTER_4_PARAMETER_VALUES.md Section 11 item 6."""
    utc = np.array(["2021-02-09T00:00:00.000", "2021-02-10T00:00:00.000", "2021-02-11T00:00:00.000"])
    columns = [
        fits.Column(name="utc", format="23A", array=utc),
        fits.Column(name="longitude", format="D", array=np.array([10.0, 0.0, 20.0])),
        fits.Column(name="latitude", format="D", array=np.array([-5.0, 0.0, 5.0])),
        fits.Column(name="chi2temp", format="D", array=np.array([1.1, 0.0, 1.3])),
        fits.Column(name="chi2a", format="D", array=np.array([0.5, 0.0, 0.7])),
        fits.Column(name="taudust", format="D", array=np.array([0.1, 0.0, 0.3])),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    _write_fits(tmp_path / "l3atm_0001.fits", hdu)

    df = load_real_emirs_directory(tmp_path, product="l3atm")

    assert len(df) == 2  # the middle (chi2temp=0, chi2a=0) row is dropped
    assert set(df["latitude"]) == {-5.0, 5.0}


def test_load_real_emirs_directory_keeps_row_with_only_one_chi2_field_zero(tmp_path):
    """A row with only ONE of chi2temp/chi2a exactly 0.0 (not both) must NOT
    be dropped -- the AND condition is deliberate, matching the real
    null-row pattern where every l3atm physical field is simultaneously
    zero, not just one goodness-of-fit statistic in isolation."""
    utc = np.array(["2021-02-09T00:00:00.000", "2021-02-10T00:00:00.000"])
    columns = [
        fits.Column(name="utc", format="23A", array=utc),
        fits.Column(name="longitude", format="D", array=np.array([10.0, 20.0])),
        fits.Column(name="latitude", format="D", array=np.array([-5.0, 5.0])),
        fits.Column(name="chi2temp", format="D", array=np.array([0.0, 1.3])),  # zero, but chi2a is real
        fits.Column(name="chi2a", format="D", array=np.array([0.9, 0.7])),
        fits.Column(name="taudust", format="D", array=np.array([0.2, 0.3])),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    _write_fits(tmp_path / "l3atm_0001.fits", hdu)

    df = load_real_emirs_directory(tmp_path, product="l3atm")
    assert len(df) == 2  # neither row dropped -- only chi2temp was zero, not chi2a too


def test_load_real_emirs_directory_chi2_filter_does_not_apply_to_l3emiss(tmp_path):
    """l3emiss files have no chi2temp/chi2a columns at all -- the filter must
    not raise or drop anything for that product."""
    _write_fits(tmp_path / "l3emiss_0001.fits", _l3emiss_hdu())
    df = load_real_emirs_directory(tmp_path, product="l3emiss")
    assert len(df) == 3


def test_load_real_emirs_directory_parses_and_collapses_real_columns(tmp_path):
    _write_fits(tmp_path / "l3atm_0001.fits", _l3atm_hdu())
    _write_fits(tmp_path / "l3emiss_0001.fits", _l3emiss_hdu())  # must be ignored for product="l3atm"

    df = load_real_emirs_directory(tmp_path, product="l3atm")

    assert len(df) == 3
    assert "kinetic_temp" not in df.columns  # l3emiss-only column excluded
    assert df["temp"].iloc[0] == pytest.approx(np.mean(range(19)))
    assert pd.api.types.is_datetime64_any_dtype(df["utc"])
    assert df["utc"].dt.tz is not None


def test_apply_pipeline_column_mapping_renames_configured_real_columns():
    schema = load_real_emirs_schema()
    df = pd.DataFrame(
        {
            "taudust": [0.1],
            "dustuncert": [0.01],
            "tauice": [0.02],
            "iceuncert": [0.002],
            "kinetic_temp": [210],
            "kinetic_temp_uncertainty": [1.0],
            "pres0": [610.0],
            "utc": [pd.Timestamp("2021-02-20", tz="UTC")],
            "temp": [220.0],
            "qls": [6.4],
            "qlt": [14.0],
            "chi2temp": [1.1],  # not in the mapping -- should pass through unchanged
        }
    )
    out = apply_pipeline_column_mapping(df, schema)
    for real_name in ["taudust", "dustuncert", "tauice", "iceuncert", "kinetic_temp", "kinetic_temp_uncertainty", "pres0", "utc", "temp", "qls", "qlt"]:
        assert real_name not in out.columns
    for canonical_name in [
        "dust_optical_depth", "dust_optical_depth_uncertainty",
        "water_ice_optical_depth", "water_ice_optical_depth_uncertainty",
        "surface_temperature_k", "surface_temperature_k_uncertainty",
        "surface_pressure_mbar", "utc_timestamp", "atmospheric_temperature_mean_k",
        "solar_longitude_ls", "local_true_solar_time",
    ]:
        assert canonical_name in out.columns
    assert out["solar_longitude_ls"].iloc[0] == pytest.approx(6.4)
    assert out["local_true_solar_time"].iloc[0] == pytest.approx(14.0)
    assert "chi2temp" in out.columns


def test_apply_pipeline_column_mapping_raises_on_missing_real_column():
    schema = load_real_emirs_schema()
    df = pd.DataFrame({"taudust": [0.1]})  # missing every other mapped column
    with pytest.raises(KeyError):
        apply_pipeline_column_mapping(df, schema)
