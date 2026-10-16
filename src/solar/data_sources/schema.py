"""Canonical normalized schema for ingested GOES and SHARP data, and shared
config-loading helpers.

Both ingestion modules (`goes_ingestion.py`, `sharp_ingestion.py`) parse
heterogeneous real-world source formats (differing NOAA satellite
generations; JSOC keyword tables) into these fixed, documented column sets,
so downstream preprocessing never needs to know which satellite generation
or export protocol a given row came from.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

# Columns present in every normalized GOES XRS record, regardless of which
# satellite generation (GOES-R vs legacy) it was sourced from.
GOES_NORMALIZED_COLUMNS = [
    "time_utc",          # pandas Timestamp, UTC
    "satellite",          # int, e.g. 16
    "satellite_series",   # "goes_r" | "goes_legacy"
    "xrsa_flux_w_m2",      # short channel (0.5-4 A), W m^-2
    "xrsb_flux_w_m2",      # long channel (1-8 A), W m^-2 -- used for flare classification
    "xrsa_quality",
    "xrsb_quality",
    "source_file",         # original filename, for traceability
]

# Columns present in every normalized SHARP record: the 18 physical indices
# (Table 7.1) plus bookkeeping keywords, all requested explicitly in
# configs/sharp.yaml -- never silently substituted if a keyword is missing
# from a given record (missing values are preserved as NaN, not dropped or
# imputed at ingestion time).
SHARP_PHYSICAL_KEYWORDS = [
    "USFLUX", "TOTUSJZ", "TOTUSJH", "ABSNJZH", "SAVNCPP", "MEANPOT",
    "TOTPOT", "R_VALUE", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH",
    "MEANALP", "MEANJZH", "MEANJZD", "MEANSHR", "AREA_ACR", "SHRGT45",
]
SHARP_BOOKKEEPING_KEYWORDS = ["HARPNUM", "T_REC", "NOAA_AR", "NOAA_ARS", "LAT_FWT", "LON_FWT", "QUALITY"]
SHARP_NORMALIZED_COLUMNS = SHARP_BOOKKEEPING_KEYWORDS + SHARP_PHYSICAL_KEYWORDS

# Columns present in every normalized flare-catalogue record (NOAA's real
# GOES Flare Report, NCEI -- see configs/flare_catalogue.yaml's header for
# the verified real archive URL/format). `active_region` is the NOAA AR
# number as a string, or "" when SWPC could not attribute the flare to a
# region (a real, documented, non-rare case -- confirmed against live data
# 2026-07-26: even some X-class flares in the well-known 2017-09-04..10
# window lack an AR) -- never fabricated, never silently dropped.
FLARE_CATALOGUE_NORMALIZED_COLUMNS = [
    "flare_id",                  # NOAA's own flare start-time-derived ID
    "time_utc_peak",             # pandas Timestamp, UTC
    "time_utc_start",
    "time_utc_end",
    "flare_class",                # e.g. "X9.3" -- raw string, not re-derived
    "xrsb_irrad_peak_w_m2",       # numeric peak flux -- classify against configs/goes.yaml's thresholds for consistency with the rest of this project, not by re-parsing flare_class
    "active_region",              # NOAA AR number as string, "" if unassigned
]


def load_goes_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "goes.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def load_sharp_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "sharp.yaml"
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    # JSOC_EMAIL env var overrides the config so a personal address need not be committed.
    if os.environ.get("JSOC_EMAIL"):
        config["jsoc_email"] = os.environ["JSOC_EMAIL"]
    if not config.get("jsoc_email"):
        raise ValueError(
            "configs/sharp.yaml: 'jsoc_email' is required and has no default. "
            "Register a free email at http://jsoc.stanford.edu/ajax/register_email.html "
            "and set it in the config (or the JSOC_EMAIL env var) before running SHARP ingestion."
        )
    return config


def load_flare_catalogue_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "flare_catalogue.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
