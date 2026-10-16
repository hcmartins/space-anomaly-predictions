"""Real EMIRS L3 FITS parsing (l3atm + l3emiss).

Mirrors the real, working logic from a real prior MSc dissertation on this
exact instrument's data (uploaded 2026-07-29; see
configs/real_emirs_schema.yaml's header) -- vectorized/config-driven per
this project's conventions, not a literal port of that document's per-row
code. Reads REAL FITS files a researcher has manually downloaded into a
local directory (see configs/real_emirs_ingestion.yaml's header for why
this is not yet an automated download).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from astropy.table import Table

from .schema import CONFIG_DIR

SOLS_PER_MARS_YEAR = 668.6   # real Martian sidereal year length, in real Mars sols
MARS_YEAR_1_EPOCH = pd.Timestamp("1955-04-11", tz="UTC")   # standard Mars Year epoch convention


def load_real_emirs_ingestion_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "real_emirs_ingestion.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def load_real_emirs_schema(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "real_emirs_schema.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _product_columns(schema: dict, product: str) -> list[str]:
    """Real column names belonging to `product` ("l3atm" | "l3emiss"), per
    configs/real_emirs_schema.yaml -- a variable's `product` field is either
    a single product name or a list (for merge-key columns shared by both)."""
    columns = []
    for var in schema["variables"]:
        products = var["product"] if isinstance(var["product"], list) else [var["product"]]
        if product in products:
            columns.append(var["name"])
    return columns


def parse_fits_file(path: Path) -> pd.DataFrame:
    """Opens one real FITS file, extracts the first non-image HDU's table
    data into a dataframe. Mirrors the real structural check the reference
    dissertation's own FITS-to-CSV helper uses (its Fig 3.3.1.1b):
    `hdul[1].header['XTENSION'] != 'IMAGE'` -- a genuine FITS table
    extension, not an image HDU."""
    with fits.open(path) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"{path}: no extension HDU (table data expected at index 1)")
        hdu = hdul[1]
        xtension = hdu.header.get("XTENSION")
        if xtension == "IMAGE":
            raise ValueError(f"{path}: HDU 1 is an IMAGE extension, not a table -- not a real l3atm/l3emiss product file")
        return Table(hdu.data).to_pandas()


def collapse_temp_profile(temp_column: pd.Series) -> pd.Series:
    """The real `temp` field is a 19-value array per row (retrieved
    atmospheric temperature every 0.25 pressure scale heights, 0.25 to 4.75
    scale heights above the surface -- configs/real_emirs_schema.yaml's
    vertical_grid). Collapses to the mean, matching the reference
    dissertation's own documented choice (its Fig 3.3.2g helper), not a
    silent reinterpretation of what that array represents."""
    return temp_column.apply(lambda values: float(np.mean(values)) if values is not None and len(values) else np.nan)


def identify_product(df: pd.DataFrame, schema: dict) -> str | None:
    """Determines whether a parsed FITS file's columns belong to l3atm or
    l3emiss by which real, product-specific columns are actually present --
    the reference dissertation's own downloads used a filename convention
    that isn't documented precisely enough to hardcode here (see
    configs/real_emirs_ingestion.yaml's header), so column membership is the
    honest, real signal available. Returns None if neither product's
    distinguishing columns are present (not a FITS file this loader
    recognizes)."""
    for product in schema["product_family"]["products"]:
        distinguishing = [c for c in _product_columns(schema, product) if c not in schema["product_family"]["merge_keys"]]
        if any(c in df.columns for c in distinguishing):
            return product
    return None


def load_real_emirs_directory(
    directory: Path,
    *,
    product: str,
    schema: dict | None = None,
    extensions: tuple[str, ...] = (".fits", ".fit"),
) -> pd.DataFrame:
    """Parses every real FITS file under `directory` (recursive) whose
    columns identify it as belonging to `product`, selects that product's
    real schema columns, converts `utc` to a real datetime, and
    concatenates. Raises a clear, actionable error if the directory does
    not exist or contains no matching real files -- never a silent empty
    result (see data_sources/real_emirs_provider.py's module docstring)."""
    schema = schema or load_real_emirs_schema()
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(
            f"Real EMIRS FITS directory does not exist: {directory}. Populate it by "
            "manually downloading real l3atm/l3emiss FITS files from "
            "https://sdc.emiratesmarsmission.ae/data/science -- see "
            "configs/real_emirs_ingestion.yaml's header for why this is manual."
        )

    fits_paths = [p for p in directory.rglob("*") if p.suffix.lower() in extensions]
    columns = _product_columns(schema, product)

    frames: list[pd.DataFrame] = []
    for path in fits_paths:
        parsed = parse_fits_file(path)
        if identify_product(parsed, schema) != product:
            continue
        present_columns = [c for c in columns if c in parsed.columns]
        frame = parsed[present_columns].copy()
        frame["source_fits_file"] = path.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No real {product} FITS files found under {directory} (scanned "
            f"{len(fits_paths)} file(s) with extensions {extensions}). Confirm the "
            "directory actually contains downloaded real EMIRS FITS files."
        )

    out = pd.concat(frames, ignore_index=True)
    if "temp" in out.columns:
        out["temp"] = collapse_temp_profile(out["temp"])
    if "utc" in out.columns:
        # format="mixed" is REQUIRED here, not cosmetic: real utc strings mix
        # a ".mmm" fractional-second suffix (present when milliseconds != 0)
        # with plain "HH:MM:SS" (when milliseconds == 0) WITHIN THE SAME
        # FILE. pandas' default single-inferred-format fast path silently
        # coerces every row not matching the first-seen format to NaT --
        # confirmed against real downloaded data (2026-08-04): up to ~80% of
        # rows in some files were silently dropped this way before this fix.
        # Empty-string rows genuinely have no timestamp (real fill/gap rows)
        # and correctly remain NaT under format="mixed" too.
        out["utc"] = pd.to_datetime(out["utc"].astype(str), utc=True, errors="coerce", format="mixed")
        # Real files contain genuine fill/sentinel rows (empty utc string,
        # never-geolocated scan-edge footprints) -- confirmed against real
        # downloaded data (2026-08-04): up to ~37% of rows in some files.
        # These are dropped HERE, not later, for two reasons: (1) a row with
        # no timestamp can never get a mars_year/sol and is unusable by any
        # downstream QC/aggregation/labeling step regardless; (2) EVERY such
        # row collapses to the identical (utc=NaT, longitude=0, latitude=0)
        # merge key, and pandas' merge() treats NaT as equal to NaT for join
        # matching (confirmed by direct repro) -- left in, RealEMIRSObservationProvider's
        # l3atm/l3emiss inner merge on this key produces a full many-to-many
        # cartesian product between every such row on each side (reproduced:
        # this is what caused mars.run_all's Step 3 120 GiB allocation
        # failure on the full real dataset).
        out = out.loc[out["utc"].notna()].reset_index(drop=True)

    if "chi2temp" in out.columns and "chi2a" in out.columns:
        # Real l3atm null/placeholder rows, DISTINCT from the empty-utc fill
        # rows dropped above: these carry a genuine, parseable, unique utc
        # timestamp, but chi2temp AND chi2a (the real per-retrieval
        # goodness-of-fit statistics this schema's own header already
        # flagged as "usable for a genuine QC rule... pending" -- see
        # configs/real_emirs_schema.yaml's quality_flag_note) are EXACTLY
        # 0.0, alongside every other l3atm physical field (dust_optical_depth,
        # atmospheric_temperature_mean_k, longitude, latitude, local true
        # solar time, ...) also exactly 0.0 -- confirmed against the real
        # downloaded archive (2026-08-13): 308,186 of 1,385,897 raw l3atm
        # rows (22.2%), all with l3emiss-side fields (surface_temperature_k,
        # kinetic_temp_wave) still real and varying, meaning geolocation/
        # atmosphere retrieval failed for these scans while the emissivity
        # retrieval still produced output. A genuine chi-square goodness-of-
        # fit statistic is not exactly 0.0 for an authentic fit to noisy
        # real spectra; the chance of BOTH chi2temp and chi2a coinciding at
        # exactly 0.0 for a real retrieval is negligible, making this a more
        # reliable identification than the symptomatic (0,0) geolocation
        # these rows also happen to carry (a real Martian latitude/longitude
        # of exactly (0,0) is not itself physically impossible, unlike an
        # exact-zero fit statistic). Previously undetected: these rows are
        # NOT unparseable-utc, so the filter above never caught them, and
        # (0,0) is within any sane coordinate-range QC check, so
        # preprocessing/quality_control.py's impossible_coordinate_or_timestamp
        # rule does not catch them either -- see
        # CHAPTER_4_PARAMETER_VALUES.md Section 11 item 6 for the full
        # writeup and the QC-count impact of dropping them here instead.
        out = out.loc[~((out["chi2temp"] == 0.0) & (out["chi2a"] == 0.0))].reset_index(drop=True)
    return out


def apply_pipeline_column_mapping(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Renames this schema's real field names onto the pipeline's canonical
    column names -- see configs/real_emirs_schema.yaml's
    `pipeline_column_mapping` for the full real-name -> canonical-name table
    and the reasoning behind each mapping."""
    mapping = schema.get("pipeline_column_mapping", {})
    missing = [real_name for real_name in mapping if real_name not in df.columns]
    if missing:
        raise KeyError(
            f"apply_pipeline_column_mapping: real column(s) {missing} not found in "
            f"the merged dataframe (columns present: {sorted(df.columns)}). "
            "configs/real_emirs_schema.yaml's pipeline_column_mapping may be out of "
            "sync with the actual merged l3atm/l3emiss frame."
        )
    return df.rename(columns=mapping)


def utc_to_mars_year_sol(utc: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Real Mars Year / sol from a real UTC timestamp -- ENGINEERING
    DECISION: a documented linear approximation (elapsed real days since
    the standard Mars Year 1 epoch, April 11 1955, divided by the real
    Martian sidereal year length of ~668.6 sols), NOT a full areocentric
    solar-longitude (Ls) based calculation (which needs Mars's real orbital
    eccentricity/equation-of-center and is a materially larger undertaking
    than this data-collection phase's scope). RESEARCHER VERIFICATION
    REQUIRED before using this for anything requiring exact Ls-based Mars
    Year boundaries (e.g. precise seasonal analysis) -- adequate for
    approximate chronological ordering/splitting, not exact science-grade
    Mars Year assignment."""
    elapsed_days = (utc - MARS_YEAR_1_EPOCH).dt.total_seconds() / 86400.0
    elapsed_sols = elapsed_days / 1.0274912510   # 1 sol = 1.0274912510 Earth days
    mars_year = (elapsed_sols // SOLS_PER_MARS_YEAR).astype("int64") + 1
    sol = (elapsed_sols % SOLS_PER_MARS_YEAR).astype("int64")
    return mars_year, sol
