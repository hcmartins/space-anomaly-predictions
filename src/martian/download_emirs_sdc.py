#!/usr/bin/env python3
"""Bulk-downloads real EMIRS L3 products (l3atm + l3emiss) from the EMM
Science Data Center, using the SDC REST API (AWS Cognito auth + API
Gateway), and writes them into `configs/real_emirs_ingestion.yaml`'s
`local_fits_directory` (default `data_cache/emirs/raw`) -- exactly where
`data_sources/real_emirs_provider.py::RealEMIRSObservationProvider` expects
real FITS files to already be.

A second, independent step (`--process-only` to run alone, skipped by
`--download-only`) converts the downloaded FITS into a single merged,
geolocated table (parquet, or CSV if pyarrow isn't installed) -- a
standalone convenience export, NOT used by `RealEMIRSObservationProvider`
(which reads the raw FITS directory directly and does its own l3atm/l3emiss
merge + canonical-column mapping, see configs/real_emirs_schema.yaml's
`pipeline_column_mapping`). Useful for ad-hoc inspection/exploration outside
the pipeline.

--------------------------------------------------------------------------------
CREDENTIALS -- NEVER hard-coded, never written to any file by this script
--------------------------------------------------------------------------------
Set your SDC login as environment variables before running (or you will be
prompted securely at runtime, with the password never echoed):

    export EMM_SDC_USER="you@example.ac.uk"
    export EMM_SDC_PASS="your-sdc-password"      # do NOT commit this anywhere

Do not paste real credentials into a chat, a config file, or a committed
script. If you ever do, treat that password as compromised and rotate it.

--------------------------------------------------------------------------------
NOTES / THINGS CONFIRMED against real downloaded data (2026-08-04)
--------------------------------------------------------------------------------
* The official SDC scripts (SDC website -> Software tab) are the reference
  implementation; this file follows the same documented flow, per real
  researcher-supplied guidance. If a call shape ever changes, cross-check
  there or email emm-sdc@lists.lasp.colorado.edu.
* `instrument_id` (not `instrument`) and `start_date`/`end_date` (not
  `date_start`/`date_end`) are the correct metadata-query parameter names --
  confirmed live.
* `data_level` MUST be the exact product descriptor ("l3atm", "l3emiss") --
  the generic "l3" is silently accepted by the API but always returns an
  empty list, no error. query_month() below queries once per descriptor.
* The API Gateway times out at 30 s, so the metadata query is chunked by
  month and files are downloaded individually (file=... => direct link, no
  tar).
* EMIRS only exists from ~MY36 (Feb 2021 onward) -- unlike the real Montabone
  MCD dust climatology this pipeline already uses (data_cache/griddedcdod_MY{34,35,36}*.nc,
  build_mcd_climatology_from_montabone.py), which covers MY34-36. There is no
  real EMIRS data to retrieve for MY34/MY35.
* Solar longitude and local true solar time ARE real, native per-footprint
  FITS columns (no computation needed) -- but NOT under the EMM EMIRS Data
  Product Guide Appendix A's documented labels (`l_sub_s`, `ltime` on
  l3atm). Every real downloaded l3atm file (all ~30 file-format versions
  checked) actually names them `qls`/`qlt` instead; only l3emiss uses the
  documented `ltime` name (and has no Ls field at all). Confirmed genuinely
  Ls/LTST, not quality flags despite the "q" prefix: `qls`/`qlt` values
  match l3emiss's `ltime` row-for-row for the same footprint, and `qls`'s
  ~6.4-degree value on a 2021-02-20 file is exactly right for a date just
  past EMIRS's documented MY36/vernal-equinox science start (Ls=0). See
  configs/real_emirs_schema.yaml's qls/qlt entries for the full evidence.
  FIELDS_COMMON below lists the documented names too, as a fallback, in
  case a future file version reverts to them.

Usage:
    cd src/martian
    python download_emirs_sdc.py                          # download, then build the merged table
    python download_emirs_sdc.py --start-date 2021-02-20 --end-date 2021-03-20 --limit 20
        # a small, fast first run to confirm auth + the API shape work at all
    python download_emirs_sdc.py --out-dir data_cache/emirs/raw
    python download_emirs_sdc.py --download-only          # skip the merged-table export
    python download_emirs_sdc.py --process-only            # (re)build the merged table from an already-downloaded --out-dir
    python download_emirs_sdc.py --keep-profile            # also export all 19 raw temp levels, not just their mean
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

MARTIAN_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = MARTIAN_ROOT / "data_cache" / "emirs" / "raw"  # matches configs/real_emirs_ingestion.yaml's local_fits_directory
DEFAULT_MERGED_OUT = MARTIAN_ROOT / "data_cache" / "emirs" / "emirs_l3_merged.parquet"

DESCRIPTORS = ["l3atm", "l3emiss"]  # the two products configs/real_emirs_ingestion.yaml declares
DEFAULT_START_DATE = date(2021, 2, 20)  # EMIRS science start (~MY36)
INSTRUMENT = "emr"  # EMIRS instrument id
LATEST_ONLY = "true"  # only latest version of each file

COGNITO_URL = "https://cognito-idp.eu-west-1.amazonaws.com/"
CLIENT_ID = "n5e6d97bl4ba76rrdtm0qaq6n"  # public SDC app client id (from API docs)
API = "https://mdhkq4bfae.execute-api.eu-west-1.amazonaws.com/prod"
META_EP = f"{API}/science-files-metadata"
DOWN_EP = f"{API}/science-files-download"

TOKEN_TTL = 45 * 60  # re-auth every 45 min (tokens ~1h)
RETRIES = 4

# =========================================================================== #
#  PART 1 -- DOWNLOAD
# =========================================================================== #


def get_token(user: str, pw: str) -> str:
    """Exchange username/password for a Cognito access token (JWT). Never
    logs or persists the password itself."""
    r = requests.post(
        COGNITO_URL,
        headers={
            "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
            "Content-Type": "application/x-amz-json-1.1",
        },
        data=json.dumps(
            {
                "ClientId": CLIENT_ID,
                "AuthFlow": "USER_PASSWORD_AUTH",
                "AuthParameters": {"USERNAME": user, "PASSWORD": pw},
            }
        ),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["AuthenticationResult"]["AccessToken"]


class Session:
    """Holds the token and refreshes it before expiry / on 401. Keeps the
    password in memory only (never written to disk, never logged)."""

    def __init__(self, user: str, pw: str):
        self.user, self._pw = user, pw
        self._refresh()

    def _refresh(self) -> None:
        self.token = get_token(self.user, self._pw)
        self.stamp = time.time()

    def headers(self) -> dict:
        if time.time() - self.stamp > TOKEN_TTL:
            self._refresh()
        return {"Accept": "application/json, text/plain, */*", "Authorization": self.token}

    def get(self, url: str, params: dict) -> requests.Response:
        for attempt in range(RETRIES):
            try:
                r = requests.get(url, params=params, headers=self.headers(), timeout=60)
                if r.status_code == 401:  # token expired -> refresh once
                    self._refresh()
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(2**attempt)  # exponential backoff
        raise RuntimeError("unreachable")  # pragma: no cover


def month_ranges(start: date, end: date):
    cur = start
    while cur <= end:
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        yield cur, min(nxt - timedelta(days=1), end)
        cur = nxt


def query_month(sess: Session, d0: date, d1: date) -> list[dict]:
    """Return metadata list for all EMIRS files in [d0, d1], across both
    DESCRIPTORS. `data_level` must be the exact product descriptor
    ("l3atm", "l3emiss") -- the generic "l3" is silently accepted but always
    returns an empty list, no error (confirmed live)."""
    out: list[dict] = []
    for desc in DESCRIPTORS:
        params = {
            "instrument_id": INSTRUMENT,
            "data_level": desc,
            "latest": LATEST_ONLY,
            "start_date": d0.isoformat(),
            "end_date": d1.isoformat(),
        }
        r = sess.get(META_EP, params)
        try:
            out.extend(r.json())
        except ValueError:
            pass
    return out


# --------------------------------------------------------------------------- #
# DOWNLOAD ONE FILE
# --------------------------------------------------------------------------- #
def resolve_download_url(sess: Session, fname: str) -> str | None:
    """file=<name> => a direct link to that single file (no tar)."""
    r = sess.get(DOWN_EP, {"file": fname})
    body = r.text.strip()
    try:
        j = r.json()
        if isinstance(j, str):
            return j
        if isinstance(j, dict):
            for k in ("url", "download_url", "link", "presigned_url"):
                if j.get(k):
                    return j[k]
            for v in j.values():  # fall back: first http-looking value
                if isinstance(v, str) and v.startswith("http"):
                    return v
    except ValueError:
        pass
    return body if body.startswith("http") else None


def md5_ok(path: Path, expected: str | None) -> bool:
    if not expected:
        return True
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def download_file(sess: Session, meta: dict, out_dir: Path) -> str:
    fname = meta["file_name"]
    dest = out_dir / fname
    if dest.exists() and md5_ok(dest, meta.get("md5checksum")):
        return "skip"  # already have a good copy
    url = resolve_download_url(sess, fname)
    if not url:
        print(f"  ! could not resolve URL for {fname}")
        return "fail"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if not md5_ok(tmp, meta.get("md5checksum")):
        tmp.unlink()
        print(f"  ! md5 mismatch for {fname}")
        return "fail"
    tmp.replace(dest)
    return "ok"


def run_download(*, out_dir: Path, start: date, end: date, limit: int | None) -> None:
    user = os.environ.get("EMM_SDC_USER") or input("SDC username (email): ").strip()
    pw = os.environ.get("EMM_SDC_PASS") or getpass.getpass("SDC password: ")
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = Session(user, pw)
    print("Authenticated. Starting download...")

    tally = {"ok": 0, "skip": 0, "fail": 0}
    total_seen = 0
    for d0, d1 in month_ranges(start, end):
        if limit is not None and total_seen >= limit:
            break
        files = query_month(sess, d0, d1)  # already filtered to DESCRIPTORS server-side via data_level
        print(f"{d0:%Y-%m}: {len(files)} EMIRS L3 files")
        for m in files:
            if limit is not None and total_seen >= limit:
                break
            status = download_file(sess, m, out_dir)
            tally[status] += 1
            total_seen += 1
            if status == "ok":
                ls = m.get("mars_ls")
                print(f"  + {m['file_name']}" + (f"  (Ls={ls:.1f})" if ls is not None else ""))

    print(f"\nDownload done. ok={tally['ok']}  skip={tally['skip']}  fail={tally['fail']}")
    print(f"Files in: {out_dir.resolve()}")


# =========================================================================== #
#  PART 2 -- FITS -> merged geolocated table (keeps solar longitude + LTST)
# =========================================================================== #
# logical name -> candidate FITS column labels. Ordered by how they're
# actually confirmed present in real downloaded data (first candidate wins);
# later candidates are the EMM EMIRS Data Product Guide's documented names,
# kept as a fallback in case a future file version reverts to them -- see
# this module's own header note.
FIELDS_COMMON = {
    "utc": ["utc"],
    "latitude": ["latitude"],
    "longitude": ["longitude"],
    "solar_longitude_ls": ["qls", "l_sub_s", "ls", "solar_longitude"],
    "local_true_solar_time": ["qlt", "ltime", "local_true_solar_time", "local_time"],
    "incidence_angle": ["incidence"],
    "emission_angle": ["emission"],
}
FIELDS_ATM = {
    "dust_optical_depth": ["taudust"],
    "dust_optical_depth_uncertainty": ["dustuncert"],
    "dust_quality_flag": ["taudust_quality_flag"],
    "water_ice_optical_depth": ["tauice"],
    "water_ice_uncertainty": ["iceuncert"],
    "water_ice_quality_flag": ["tauice_quality_flag"],
    "water_vapor_prum": ["prum"],
    "water_vapor_uncertainty": ["wateruncert"],
    "water_vapor_quality_flag": ["prum_quality_flag"],
    "surface_pressure_mbar": ["pres0", "press0"],
    "zdust": ["zdust"],
    "zh2o": ["zh2o", "zh20"],
    "chi2temp": ["chi2temp"],
    "chi2a": ["chi2a"],
    "atm_temp_quality_flag": ["temp_quality_flag"],
}
FIELDS_EMISS = {
    "surface_temperature_k": ["kinetic_temp"],
    "surface_temperature_uncertainty": ["kinetic_temp_uncertainty"],
    "surface_temperature_wave": ["kinetic_temp_wave"],
    "emissivity": ["emissivity"],
}


def _pick(colnames, candidates):
    low = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _as_str(col):
    import numpy as np

    if col.dtype.kind in ("S", "a"):
        return np.char.decode(col.astype("S"), "utf-8", "ignore")
    return col.astype(str)


def _native_byteorder(arr):
    """FITS stores numeric arrays big-endian (dtype e.g. '>f8'); pyarrow's
    parquet writer rejects that directly ("Byte-swapped arrays not
    supported"). astype() (not view()) performs the actual conversion, not
    just a reinterpretation."""
    import numpy as np

    arr = np.asarray(arr)
    if arr.dtype.kind in ("f", "i", "u") and arr.dtype.byteorder not in ("=", "|"):
        return arr.astype(arr.dtype.newbyteorder("="))
    return arr


def _read_fits(path, field_map, *, keep_profile: bool):
    """Return a DataFrame with the requested logical fields (those present)."""
    import numpy as np
    import pandas as pd
    from astropy.io import fits

    with fits.open(path) as h:
        d = h[1].data
        cols = d.columns.names
        out = {}
        for logical, cands in field_map.items():
            src = _pick(cols, cands)
            if src is None:
                continue
            col = d[src]
            if np.asarray(col).ndim != 1:
                # e.g. real l3emiss `emissivity` is a (N, 700) per-wavenumber
                # spectrum, not a scalar per footprint -- there is no single
                # representative value to collapse it to (unlike `temp`,
                # whose mean is the reference dissertation's own documented
                # choice), so it's left out of the merged table (column stays
                # NaN-filled after reindex) rather than fabricating a
                # collapse method nobody asked for.
                continue
            out[logical] = _as_str(col) if logical == "utc" else _native_byteorder(col)
        tsrc = _pick(cols, ["temp"])
        if tsrc is not None:
            arr = _native_byteorder(np.asarray(d[tsrc], dtype=float))  # (N, 19) real pressure-scale-height levels
            valid = np.where(arr > 0, arr, np.nan)  # drop fills
            out["atmospheric_temperature_mean_k"] = np.nanmean(valid, axis=1)
            if keep_profile:
                for i in range(arr.shape[1]):
                    out[f"atm_temp_k_level_{i + 1:02d}"] = arr[:, i]
        df = pd.DataFrame(out)
        if "utc" in df:
            # format="mixed" is required: real utc strings mix a ".mmm"
            # fractional-second suffix with plain "HH:MM:SS" within the
            # same file -- see data_sources/real_emirs_fits.py's
            # load_real_emirs_directory for the full explanation.
            df["utc"] = pd.to_datetime(df["utc"].astype(str).str.strip(), utc=True, errors="coerce", format="mixed")
        return df


def _file_key(fname):
    """startTime_orbit key shared by an l3atm/l3emiss pair, e.g.
    emm_emr_l3atm_20210220t112122_0008_f_v02-20.fits and
    emm_emr_l3emiss_20210220t112122_0008_f_v04-06.fits both key to
    '20210220t112122_0008' -- confirmed against real downloaded filenames."""
    p = os.path.basename(fname).split("_")
    return f"{p[3]}_{p[4]}" if len(p) > 4 else os.path.basename(fname)


def run_process(*, out_dir: Path, merged_out: Path, keep_profile: bool) -> None:
    import glob

    import pandas as pd

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        use_parquet = True
    except ImportError:
        use_parquet = False
        print("pyarrow not found -> writing CSV instead of parquet")

    atm_files = sorted(glob.glob(str(out_dir / "*l3atm*.fits")))
    emiss_index = {_file_key(f): f for f in glob.glob(str(out_dir / "*l3emiss*.fits"))}
    print(f"Pairing {len(atm_files)} l3atm files with {len(emiss_index)} l3emiss files")

    out_cols = (
        list(FIELDS_COMMON) + ["atmospheric_temperature_mean_k"] + list(FIELDS_ATM) + list(FIELDS_EMISS)
    )
    if keep_profile:
        out_cols += [f"atm_temp_k_level_{i + 1:02d}" for i in range(19)]

    merged_out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    csv_out = merged_out.with_suffix(".csv")
    n_rows = n_pairs = 0

    for apath in atm_files:
        epath = emiss_index.get(_file_key(apath))
        if epath is None:
            continue
        atm = _read_fits(apath, {**FIELDS_COMMON, **FIELDS_ATM}, keep_profile=keep_profile)
        emiss = _read_fits(epath, {**FIELDS_COMMON, **FIELDS_EMISS}, keep_profile=False)
        # merge on shared footprint geometry (utc + lat/lon), matching
        # RealEMIRSObservationProvider's own l3atm/l3emiss merge keys
        # (configs/real_emirs_schema.yaml's product_family.merge_keys).
        keep_e = ["utc", "latitude", "longitude"] + list(FIELDS_EMISS)
        merged = atm.merge(emiss[[c for c in keep_e if c in emiss]], on=["utc", "latitude", "longitude"], how="inner")
        if merged.empty:
            continue
        merged = merged.reindex(columns=out_cols)  # fixed schema, NaN-fill
        n_rows += len(merged)
        n_pairs += 1

        if use_parquet:
            table = pa.Table.from_pandas(merged, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(merged_out), table.schema)
            writer.write_table(table)
        else:
            merged.to_csv(csv_out, mode="a", header=(n_pairs == 1), index=False)

        if n_pairs % 500 == 0:
            print(f"  merged {n_pairs} pairs, {n_rows} rows so far")

    if writer is not None:
        writer.close()
    target = merged_out if use_parquet else csv_out
    print(f"\nWrote {n_rows} rows from {n_pairs} paired observations -> {target}")
    print("Columns include solar_longitude_ls and local_true_solar_time, ready for the MCD z-score run.")


# =========================================================================== #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Where FITS files are/will be saved (default: configs/real_emirs_ingestion.yaml's local_fits_directory).")
    parser.add_argument("--merged-out", default=str(DEFAULT_MERGED_OUT), help="Path for the merged table (default: data_cache/emirs/emirs_l3_merged.parquet).")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat(), help="YYYY-MM-DD (default: EMIRS science start, 2021-02-20).")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="YYYY-MM-DD (default: today). Narrow this for a fast test run.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many files total -- useful for a first smoke-test run before committing to the full range.")
    parser.add_argument("--download-only", action="store_true", help="Skip building the merged table.")
    parser.add_argument("--process-only", action="store_true", help="Skip downloading; (re)build the merged table from --out-dir.")
    parser.add_argument("--keep-profile", action="store_true", help="Also export all 19 raw temp levels (atm_temp_k_level_01..19), not just their mean.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    merged_out = Path(args.merged_out)
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    if not args.process_only:
        run_download(out_dir=out_dir, start=start, end=end, limit=args.limit)
    if not args.download_only:
        run_process(out_dir=out_dir, merged_out=merged_out, keep_profile=args.keep_profile)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit("\nInterrupted (safe to re-run; completed files are skipped).")
