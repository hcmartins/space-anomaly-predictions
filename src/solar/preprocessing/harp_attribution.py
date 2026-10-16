"""NOAA AR -> HARPNUM attribution (Chapter 3.2.1/Algorithm 3.2's H
mapping), and cataloged-flare-to-HARP attribution built on top of it.

Phase 1 of scoping out per-active-region forecasting (see planning/
FLARE_CATALOGUE_DOCUMENTATION.md) -- this module is purely additive: it
does not touch the existing disk-integrated preprocessing/labeling
pipeline. It answers, for each real cataloged flare
(data_sources/flare_catalogue_ingestion.py) with a NOAA active_region
assigned, which HARPNUM(s) (if any) enclosed that AR around the flare's
own time -- using SHARP's NOAA_ARS keyword, which can change over a
HARP's lifetime as regions merge/split/rotate off-disk, so attribution is
matched NEAR the flare's OWN time, not from a single static snapshot..

Fully vectorized (merge/explode/groupby) -- no per-row Python loop, per
this session's `preprocessing/features.py` lesson (a per-row
dict/loop-based implementation measured at ~76ms/row and became a real
~14-hour bottleneck at full scale; this module is written vectorized from
the start rather than repeating that mistake).
"""
from __future__ import annotations

import pandas as pd


def parse_noaa_ars(value) -> set[str]:
    """NOAA_ARS is a comma-separated string of AR numbers, e.g. "12673,12674",
    or empty/NaN if the HARP encloses none. Returns bare AR-number strings.

    CONFIRMED against live JSOC data 2026-07-26: NOAA_ARS reports the
    FULL 5-digit AR number (e.g. "12673", matching the real HARP 7115 /
    AR 12673 pairing from the 2017-09 pilot -- the flare catalogue's
    `active_region` column uses the abbreviated 4-digit form instead, so
    both forms are registered here for matching). Also confirmed: a HARP
    with no assigned NOAA AR reports the literal SENTINEL STRING
    "MISSING" (not NaN, not "0") -- filtered out below via the
    digits-only check, which also guards against any other future
    non-numeric sentinel JSOC might use."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    raw = [p.strip() for p in str(value).split(",") if p.strip().isdigit()]
    out: set[str] = set()
    for r in raw:
        out.add(r)
        if len(r) == 5 and r.startswith("1"):
            out.add(r[1:])
    return out


def build_harp_ar_index(sharp_df: pd.DataFrame, *, time_col: str = "T_REC") -> pd.DataFrame:
    """Explodes SHARP's NOAA_ARS keyword into one row per (HARPNUM,
    active_region, time_utc) -- the index `attribute_flares_to_harps`
    queries against. Requires sharp_df to already carry a `NOAA_ARS`
    column (configs/sharp.yaml's bookkeeping_keywords) and a parsed
    time_utc-equivalent column (pass `time_col` for SHARP's own
    TAI-string T_REC, or a pre-parsed alignment.py-style time_utc)."""
    required = {"HARPNUM", time_col, "NOAA_ARS"}
    missing = required - set(sharp_df.columns)
    if missing:
        raise KeyError(f"sharp_df is missing required column(s): {sorted(missing)}")

    df = sharp_df[["HARPNUM", time_col, "NOAA_ARS"]].dropna(subset=["HARPNUM", time_col]).copy()
    df["NOAA_ARS"] = df["NOAA_ARS"].astype(str)
    df = df.loc[df["NOAA_ARS"].str.strip().str.len() > 0]
    if df.empty:
        return pd.DataFrame(columns=["HARPNUM", "time_utc", "active_region"])

    df["active_region"] = df["NOAA_ARS"].str.split(",")
    df = df.explode("active_region")
    df["active_region"] = df["active_region"].str.strip()
    df = df.loc[(df["active_region"] != "") & (df["active_region"].str.lower() != "nan")]

    # Also register the abbreviated 4-digit form for any 5-digit AR
    # starting with "1" (see parse_noaa_ars's verification note).
    abbreviated = df.loc[df["active_region"].str.len().eq(5) & df["active_region"].str.startswith("1")].copy()
    abbreviated["active_region"] = abbreviated["active_region"].str[1:]
    df = pd.concat([df, abbreviated], ignore_index=True)

    return (
        df.rename(columns={time_col: "time_utc"})[["HARPNUM", "time_utc", "active_region"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def attribute_flares_to_harps(
    flare_catalogue_df: pd.DataFrame,
    harp_ar_index: pd.DataFrame,
    *,
    time_tolerance_hours: float = 12.0,
) -> pd.DataFrame:
    """Returns a COPY of flare_catalogue_df with a new `harpnums` column
    (a list of int HARPNUMs, possibly empty). A flare is attributed to
    every HARPNUM whose `NOAA_ARS` listed that flare's `active_region`
    within `time_tolerance_hours` of the flare's own peak time. Flares
    with no `active_region` (Algorithm 3.2 step 11's unassigned-flare
    case) always get an empty list here -- attribution is impossible
    without a real AR to match on; masking policy for these rows is
    Phase 2's responsibility, deliberately not decided in this module.

    `time_tolerance_hours` default of 12h is an ENGINEERING DECISION (not
    thesis-specified -- Algorithm 3.2 does not state one): wide enough to
    survive real SHARP cadence gaps around a flare's peak time, narrow
    enough that a HARP's AR composition genuinely hasn't changed. Revisit
    once Phase 1's real attribution-coverage numbers are in hand."""
    out = flare_catalogue_df.copy()
    out["harpnums"] = [[] for _ in range(len(out))]

    assigned = out.loc[out["active_region"] != ""]
    if assigned.empty or harp_ar_index.empty:
        return out

    merged = assigned[["flare_id", "time_utc_peak", "active_region"]].merge(
        harp_ar_index, on="active_region", how="inner"
    )
    if merged.empty:
        return out

    time_diff_hours = (merged["time_utc_peak"] - merged["time_utc"]).abs().dt.total_seconds() / 3600.0
    within_tolerance = merged.loc[time_diff_hours <= time_tolerance_hours]
    if within_tolerance.empty:
        return out

    harpnums_by_flare = (
        within_tolerance.groupby("flare_id")["HARPNUM"]
        .apply(lambda s: sorted(int(h) for h in s.dropna().unique()))
    )
    out["harpnums"] = out["flare_id"].map(harpnums_by_flare)
    out["harpnums"] = out["harpnums"].apply(lambda v: v if isinstance(v, list) else [])
    return out


def attribution_quality_report(attributed_df: pd.DataFrame) -> dict:
    """Real, honest coverage numbers -- never asserted from memory. Used
    to fill Chapter 3's Table 3.1 and to decide whether Phase 2 (the full
    per-HARP pipeline rewrite) is worth its cost given real attribution
    coverage."""
    n_total = len(attributed_df)
    n_with_ar = int((attributed_df["active_region"] != "").sum())
    n_attributed = int(attributed_df["harpnums"].apply(bool).sum())
    n_multi_harp = int(attributed_df["harpnums"].apply(lambda v: len(v) > 1).sum())
    return {
        "n_flares_total": n_total,
        "n_flares_with_active_region": n_with_ar,
        "pct_flares_with_active_region": (n_with_ar / n_total * 100.0) if n_total else float("nan"),
        "n_flares_attributed_to_at_least_one_harp": n_attributed,
        "pct_with_ar_successfully_attributed": (n_attributed / n_with_ar * 100.0) if n_with_ar else float("nan"),
        "n_flares_attributed_to_multiple_harps": n_multi_harp,
    }
