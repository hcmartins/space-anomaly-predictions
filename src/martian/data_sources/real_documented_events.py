"""Real, independently-sourced documented dust-storm catalogue -- Guha et al.
(2024)'s EMM/EXI Mars Year 36 dust-storm database (Zenodo, CC-BY-4.0),
used to validate this pipeline's own detected `storm_episode`s against a
genuinely independent real source (Section 4.4.1 / Table 4.6), not a
synthetic stand-in.

Why this specific catalogue: real EMIRS coverage begins 2021-02-20 (MY36),
but the two well-known Mars dust-storm catalogues both end before that --
MDAD (Battalio & Wang 2021) covers MY24-32, MDSSD covers MY24-33 -- so
neither can validate this pipeline's real labels at all. This catalogue
covers exactly MY36, from the same mission (Emirates Mars Mission) as the
real EMIRS data itself, and directly reports the fields validation needs
(storm onset/end time, centroid lat/lon, areal extent) -- see
configs/real_documented_events.yaml for the full citation.

**MY36 only.** This catalogue has no MY37 coverage, so validation against it
is only meaningful for the MY36 portion of this pipeline's real EMIRS range
(mars_year_range [36, 37]) -- real MY37 retrievals have no documented-catalogue
counterpart and are excluded from matching entirely, never silently treated
as "no event occurred."
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .real_emirs_fits import utc_to_mars_year_sol
from .schema import CONFIG_DIR


def load_real_documented_events_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "real_documented_events.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _parse_image_utc(image_filename: pd.Series) -> pd.Series:
    """Real EXI image filenames embed a real UTC timestamp, e.g.
    'emm_exi_l2a_20211129T223059_0140_xos1_f635_p_v03-01' -> 2021-11-29
    22:30:59 UTC -- the same underscore-delimited convention as real EMIRS
    filenames (data_sources/real_emirs_provider.py's own source filenames)."""
    timestamp_token = image_filename.astype(str).str.split("_").str[3]
    return pd.to_datetime(timestamp_token, format="%Y%m%dT%H%M%S", utc=True, errors="coerce")


def load_real_documented_events(config: dict | None = None) -> pd.DataFrame:
    """Loads the real Guha et al. (2024) EMM/EXI MY36 dust-storm database and
    collapses its per-image-detection rows (one row per EXI image a storm
    appears in, ~248 rows total) into one row per documented EVENT, grouped
    by the catalogue's own `Storm no.` column.

    Onset/end UTC -> (mars_year, sol) via the SAME `utc_to_mars_year_sol`
    approximation used throughout this pipeline (`real_emirs_fits.py`), for
    methodological consistency -- not a second, independently-invented
    conversion. A storm essentially never straddles a Mars-Year boundary (all
    are single-mission, day-scale events); `mars_year` is assigned via the
    per-event mode as a defensive fallback if any real detection's timestamp
    lands on the wrong side of the boundary.

    Real longitude is stored West/negative in the source file; converted here
    to this pipeline's 0-360 East convention (`configs/grid.yaml`).
    """
    config = config or load_real_documented_events_config()
    path = Path(config["catalogue_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Real documented dust-storm catalogue not found at {path}. Download it "
            f"from {config.get('source_doi', 'the source DOI in configs/real_documented_events.yaml')} "
            "and place it there -- this pipeline never fabricates a stand-in for it."
        )
    raw = pd.read_excel(path)
    raw.columns = raw.columns.str.strip()  # real source file has a trailing space on "Image " -- confirmed 2026-08-05
    raw["utc"] = _parse_image_utc(raw["Image"])
    raw = raw.loc[raw["utc"].notna()].reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"{path}: no row's Image filename produced a parseable UTC timestamp.")

    mars_year, sol = utc_to_mars_year_sol(raw["utc"])
    raw["mars_year"] = mars_year
    raw["sol"] = sol

    rows: list[dict] = []
    for storm_no, group in raw.groupby("Storm no."):
        # Real, confirmed case (2026-08-05, event/storm 98): a storm can
        # genuinely straddle a Mars-Year boundary (e.g. detections at MY36
        # sol 665-666 and MY37 sol 1, four real days apart). `sol` resets to
        # 0 at the boundary, so naively taking min/max sol across ALL of the
        # group's detections would read the post-boundary sol=1 rows as an
        # EARLIER onset than the pre-boundary sol=665 rows -- a spurious
        # ~665-sol span for what is really a few-day event. Fixed by
        # restricting onset/end/centroid/size to the detections that fall on
        # the SAME side of the boundary as the group's own assigned
        # mars_year (its mode); the minority-year detections are excluded
        # from this event's own summary, not fabricated into a second event.
        mars_year = int(group["mars_year"].mode().iloc[0])
        same_year = group.loc[group["mars_year"] == mars_year]
        rows.append(
            {
                "event_id": int(storm_no),
                "mars_year": mars_year,
                "onset_sol": int(same_year["sol"].min()),
                "end_sol": int(same_year["sol"].max()),
                "onset_utc": same_year["utc"].min(),
                "end_utc": same_year["utc"].max(),
                "centroid_lat": float(same_year["Centroid Latitude"].mean()),
                "centroid_lon": float(same_year["Centroid Longitude"].mean()) % 360.0,
                "peak_size_km2": float(same_year["Size"].max()),
                "n_detections": int(len(group)),
            }
        )
    events = pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)
    events.attrs["citation"] = config.get("citation")
    events.attrs["licence"] = config.get("licence")
    events.attrs["source_doi"] = config.get("source_doi")
    events.attrs["covers_mars_year"] = config.get("covers_mars_year")
    return events
