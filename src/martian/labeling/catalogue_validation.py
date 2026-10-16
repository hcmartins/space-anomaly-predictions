"""Independent-catalogue validation of detected storm_episodes (Table 4.6,
Section 4.4.1) -- real, both sides. Matches this pipeline's own detected
storm_episode catalogue against the real, independently-sourced Guha et al.
(2024) EMM/EXI Mars Year 36 documented dust-storm catalogue
(`data_sources/real_documented_events.py`).

Three things this module is careful never to conflate:

1. **Event detection** -- hits/misses/POD/episode_precision: does a detected
   episode temporally + spatially match a documented event at all?
2. **Event classification is explicitly OUT of scope.** This pipeline's own
   `storm_taxonomy` classification (local/regional/planet_encircling,
   cell-count-based on a 5x5-degree grid) is never compared against the real
   catalogue's own `peak_size_km2` (physical units) -- the two are not
   directly comparable without a unit-conversion/footprint model this
   project has not built, and doing so without one would misrepresent a
   guess as a measurement. Only detection (did an episode occur near this
   documented event, at all) is validated here.
3. **Atmospheric-anomaly detection** -- explicitly out of scope: the
   catalogue records discrete multi-hour/multi-sol events, not pointwise
   exceedances, so there is no principled way to validate
   `atmospheric_anomaly` against it. Only qualifying `storm_episode`s
   (the persistence+area-gated label) are matched.

**MY36 only.** The real documented catalogue has no MY37 coverage
(`configs/real_documented_events.yaml`) -- both detected episodes and
documented events are restricted to `mars_year == documented catalogue's
covers_mars_year` before matching, so real MY37 material is neither
penalised nor credited by this validation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR
from preprocessing.grid import GridDefinition, cell_id, lat_to_band_index, lon_to_band_index, neighbours

UNMATCHED_EPISODE_LABEL_DEFAULT = "false_positive_candidate"


def load_catalogue_validation_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "catalogue_validation.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _region_cell_ids(lat: float, lon: float, grid: GridDefinition, radius_cells: int) -> set[int]:
    """BFS-expanded set of cell ids within `radius_cells` grid-steps of the
    cell containing (lat, lon), inclusive of the centre cell itself."""
    lat_band = int(lat_to_band_index(np.array([lat]), grid)[0])
    lon_band = int(lon_to_band_index(np.array([lon]), grid)[0])
    visited = {(lat_band, lon_band)}
    frontier = set(visited)
    for _ in range(radius_cells):
        next_frontier: set[tuple[int, int]] = set()
        for lb, lo in frontier:
            for n_lb, n_lo in neighbours(lb, lo, grid):
                if (n_lb, n_lo) not in visited:
                    next_frontier.add((n_lb, n_lo))
        visited |= next_frontier
        frontier = next_frontier
    return {int(cell_id(lb, lo, grid)) for lb, lo in visited}


def _episode_member_cells(labeled_df: pd.DataFrame, episode_ids) -> dict:
    if len(episode_ids) == 0:
        return {}
    member_rows = labeled_df.loc[labeled_df["episode_id"].isin(episode_ids)]
    return {eid: set(group["cell_id"].tolist()) for eid, group in member_rows.groupby("episode_id")}


def find_matching_pairs(
    labeled_df: pd.DataFrame,
    episode_catalogue: pd.DataFrame,
    documented_events: pd.DataFrame,
    grid: GridDefinition,
    *,
    temporal_tolerance_sols: int,
    spatial_region_radius_cells: int,
) -> pd.DataFrame:
    """Every (event_id, episode_id) pair satisfying BOTH the temporal-overlap
    and spatial-region criteria. A documented event or a detected episode may
    appear in zero, one, or several pairs -- this is the raw many-to-many
    match relation; `run_catalogue_validation` derives hit/miss/precision
    from it."""
    qualifying = episode_catalogue.loc[episode_catalogue["qualifies"]].reset_index(drop=True)
    episode_cells = _episode_member_cells(labeled_df, qualifying["episode_id"].tolist())

    pairs: list[dict] = []
    for _, event in documented_events.iterrows():
        region = _region_cell_ids(event["centroid_lat"], event["centroid_lon"], grid, spatial_region_radius_cells)
        window_lo = event["onset_sol"] - temporal_tolerance_sols
        window_hi = event["end_sol"] + temporal_tolerance_sols

        same_year = qualifying.loc[qualifying["mars_year"] == event["mars_year"]]
        temporal_candidates = same_year.loc[(same_year["onset_sol"] <= window_hi) & (same_year["end_sol"] >= window_lo)]

        for _, cand in temporal_candidates.iterrows():
            if episode_cells.get(cand["episode_id"], set()) & region:
                pairs.append(
                    {
                        "event_id": event["event_id"],
                        "episode_id": cand["episode_id"],
                        "temporal_offset_sols": abs(int(cand["onset_sol"]) - int(event["onset_sol"])),
                    }
                )
    return pd.DataFrame(pairs, columns=["event_id", "episode_id", "temporal_offset_sols"])


def run_catalogue_validation(
    labeled_df: pd.DataFrame,
    episode_catalogue: pd.DataFrame,
    documented_events: pd.DataFrame,
    grid: GridDefinition,
    *,
    config: dict | None = None,
) -> dict:
    """End-to-end: restricts both sides to the documented catalogue's covered
    Mars Year, finds matching pairs, and returns hit/miss/POD (documented
    side) and false-positive-candidate/precision (detected side) -- the
    real Table 4.6."""
    config = config or load_catalogue_validation_config()
    covered_year = int(documented_events.attrs.get("covers_mars_year") or documented_events["mars_year"].iloc[0])
    unmatched_label = config.get("unmatched_episode_label", UNMATCHED_EPISODE_LABEL_DEFAULT)

    documented_in_scope = documented_events.loc[documented_events["mars_year"] == covered_year].reset_index(drop=True)
    qualifying = episode_catalogue.loc[
        episode_catalogue["qualifies"] & (episode_catalogue["mars_year"] == covered_year)
    ].reset_index(drop=True)

    pairs = find_matching_pairs(
        labeled_df,
        episode_catalogue,
        documented_in_scope,
        grid,
        temporal_tolerance_sols=config["temporal_tolerance_sols"],
        spatial_region_radius_cells=config["spatial_region_radius_cells"],
    )

    matched_event_ids = set(pairs["event_id"].unique().tolist())
    matched_episode_ids = set(pairs["episode_id"].unique().tolist())

    n_documented = len(documented_in_scope)
    n_hits = len(matched_event_ids)
    n_misses = n_documented - n_hits
    pod = (n_hits / n_documented) if n_documented else float("nan")

    n_qualifying = len(qualifying)
    n_matched_episodes = len(matched_episode_ids)
    n_false_positive_candidates = n_qualifying - n_matched_episodes
    episode_precision = (n_matched_episodes / n_qualifying) if n_qualifying else float("nan")

    mean_temporal_offset = float(pairs["temporal_offset_sols"].mean()) if len(pairs) else float("nan")

    per_event = documented_in_scope[["event_id", "mars_year", "onset_sol", "end_sol", "centroid_lat", "centroid_lon", "peak_size_km2"]].copy()
    per_event["hit"] = per_event["event_id"].isin(matched_event_ids)

    per_episode = qualifying[["episode_id", "mars_year", "onset_sol", "end_sol", "taxonomy_class"]].copy() if len(qualifying) else pd.DataFrame(columns=["episode_id", "mars_year", "onset_sol", "end_sol", "taxonomy_class"])
    if len(per_episode):
        per_episode["matched"] = per_episode["episode_id"].isin(matched_episode_ids)
        per_episode["status"] = np.where(per_episode["matched"], "hit", unmatched_label)

    summary = {
        "covered_mars_year": covered_year,
        "n_documented_events": n_documented,
        "n_qualifying_episodes": n_qualifying,
        "n_hits": n_hits,
        "n_misses": n_misses,
        "pod": pod,
        "n_matched_episodes": n_matched_episodes,
        "n_false_positive_candidates": n_false_positive_candidates,
        "episode_precision": episode_precision,
        "mean_temporal_offset_sols": mean_temporal_offset,
    }

    return {
        "summary": summary,
        "matched_pairs": pairs,
        "per_documented_event": per_event,
        "per_episode": per_episode,
    }


def build_table_4_6(result: dict) -> pd.DataFrame:
    """Table 4.6 as a single-row-per-Mars-Year machine-generated table,
    directly from `run_catalogue_validation`'s own summary -- never
    hand-edited."""
    s = result["summary"]
    return pd.DataFrame(
        [
            {
                "mars_year": s["covered_mars_year"],
                "n_documented_events": s["n_documented_events"],
                "n_qualifying_episodes": s["n_qualifying_episodes"],
                "n_hits": s["n_hits"],
                "n_misses": s["n_misses"],
                "pod": s["pod"],
                "n_matched_episodes": s["n_matched_episodes"],
                "n_false_positive_candidates": s["n_false_positive_candidates"],
                "episode_precision": s["episode_precision"],
                "mean_temporal_offset_sols": s["mean_temporal_offset_sols"],
            }
        ]
    )


def write_table_4_6(result: dict, out_dir: Path, *, data_source_label: str) -> tuple[Path, Path]:
    from data_sources.provenance import stamp_table_markdown

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_table_4_6(result)
    csv_path = out_dir / "table_4_6_independent_catalogue_validation.csv"
    md_path = out_dir / "table_4_6_independent_catalogue_validation.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(
        stamp_table_markdown(table, data_source_label, title="Table 4.6 -- Independent-Catalogue Label Validation"),
        encoding="utf-8",
    )
    return csv_path, md_path
