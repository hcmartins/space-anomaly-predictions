"""Descriptive local/regional/planet-encircling storm-taxonomy classifier.

Pure reporting: attaches a taxonomy_class to each QUALIFYING storm episode
for prevalence-by-storm-category tables and Figure-4.7-style plots. Per
configs/storm_taxonomy.yaml's own `classification_target: "reporting_only"`,
this is NEVER used as a machine-learning target -- the ML target is the
generic `storm_episode` label (and its horizon derivatives) defined in
configs/anomaly_definition.yaml, independent of taxonomy class.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR


def load_storm_taxonomy_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "storm_taxonomy.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def classify_episode(
    *,
    peak_area_cells: int,
    duration_sols: int,
    longitude_sectors_spanned: int,
    config: dict | None = None,
) -> str:
    """Returns "local", "regional", "planet_encircling", or "unclassified"
    (an episode matching none of the three bands -- reported as-is rather
    than forced into the nearest class)."""
    config = config or load_storm_taxonomy_config()
    local = config["local"]
    regional = config["regional"]
    planet = config["planet_encircling"]

    if (
        peak_area_cells >= planet["min_area_cells"]
        and duration_sols >= planet["min_duration_sols"]
        and longitude_sectors_spanned >= planet["min_longitude_sectors_spanned"]
    ):
        return "planet_encircling"
    if (
        regional["min_area_cells"] <= peak_area_cells <= regional["max_area_cells"]
        and duration_sols <= regional["max_duration_sols"]
    ):
        return "regional"
    if peak_area_cells <= local["max_area_cells"] and duration_sols <= local["max_duration_sols"]:
        return "local"
    return "unclassified"


def classify_episode_catalogue(episode_catalogue: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Attaches a `taxonomy_class` column to every row of an episode
    catalogue (see labeling/connected_components.py::summarize_episodes)."""
    config = config or load_storm_taxonomy_config()
    out = episode_catalogue.copy()
    if out.empty:
        out["taxonomy_class"] = pd.Series([], dtype=object)
        return out
    out["taxonomy_class"] = out.apply(
        lambda row: classify_episode(
            peak_area_cells=int(row["peak_area_cells"]),
            duration_sols=int(row["duration_sols"]),
            longitude_sectors_spanned=int(row["longitude_sectors_spanned"]),
            config=config,
        ),
        axis=1,
    )
    return out
