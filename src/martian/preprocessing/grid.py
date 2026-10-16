"""Spatial grid definition for Martian cell-sol aggregation.

Grid geometry only — no physical-variable logic lives here. Consumed by
preprocessing/aggregation.py and preprocessing/sensitivity.py.

Convention (see configs/grid.yaml):
  - longitude: 0-360 degrees East, WITH wraparound (the last longitude band is
    adjacent to the first)
  - latitude: -90 to 90 degrees, NO wraparound (the poles are edges of the
    grid, not a torus)
  - eight-neighbour adjacency in the interior; a cell in the topmost or
    bottommost latitude band has only FIVE neighbours, because there is no
    row beyond the pole to wrap into — this is deliberate and tested
    explicitly (see tests/test_grid.py), not a bug
  - polar cells are RETAINED, never merged or excluded, per configs/grid.yaml
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR


@dataclasses.dataclass(frozen=True)
class GridDefinition:
    resolution_deg: float
    n_lat_bands: int
    n_lon_bands: int
    n_cells: int

    @classmethod
    def from_resolution(cls, resolution_deg: float) -> "GridDefinition":
        if not np.isclose(180.0 % resolution_deg, 0.0) and not np.isclose(180.0 % resolution_deg, resolution_deg):
            raise ValueError(
                f"resolution_deg={resolution_deg} does not evenly divide 180 "
                f"degrees of latitude; a grid built from this resolution would "
                f"have a partial band."
            )
        if not np.isclose(360.0 % resolution_deg, 0.0) and not np.isclose(360.0 % resolution_deg, resolution_deg):
            raise ValueError(
                f"resolution_deg={resolution_deg} does not evenly divide 360 "
                f"degrees of longitude; a grid built from this resolution "
                f"would have a partial band."
            )
        n_lat = round(180.0 / resolution_deg)
        n_lon = round(360.0 / resolution_deg)
        return cls(resolution_deg=resolution_deg, n_lat_bands=n_lat, n_lon_bands=n_lon, n_cells=n_lat * n_lon)


def load_primary_grid(config_path: Path | None = None) -> GridDefinition:
    config_path = config_path or CONFIG_DIR / "grid.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    grid = GridDefinition.from_resolution(raw["resolution_deg"])
    # Cross-check against the config's own stated expectations, to catch drift
    # if grid.yaml is edited without updating its recorded n_lat_bands etc.
    if grid.n_lat_bands != raw["n_lat_bands"] or grid.n_lon_bands != raw["n_lon_bands"] or grid.n_cells != raw["n_cells"]:
        raise ValueError(
            f"configs/grid.yaml's stated n_lat_bands/n_lon_bands/n_cells "
            f"({raw['n_lat_bands']}, {raw['n_lon_bands']}, {raw['n_cells']}) "
            f"do not match what resolution_deg={raw['resolution_deg']} actually "
            f"implies ({grid.n_lat_bands}, {grid.n_lon_bands}, {grid.n_cells})."
        )
    return grid


def load_sensitivity_grids(config_path: Path | None = None) -> list[GridDefinition]:
    config_path = config_path or CONFIG_DIR / "grid.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [GridDefinition.from_resolution(r) for r in raw["sensitivity_resolutions_deg"]]


def lat_to_band_index(latitude, grid: GridDefinition) -> np.ndarray:
    """Map latitude in [-90, 90] to a band index in [0, n_lat_bands - 1].
    Band 0 = southernmost row; band n_lat_bands - 1 = northernmost row.
    Exact latitude == 90 (north pole) is clamped into the last band rather
    than overflowing one index beyond it."""
    lat = np.asarray(latitude, dtype=float)
    if np.any(np.isnan(lat)):
        raise ValueError(
            "latitude contains NaN — assign_cells requires QC-accepted data "
            "with valid coordinates (see preprocessing.quality_control)."
        )
    if np.any((lat < -90.0) | (lat > 90.0)):
        raise ValueError("latitude out of [-90, 90] range")
    shifted = lat + 90.0  # now in [0, 180]
    idx = np.floor(shifted / grid.resolution_deg).astype(int)
    idx = np.clip(idx, 0, grid.n_lat_bands - 1)
    return idx


def lon_to_band_index(longitude, grid: GridDefinition) -> np.ndarray:
    """Map longitude to a band index in [0, n_lon_bands - 1] under the
    0-360-degrees-East convention, WITH wraparound (360 maps to the same band
    as 0; negative input is also wrapped)."""
    lon = np.asarray(longitude, dtype=float)
    if np.any(np.isnan(lon)):
        raise ValueError(
            "longitude contains NaN — assign_cells requires QC-accepted data "
            "with valid coordinates (see preprocessing.quality_control)."
        )
    wrapped = np.mod(lon, 360.0)
    idx = np.floor(wrapped / grid.resolution_deg).astype(int)
    idx = np.clip(idx, 0, grid.n_lon_bands - 1)
    return idx


def cell_id(lat_band, lon_band, grid: GridDefinition) -> np.ndarray:
    """Deterministic integer cell id: lat_band * n_lon_bands + lon_band. Unique
    per (lat_band, lon_band) pair for a given grid, stable across calls."""
    return np.asarray(lat_band, dtype=int) * grid.n_lon_bands + np.asarray(lon_band, dtype=int)


def bands_from_cell_id(cell_id_value, grid: GridDefinition) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of cell_id(): recover (lat_band, lon_band) from a cell id.
    Exact inverse of `cell_id = lat_band * n_lon_bands + lon_band`."""
    arr = np.asarray(cell_id_value, dtype=int)
    lat_band = arr // grid.n_lon_bands
    lon_band = arr % grid.n_lon_bands
    return lat_band, lon_band


def assign_cells(df: pd.DataFrame, grid: GridDefinition, *, lat_col: str = "latitude", lon_col: str = "longitude") -> pd.DataFrame:
    """Return a copy of df with `lat_band`, `lon_band`, and `cell_id` columns
    added at the given grid resolution."""
    out = df.copy()
    lat_band = lat_to_band_index(out[lat_col].to_numpy(), grid)
    lon_band = lon_to_band_index(out[lon_col].to_numpy(), grid)
    out["lat_band"] = lat_band
    out["lon_band"] = lon_band
    out["cell_id"] = cell_id(lat_band, lon_band, grid)
    return out


def neighbours(lat_band: int, lon_band: int, grid: GridDefinition) -> list[tuple[int, int]]:
    """Return the (lat_band, lon_band) tuples of all EXISTING eight-neighbours
    of the given cell, under longitude wraparound and NO latitude wraparound.

    - Interior cells (not in the topmost or bottommost latitude band): 8
      neighbours.
    - Cells in the topmost or bottommost latitude band: 5 neighbours (there is
      no row beyond the pole to include — this is a real physical boundary,
      not a wraparound, and is deliberately NOT filled in with a wrapped
      opposite-pole row).
    """
    if not (0 <= lat_band < grid.n_lat_bands):
        raise ValueError(f"lat_band={lat_band} out of range [0, {grid.n_lat_bands})")
    if not (0 <= lon_band < grid.n_lon_bands):
        raise ValueError(f"lon_band={lon_band} out of range [0, {grid.n_lon_bands})")

    lon_left = (lon_band - 1) % grid.n_lon_bands
    lon_right = (lon_band + 1) % grid.n_lon_bands

    lon_candidates_by_row = {
        lat_band - 1: (lon_left, lon_band, lon_right),
        lat_band: (lon_left, lon_right),  # same row: exclude the cell itself
        lat_band + 1: (lon_left, lon_band, lon_right),
    }

    result: list[tuple[int, int]] = []
    for lat_c in (lat_band - 1, lat_band, lat_band + 1):
        if not (0 <= lat_c < grid.n_lat_bands):
            continue  # no row beyond the pole -- a real boundary, not a wrap
        for lon_c in lon_candidates_by_row[lat_c]:
            result.append((lat_c, lon_c))
    return result
