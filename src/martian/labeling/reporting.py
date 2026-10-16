"""Reporting outputs for the labeling pipeline: prevalence tables (by split,
season, storm category, horizon), episode catalogue enrichment, spatial
episode maps, and episode timelines. Every figure/table carries a visible
data-source stamp, reusing the Phase 1 provenance helpers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from data_sources.provenance import stamp_figure, stamp_table_markdown
from data_sources.schema import CONFIG_DIR
from preprocessing.grid import GridDefinition, bands_from_cell_id

_LABEL_COLUMNS = ("dust_anomaly", "atmospheric_anomaly", "storm_episode")
_SEASON_BOUNDARIES = [
    (0.0, 90.0, "Northern Spring (Ls 0-90)"),
    (90.0, 180.0, "Northern Summer (Ls 90-180)"),
    (180.0, 270.0, "Northern Autumn (Ls 180-270)"),
    (270.0, 360.0, "Northern Winter (Ls 270-360)"),
]


def load_splits_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "splits.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def assign_split(
    df: pd.DataFrame,
    splits_config: dict | None = None,
    *,
    mars_year_col: str = "mars_year",
    sol_col: str = "sol",
) -> pd.Series:
    """Reporting-only chronological split tag (train/val/test/unassigned) --
    see configs/splits.yaml's scope note: this is NOT the leak-audited Phase
    4 splitting module, just enough to break down label prevalence by split."""
    splits_config = splits_config or load_splits_config()
    my = df[mars_year_col].to_numpy()
    sol = df[sol_col].to_numpy()

    train_lo, train_hi = splits_config["train_sol_range"]
    val_lo, val_hi = splits_config["val_sol_range"]
    test_lo, test_hi = splits_config["test_sol_range"]

    is_train = (my == splits_config["train_mars_year"]) & (sol >= train_lo) & (sol <= train_hi)
    is_val = (my == splits_config["val_mars_year"]) & (sol >= val_lo) & (sol <= val_hi)
    is_test = (my == splits_config["test_mars_year"]) & (sol >= test_lo) & (sol <= test_hi)

    out = np.full(len(df), "unassigned", dtype=object)
    out[is_train] = "train"
    out[is_val] = "val"
    out[is_test] = "test"
    return pd.Series(out, index=df.index, name="split")


def assign_mars_season(ls: pd.Series | np.ndarray) -> np.ndarray:
    """Standard areocentric-longitude (Ls) quadrant season convention -- a
    definitional convention, not a data claim, so it needs no
    RESEARCHER VERIFICATION REQUIRED tag."""
    ls_arr = np.mod(np.asarray(ls, dtype=float), 360.0)
    out = np.full(len(ls_arr), "unknown", dtype=object)
    for lo, hi, label in _SEASON_BOUNDARIES:
        mask = (ls_arr >= lo) & (ls_arr < hi)
        out[mask] = label
    return out


def _rate_row(group: pd.DataFrame) -> dict:
    n = len(group)
    row = {"n_cell_sols": n}
    for col in _LABEL_COLUMNS:
        row[f"{col}_rate"] = float(group[col].mean()) if n else float("nan")
    for col in group.columns:
        if col.startswith("horizon_"):
            row[f"{col}_rate"] = float(group[col].mean()) if n else float("nan")
    return row


def prevalence_by_split(labeled_df: pd.DataFrame, splits_config: dict | None = None) -> pd.DataFrame:
    split = assign_split(labeled_df, splits_config)
    rows = []
    for split_name, group in labeled_df.groupby(split):
        row = {"split": split_name}
        row.update(_rate_row(group))
        rows.append(row)
    return pd.DataFrame(rows)


def prevalence_by_season(labeled_df: pd.DataFrame, *, ls_col: str = "solar_longitude_ls") -> pd.DataFrame:
    season = assign_mars_season(labeled_df[ls_col])
    rows = []
    for season_name, group in labeled_df.groupby(season):
        row = {"season": season_name}
        row.update(_rate_row(group))
        rows.append(row)
    return pd.DataFrame(rows)


def prevalence_by_horizon(labeled_df: pd.DataFrame) -> pd.DataFrame:
    horizon_cols = [c for c in labeled_df.columns if c.startswith("horizon_")]
    rows = []
    for col in horizon_cols:
        rows.append(
            {
                "horizon": col,
                "n_cell_sols": len(labeled_df),
                "n_positive": int(labeled_df[col].sum()),
                "prevalence": float(labeled_df[col].mean()) if len(labeled_df) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def prevalence_by_storm_category(episode_catalogue_with_taxonomy: pd.DataFrame) -> pd.DataFrame:
    if episode_catalogue_with_taxonomy.empty:
        return pd.DataFrame(columns=["taxonomy_class", "n_episodes", "n_qualifying_episodes", "mean_duration_sols", "mean_peak_area_cells"])
    rows = []
    for cls, group in episode_catalogue_with_taxonomy.groupby("taxonomy_class"):
        rows.append(
            {
                "taxonomy_class": cls,
                "n_episodes": len(group),
                "n_qualifying_episodes": int(group["qualifies"].sum()) if "qualifies" in group.columns else len(group),
                "mean_duration_sols": float(group["duration_sols"].mean()),
                "mean_peak_area_cells": float(group["peak_area_cells"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_prevalence_tables(
    labeled_df: pd.DataFrame,
    episode_catalogue_with_taxonomy: pd.DataFrame,
    *,
    splits_config: dict | None = None,
) -> dict[str, pd.DataFrame]:
    return {
        "prevalence_by_split": prevalence_by_split(labeled_df, splits_config),
        "prevalence_by_season": prevalence_by_season(labeled_df),
        "prevalence_by_horizon": prevalence_by_horizon(labeled_df),
        "prevalence_by_storm_category": prevalence_by_storm_category(episode_catalogue_with_taxonomy),
    }


def episode_map(labeled_df: pd.DataFrame, grid: GridDefinition, data_source_label: str):
    """Spatial heatmap of storm_episode cell-sol counts per grid cell.
    `lat_band`/`lon_band` are recovered from `cell_id` (aggregate_by_cell_sol's
    output does not carry the band-index columns through)."""
    counts = np.zeros((grid.n_lat_bands, grid.n_lon_bands), dtype=int)
    storm_rows = labeled_df.loc[labeled_df["storm_episode"]]
    if len(storm_rows):
        lat_band, lon_band = bands_from_cell_id(storm_rows["cell_id"].to_numpy(), grid)
        grouped = pd.DataFrame({"lat_band": lat_band, "lon_band": lon_band}).groupby(["lat_band", "lon_band"]).size()
        for (lb, lo), n in grouped.items():
            counts[int(lb), int(lo)] = n

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(counts, origin="lower", aspect="auto", cmap="inferno")
    ax.set_xlabel("Longitude band (0-360°E)")
    ax.set_ylabel("Latitude band (-90 to 90°)")
    ax.set_title("Storm-episode cell-sol count per cell")
    fig.colorbar(im, ax=ax, label="storm_episode cell-sol count")
    stamp_figure(fig, data_source_label)
    return fig


def _pack_rows(intervals: list[tuple[int, int]]) -> tuple[list[int], int]:
    """Greedy interval-scheduling row-packing: episodes that don't overlap
    in sol share a row; only genuinely time-overlapping episodes need
    separate rows. Keeps the figure's row count (and therefore its size)
    proportional to actual simultaneous-episode density, not to the total
    episode count -- with thousands of mostly single-sol, non-overlapping
    episodes (typical under the sparse simulated sampling), this stays
    small even when the catalogue itself is large."""
    row_ends: list[int] = []
    assignments: list[int] = []
    for onset, end in intervals:
        placed = False
        for r in range(len(row_ends)):
            if row_ends[r] < onset:
                row_ends[r] = end
                assignments.append(r)
                placed = True
                break
        if not placed:
            row_ends.append(end)
            assignments.append(len(row_ends) - 1)
    return assignments, len(row_ends)


def episode_timeline(episode_catalogue_with_taxonomy: pd.DataFrame, data_source_label: str):
    """Horizontal timeline: one bar per QUALIFYING episode spanning onset_sol
    to end_sol, coloured by taxonomy_class. Row-packed (see `_pack_rows`) so
    figure size stays bounded regardless of how many episodes exist."""
    qualifying = episode_catalogue_with_taxonomy
    if "qualifies" in qualifying.columns:
        qualifying = qualifying.loc[qualifying["qualifies"]]
    qualifying = qualifying.sort_values("onset_sol").reset_index(drop=True)

    color_map = {"local": "tab:blue", "regional": "tab:orange", "planet_encircling": "tab:red", "unclassified": "tab:gray"}

    intervals = list(zip(qualifying["onset_sol"].tolist(), qualifying["end_sol"].tolist()))
    row_assignments, n_rows = _pack_rows(intervals)

    fig_height = min(max(3.0, 0.3 * n_rows), 12.0)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    for (i, row), row_idx in zip(qualifying.iterrows(), row_assignments):
        color = color_map.get(row.get("taxonomy_class", "unclassified"), "tab:gray")
        ax.barh(row_idx, row["end_sol"] - row["onset_sol"] + 1, left=row["onset_sol"], color=color, height=0.8)

    ax.set_xlabel("Sol")
    ax.set_ylabel(f"Row (packed; {len(qualifying)} episodes in {n_rows} rows)")
    ax.set_title(f"Storm-episode timeline ({len(qualifying)} qualifying episodes)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    ax.legend(handles, color_map.keys(), loc="upper right", fontsize=7)
    stamp_figure(fig, data_source_label)
    return fig
