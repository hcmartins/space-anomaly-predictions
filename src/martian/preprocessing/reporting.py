"""Reporting outputs for the Chapter 4 preprocessing pipeline: QC summaries,
spatial coverage maps, cell-sol count distributions, seasonal and local-time
coverage, and immutable QC logs. Every figure/table/report carries a visible
data-source stamp (task requirement 6), reusing the Phase 1 provenance
helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_sources.manifest import new_run_id, write_immutable_manifest
from data_sources.provenance import stamp_figure, stamp_report, stamp_table_markdown

from .causal_filters import (
    CAUSAL_COLUMN_SUFFIX,
    CENTERED_COLUMN_SUFFIX,
    apply_causal_savgol,
    apply_centered_savgol_for_descriptive_comparison_only,
    load_causal_filter_config,
)
from .grid import GridDefinition
from .quality_control import QCReport


def qc_summary_report(qc_report: QCReport, data_source_label: str) -> str:
    """Markdown report of QC rule-by-rule flag/rejection counts plus the
    overall accept/reject summary."""
    summary = qc_report.summary()
    rule_df = pd.DataFrame(
        [
            {
                "id": r.id,
                "rule": r.name,
                "enabled": r.enabled,
                "source": r.source,
                "n_flagged": r.n_flagged,
                "n_rejected_here": r.n_rejected_here,
            }
            for r in qc_report.rule_results
        ]
    )
    table_md = stamp_table_markdown(rule_df, data_source_label, title="QC rule results (priority order)")
    body = (
        f"## Quality-control summary\n\n"
        f"- Input retrievals: {summary['n_input']}\n"
        f"- Accepted: {summary['n_accepted']}\n"
        f"- Rejected: {summary['n_rejected']}\n\n"
        f"{table_md}\n"
    )
    return stamp_report(body, data_source_label)


def spatial_coverage_map(gridded_accepted_df: pd.DataFrame, grid: GridDefinition, data_source_label: str):
    """Heatmap of accepted-observation count per (lat_band, lon_band) cell.
    Requires `lat_band`/`lon_band` columns (see preprocessing.grid.assign_cells)."""
    counts = np.zeros((grid.n_lat_bands, grid.n_lon_bands), dtype=int)
    grouped = gridded_accepted_df.groupby(["lat_band", "lon_band"]).size()
    for (lat_band, lon_band), n in grouped.items():
        counts[int(lat_band), int(lon_band)] = n

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(counts, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("Longitude band (0-360°E)")
    ax.set_ylabel("Latitude band (-90 to 90°)")
    ax.set_title(f"Spatial coverage: accepted retrieval count per cell ({grid.resolution_deg}° grid)")
    fig.colorbar(im, ax=ax, label="Accepted retrieval count")
    stamp_figure(fig, data_source_label)
    return fig


def cell_sol_count_distribution(aggregated_df: pd.DataFrame, data_source_label: str):
    """Histogram of observation_count across all populated cell-sols."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(aggregated_df["observation_count"], bins=30, color="steelblue", edgecolor="black")
    ax.set_xlabel("Observations per cell-sol")
    ax.set_ylabel("Number of cell-sols")
    ax.set_title("Cell-sol observation-count distribution")
    stamp_figure(fig, data_source_label)
    return fig


def seasonal_coverage(accepted_df: pd.DataFrame, data_source_label: str, *, ls_bin_width_deg: float = 10.0):
    """Coverage of accepted retrievals binned by solar longitude (Ls),
    returning (figure, table)."""
    bins = np.arange(0.0, 360.0 + ls_bin_width_deg, ls_bin_width_deg)
    ls = accepted_df["solar_longitude_ls"].to_numpy(dtype=float)
    counts, edges = np.histogram(ls, bins=bins)
    table = pd.DataFrame(
        {
            "ls_bin_start_deg": edges[:-1],
            "ls_bin_end_deg": edges[1:],
            "n_accepted_retrievals": counts,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(table["ls_bin_start_deg"], table["n_accepted_retrievals"], width=ls_bin_width_deg * 0.9, align="edge", color="darkorange")
    ax.set_xlabel("Solar longitude Ls (degrees)")
    ax.set_ylabel("Accepted retrieval count")
    ax.set_title("Seasonal coverage")
    stamp_figure(fig, data_source_label)
    return fig, table


def local_time_coverage(accepted_df: pd.DataFrame, data_source_label: str, *, ltst_bin_width_hours: float = 1.0):
    """Coverage of accepted retrievals binned by local true solar time,
    returning (figure, table)."""
    bins = np.arange(0.0, 24.0 + ltst_bin_width_hours, ltst_bin_width_hours)
    ltst = accepted_df["local_true_solar_time"].to_numpy(dtype=float)
    counts, edges = np.histogram(ltst, bins=bins)
    table = pd.DataFrame(
        {
            "ltst_bin_start_hr": edges[:-1],
            "ltst_bin_end_hr": edges[1:],
            "n_accepted_retrievals": counts,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(table["ltst_bin_start_hr"], table["n_accepted_retrievals"], width=ltst_bin_width_hours * 0.9, align="edge", color="seagreen")
    ax.set_xlabel("Local true solar time (hr)")
    ax.set_ylabel("Accepted retrieval count")
    ax.set_title("Local-time coverage")
    stamp_figure(fig, data_source_label)
    return fig, table


def write_qc_log(
    manifest_dir: Path,
    qc_report: QCReport,
    *,
    data_source_label: str,
    extra: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> Path:
    """Write a machine-readable, immutable QC log (task requirement:
    'machine-readable QC logs') using the Phase 1 immutable-manifest
    mechanism."""
    run_id = run_id or new_run_id(prefix="qc")
    payload = {
        "data_source": data_source_label,
        "qc_config": qc_report.config,
        "summary": qc_report.summary(),
        "rule_results": [
            {
                "id": r.id,
                "name": r.name,
                "enabled": r.enabled,
                "source": r.source,
                "n_flagged": r.n_flagged,
                "n_rejected_here": r.n_rejected_here,
            }
            for r in qc_report.rule_results
        ],
    }
    if extra:
        payload["extra"] = extra
    return write_immutable_manifest(manifest_dir, run_id, payload)


def figure_4_3_causal_vs_centered_filtering(
    df: pd.DataFrame,
    *,
    cell_id: int,
    mars_year: int,
    data_source_label: str,
    causal_config: dict | None = None,
    sol_col: str = "sol",
    value_col: str | None = None,
):
    """Figure 4.3: irregular observation sampling, cell-sol alignment, and
    causal-vs-centred Savitzky-Golay filtering for a single (cell_id,
    mars_year) series. The centred line is drawn ONLY for this descriptive
    comparison -- it is produced by
    apply_centered_savgol_for_descriptive_comparison_only and must never be
    reused as a model feature (see preprocessing/causal_filters.py's
    assert_feature_safe guard)."""
    causal_config = causal_config or load_causal_filter_config()
    value_col = value_col or causal_config["variable"]
    window_length = causal_config["window_length"]
    polyorder = causal_config["polyorder"]

    cell_df = df[(df["cell_id"] == cell_id) & (df["mars_year"] == mars_year)].sort_values(sol_col)
    if cell_df.empty:
        raise ValueError(f"No rows found for cell_id={cell_id}, mars_year={mars_year}.")

    causal_out = apply_causal_savgol(cell_df, value_col=value_col, config=causal_config)
    centered_out = apply_centered_savgol_for_descriptive_comparison_only(
        cell_df, value_col=value_col, config=causal_config
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    # Cell-sol alignment: a light gridline at every integer sol in range,
    # so gaps in the scatter below are visually unambiguous.
    sol_min, sol_max = int(cell_df[sol_col].min()), int(cell_df[sol_col].max())
    for s in range(sol_min, sol_max + 1):
        ax.axvline(s, color="lightgray", linewidth=0.4, zorder=0)

    if "is_observed" in cell_df.columns:
        observed_mask = cell_df["is_observed"].astype(bool)
    else:
        observed_mask = pd.Series(True, index=cell_df.index)

    # Irregular observation sampling.
    ax.scatter(
        cell_df.loc[observed_mask, sol_col], cell_df.loc[observed_mask, value_col],
        color="black", marker="o", s=32, label="Observed retrieval (irregular sampling)", zorder=5,
    )
    if (~observed_mask).any():
        ax.scatter(
            cell_df.loc[~observed_mask, sol_col], cell_df.loc[~observed_mask, value_col],
            color="gray", marker="x", s=40, label="Trailing-filled cell-sol", zorder=5,
        )

    ax.plot(
        causal_out[sol_col], causal_out[f"{value_col}{CAUSAL_COLUMN_SUFFIX}"],
        color="tab:blue", linewidth=2, label="Causal (trailing) Savitzky-Golay",
    )
    ax.plot(
        centered_out[sol_col], centered_out[f"{value_col}{CENTERED_COLUMN_SUFFIX}"],
        color="tab:orange", linewidth=2, linestyle="--",
        label="Centred Savitzky-Golay (descriptive comparison only -- never a model feature)",
    )

    ax.set_xlabel("Sol")
    ax.set_ylabel(value_col.replace("_", " "))
    ax.set_title(f"Figure 4.3: Causal vs. centred filtering (cell_id={cell_id}, MY{mars_year})")
    ax.legend(loc="best", fontsize=8)

    params_text = (
        f"variable = {value_col}\n"
        f"window_length = {window_length}\n"
        f"polyorder = {polyorder}\n"
        f"causal: trailing coefficient position (pos={window_length - 1})\n"
        f"centred: mid-window position (pos={window_length // 2}), descriptive only"
    )
    ax.text(
        0.01, 0.98, params_text, transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    stamp_figure(fig, data_source_label)
    return fig
