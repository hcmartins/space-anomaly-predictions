"""Blind dev/test tuning of the Table 4.6 catalogue-matching tolerance
(temporal_tolerance_sols, spatial_region_radius_cells) -- CHAPTER_4_
PARAMETER_VALUES.md open item 4 (2026-08-12).

`configs/catalogue_validation.yaml`'s +/-2-sol / 1-cell tolerance was a
carried-over placeholder, never independently confirmed against the Guha et
al. (2024) catalogue's own registration accuracy. Tuning it directly against
the same 98 documented events Table 4.6 reports results on would be
circular -- the reported POD/precision would then measure how well the
tolerance was fitted to this one catalogue, not genuine detection skill.

This module avoids that by splitting the 98 documented MY36 events into a
DEV half (tolerance search only) and a TEST half (held out, touched only
once, after the tolerance is frozen). The dev-side grid search and the
test-side held-out result are both returned in full, so the whole procedure
is auditable end to end -- see `tune_and_validate_tolerance`.

This module never modifies `labeling/catalogue_validation.py` or the
original Table 4.6: it is a strictly additive analysis, built on top of
`run_catalogue_validation`, producing a second table (4.6b) alongside it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_sources.schema import CONFIG_DIR
from preprocessing.grid import GridDefinition

from .catalogue_validation import (
    UNMATCHED_EPISODE_LABEL_DEFAULT,
    build_table_4_6,
    run_catalogue_validation,
)


def load_catalogue_tolerance_tuning_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "catalogue_tolerance_tuning.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def split_documented_events_dev_test(
    documented_events: pd.DataFrame, *, dev_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed-seed RANDOM split of documented events into dev/test, by row
    (one row per documented event) -- not chronological, not storm-number
    parity, to avoid any systematic bias by season or detection order.
    `covers_mars_year` (a pandas `.attrs` value `run_catalogue_validation`
    reads to restrict scope) is explicitly re-set on both halves, since
    `.attrs` propagation through `.iloc` indexing is not guaranteed across
    pandas versions."""
    covered_year = documented_events.attrs.get("covers_mars_year")
    n = len(documented_events)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    n_dev = int(round(n * dev_fraction))
    dev_idx = shuffled[:n_dev]
    test_idx = shuffled[n_dev:]

    dev = documented_events.iloc[dev_idx].reset_index(drop=True)
    test = documented_events.iloc[test_idx].reset_index(drop=True)
    if covered_year is not None:
        dev.attrs["covers_mars_year"] = covered_year
        test.attrs["covers_mars_year"] = covered_year
    return dev, test


def _f1(pod: float, precision: float) -> float:
    """Harmonic mean of POD and episode precision; 0.0 (not NaN) whenever
    either is undefined or both are zero -- a tolerance combination that
    detects nothing genuinely scores worst, it does not silently drop out
    of the ranking."""
    if pod is None or precision is None:
        return 0.0
    if np.isnan(pod) or np.isnan(precision) or (pod + precision) == 0.0:
        return 0.0
    return float(2.0 * pod * precision / (pod + precision))


def search_tolerance_grid(
    labeled_df: pd.DataFrame,
    episode_catalogue: pd.DataFrame,
    dev_events: pd.DataFrame,
    grid: GridDefinition,
    *,
    temporal_candidates: list[int],
    spatial_candidates: list[int],
    unmatched_episode_label: str = UNMATCHED_EPISODE_LABEL_DEFAULT,
) -> pd.DataFrame:
    """Evaluates every (temporal_tolerance_sols, spatial_region_radius_cells)
    combination in the grid against the DEV documented events ONLY (`test`
    is never touched here), scoring each by F1 of episode detection. Returns
    every row of the search, not just the winner, so the full grid is
    independently auditable rather than only the selected result."""
    rows: list[dict] = []
    for t in temporal_candidates:
        for s in spatial_candidates:
            config = {
                "temporal_tolerance_sols": t,
                "spatial_region_radius_cells": s,
                "unmatched_episode_label": unmatched_episode_label,
            }
            result = run_catalogue_validation(labeled_df, episode_catalogue, dev_events, grid, config=config)
            summary = result["summary"]
            f1 = _f1(summary["pod"], summary["episode_precision"])
            rows.append(
                {
                    "temporal_tolerance_sols": t,
                    "spatial_region_radius_cells": s,
                    "dev_pod": summary["pod"],
                    "dev_episode_precision": summary["episode_precision"],
                    "dev_f1": f1,
                    "dev_n_hits": summary["n_hits"],
                    "dev_n_documented_events": summary["n_documented_events"],
                    "dev_n_matched_episodes": summary["n_matched_episodes"],
                    "dev_n_qualifying_episodes": summary["n_qualifying_episodes"],
                }
            )
    return pd.DataFrame(rows)


def select_best_tolerance(search_table: pd.DataFrame) -> dict:
    """Picks the grid row with the highest `dev_f1`; ties broken by
    preferring the SMALLEST temporal tolerance, then the smallest spatial
    radius -- i.e. the tightest, most conservative tolerance among
    equally-good candidates. An explicit, documented tie-break rule, not an
    artefact of grid iteration order."""
    if search_table.empty:
        raise ValueError("search_table is empty -- cannot select a tolerance from zero candidates")
    ordered = search_table.sort_values(
        by=["dev_f1", "temporal_tolerance_sols", "spatial_region_radius_cells"],
        ascending=[False, True, True],
    )
    best = ordered.iloc[0]
    return {
        "temporal_tolerance_sols": int(best["temporal_tolerance_sols"]),
        "spatial_region_radius_cells": int(best["spatial_region_radius_cells"]),
        "dev_f1": float(best["dev_f1"]),
        "dev_pod": float(best["dev_pod"]),
        "dev_episode_precision": float(best["dev_episode_precision"]),
    }


def tune_and_validate_tolerance(
    labeled_df: pd.DataFrame,
    episode_catalogue: pd.DataFrame,
    documented_events: pd.DataFrame,
    grid: GridDefinition,
    *,
    config: dict | None = None,
    unmatched_episode_label: str = UNMATCHED_EPISODE_LABEL_DEFAULT,
) -> dict:
    """End-to-end blind tuning: split documented events into dev/test
    (`split_documented_events_dev_test`), grid-search the matching
    tolerance on DEV ONLY (`search_tolerance_grid` -- `test_events` is not
    passed to it and is not touched until the final step below), freeze the
    dev-selected tolerance (`select_best_tolerance`), then evaluate
    Table-4.6-style hit/miss/POD/precision on the held-out TEST events under
    that single frozen tolerance. `result["test_validation"]` is the
    honest, non-circular estimate this analysis exists to produce;
    `result["dev_search_table"]` is retained in full for audit."""
    config = config or load_catalogue_tolerance_tuning_config()

    split_cfg = config["dev_test_split"]
    dev_events, test_events = split_documented_events_dev_test(
        documented_events, dev_fraction=split_cfg["dev_fraction"], seed=split_cfg["seed"]
    )

    grid_cfg = config["tolerance_grid"]
    search_table = search_tolerance_grid(
        labeled_df,
        episode_catalogue,
        dev_events,
        grid,
        temporal_candidates=grid_cfg["temporal_tolerance_sols"],
        spatial_candidates=grid_cfg["spatial_region_radius_cells"],
        unmatched_episode_label=unmatched_episode_label,
    )
    selected = select_best_tolerance(search_table)

    test_config = {
        "temporal_tolerance_sols": selected["temporal_tolerance_sols"],
        "spatial_region_radius_cells": selected["spatial_region_radius_cells"],
        "unmatched_episode_label": unmatched_episode_label,
    }
    test_result = run_catalogue_validation(labeled_df, episode_catalogue, test_events, grid, config=test_config)

    return {
        "selection_metric": config.get("selection_metric", "f1"),
        "split": {
            "dev_fraction": split_cfg["dev_fraction"],
            "seed": split_cfg["seed"],
            "n_dev_events": len(dev_events),
            "n_test_events": len(test_events),
        },
        "dev_search_table": search_table,
        "selected_tolerance": selected,
        "test_validation": test_result,
    }


def build_table_4_6b(result: dict) -> pd.DataFrame:
    """Table 4.6b: the blind, held-out-test-side validation under the
    dev-selected tolerance -- same columns as Table 4.6
    (`catalogue_validation.build_table_4_6`) plus the selected tolerance and
    split provenance, so it can be read on its own without cross-referencing
    the dev search table."""
    table = build_table_4_6(result["test_validation"])
    table["temporal_tolerance_sols"] = result["selected_tolerance"]["temporal_tolerance_sols"]
    table["spatial_region_radius_cells"] = result["selected_tolerance"]["spatial_region_radius_cells"]
    table["selection_metric"] = result["selection_metric"]
    table["dev_f1_at_selected_tolerance"] = result["selected_tolerance"]["dev_f1"]
    table["dev_fraction"] = result["split"]["dev_fraction"]
    table["split_seed"] = result["split"]["seed"]
    table["n_dev_events"] = result["split"]["n_dev_events"]
    table["n_test_events"] = result["split"]["n_test_events"]
    return table


def write_table_4_6b(result: dict, out_dir: Path, *, data_source_label: str) -> tuple[Path, Path, Path]:
    """Writes Table 4.6b (csv+md) plus the full dev-side grid-search table
    (csv only -- an audit artefact, not itself a thesis table) so the
    tolerance selection can be independently re-checked rather than taken on
    trust."""
    from data_sources.provenance import stamp_table_markdown

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_table_4_6b(result)
    csv_path = out_dir / "table_4_6b_blind_tolerance_validation.csv"
    md_path = out_dir / "table_4_6b_blind_tolerance_validation.md"
    search_path = out_dir / "table_4_6b_dev_tolerance_search.csv"
    table.to_csv(csv_path, index=False)
    md_path.write_text(
        stamp_table_markdown(
            table,
            data_source_label,
            title="Table 4.6b -- Blind Dev/Test Validation of the Catalogue-Matching Tolerance",
        ),
        encoding="utf-8",
    )
    result["dev_search_table"].to_csv(search_path, index=False)
    return csv_path, md_path, search_path
