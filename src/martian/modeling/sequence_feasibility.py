"""Sequence-model feasibility assessment (run BEFORE any LSTM/GRU training).

Computes the length of every consecutive-valid-sol run, PER GRID CELL, in
the trailing-filled cell-sol series (the same series feature engineering is
built from -- see preprocessing/features.py) and compares those run lengths
against the configured sequence input-window length. Reports mean, median,
percentiles, and maximum, plus how many cells have at least one run long
enough to supply a single window -- exactly the three steps this task
requires, always run and always reported, never skipped even when the
result is negative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_consecutive_run_lengths(
    df: pd.DataFrame,
    *,
    cell_col: str = "cell_id",
    mars_year_col: str = "mars_year",
    sol_col: str = "sol",
) -> pd.DataFrame:
    """One row per consecutive-sol run within a (cell_id, mars_year) group:
    cell_id, mars_year, start_sol, end_sol, run_length_sols."""
    rows = []
    for (cid, my), group in df.groupby([cell_col, mars_year_col], sort=False):
        sols = np.sort(group[sol_col].to_numpy(dtype=int))
        if len(sols) == 0:
            continue
        run_start = sols[0]
        prev = sols[0]
        for s in sols[1:]:
            if s == prev + 1:
                prev = s
                continue
            rows.append({"cell_id": cid, "mars_year": my, "start_sol": run_start, "end_sol": prev, "run_length_sols": prev - run_start + 1})
            run_start = s
            prev = s
        rows.append({"cell_id": cid, "mars_year": my, "start_sol": run_start, "end_sol": prev, "run_length_sols": prev - run_start + 1})
    return pd.DataFrame(rows, columns=["cell_id", "mars_year", "start_sol", "end_sol", "run_length_sols"])


def summarize_run_lengths(run_lengths: np.ndarray) -> dict:
    if len(run_lengths) == 0:
        return {
            "n_runs": 0, "mean": float("nan"), "median": float("nan"),
            "p25": float("nan"), "p75": float("nan"), "p90": float("nan"), "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "n_runs": int(len(run_lengths)),
        "mean": float(np.mean(run_lengths)),
        "median": float(np.median(run_lengths)),
        "p25": float(np.percentile(run_lengths, 25)),
        "p75": float(np.percentile(run_lengths, 75)),
        "p90": float(np.percentile(run_lengths, 90)),
        "p95": float(np.percentile(run_lengths, 95)),
        "max": float(np.max(run_lengths)),
    }


def assess_window_feasibility(
    run_length_df: pd.DataFrame,
    window_length_sols: int,
    *,
    cell_col: str = "cell_id",
) -> dict:
    """Compares the run-length distribution against one candidate window
    length. `feasible` is True iff at least one cell has at least one run
    long enough to supply a full window."""
    summary = summarize_run_lengths(run_length_df["run_length_sols"].to_numpy())
    if len(run_length_df):
        max_run_per_cell = run_length_df.groupby(cell_col)["run_length_sols"].max()
        n_cells_with_sufficient_run = int((max_run_per_cell >= window_length_sols).sum())
        n_cells_total = int(run_length_df[cell_col].nunique())
    else:
        n_cells_with_sufficient_run = 0
        n_cells_total = 0

    return {
        "window_length_sols": window_length_sols,
        "feasible": n_cells_with_sufficient_run > 0,
        "n_cells_with_sufficient_run": n_cells_with_sufficient_run,
        "n_cells_total": n_cells_total,
        **summary,
    }


def run_feasibility_report(
    filled_df: pd.DataFrame,
    window_lengths_sols: list[int],
    *,
    cell_col: str = "cell_id",
    mars_year_col: str = "mars_year",
    sol_col: str = "sol",
) -> pd.DataFrame:
    """The full report this task requires: one row per candidate window
    length (e.g. the primary window plus every sensitivity window), each
    with the SAME underlying run-length distribution (mean/median/
    percentiles/max never depend on the window length) plus that window's
    own feasibility verdict."""
    run_length_df = compute_consecutive_run_lengths(
        filled_df, cell_col=cell_col, mars_year_col=mars_year_col, sol_col=sol_col
    )
    rows = [assess_window_feasibility(run_length_df, w, cell_col=cell_col) for w in window_lengths_sols]
    return pd.DataFrame(rows)
