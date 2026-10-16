"""Tests for the sequence-model feasibility report: consecutive valid-sol
run lengths by grid cell, compared against configured window lengths, with
mean/median/percentiles/maximum -- run and reported BEFORE any sequence
model is trained."""
import numpy as np
import pandas as pd

from martian.modeling.sequence_feasibility import (
    assess_window_feasibility,
    compute_consecutive_run_lengths,
    run_feasibility_report,
    summarize_run_lengths,
)


def test_compute_consecutive_run_lengths_hand_computed():
    df = pd.DataFrame(
        {
            "cell_id": [0, 0, 0, 0, 0, 1, 1],
            "mars_year": [1, 1, 1, 1, 1, 1, 1],
            "sol": [0, 1, 2, 5, 6, 0, 1],
        }
    )
    runs = compute_consecutive_run_lengths(df)
    cell0_runs = runs.loc[runs["cell_id"] == 0, "run_length_sols"].tolist()
    cell1_runs = runs.loc[runs["cell_id"] == 1, "run_length_sols"].tolist()
    assert sorted(cell0_runs) == [2, 3]  # sols 0-2 (length 3), sols 5-6 (length 2)
    assert cell1_runs == [2]


def test_compute_consecutive_run_lengths_respects_mars_year_boundary():
    # Same cell, sol=667 of MY1 immediately followed by sol=0 of MY2 --
    # these must NOT be treated as a consecutive run (different mars_year).
    df = pd.DataFrame({"cell_id": [0, 0], "mars_year": [1, 2], "sol": [667, 0]})
    runs = compute_consecutive_run_lengths(df)
    assert len(runs) == 2
    assert set(runs["run_length_sols"]) == {1}


def test_summarize_run_lengths_hand_computed():
    summary = summarize_run_lengths(np.array([1, 2, 3, 4, 5]))
    assert summary["n_runs"] == 5
    assert summary["mean"] == 3.0
    assert summary["median"] == 3.0
    assert summary["max"] == 5.0


def test_summarize_run_lengths_empty_input():
    summary = summarize_run_lengths(np.array([]))
    assert summary["n_runs"] == 0
    assert np.isnan(summary["mean"])


def test_assess_window_feasibility_true_when_a_cell_qualifies():
    run_length_df = pd.DataFrame({"cell_id": [0, 1], "run_length_sols": [10, 2]})
    result = assess_window_feasibility(run_length_df, window_length_sols=5)
    assert result["feasible"] is True
    assert result["n_cells_with_sufficient_run"] == 1
    assert result["n_cells_total"] == 2


def test_assess_window_feasibility_false_when_no_cell_qualifies():
    run_length_df = pd.DataFrame({"cell_id": [0, 1], "run_length_sols": [2, 3]})
    result = assess_window_feasibility(run_length_df, window_length_sols=10)
    assert result["feasible"] is False
    assert result["n_cells_with_sufficient_run"] == 0


def test_run_feasibility_report_covers_all_configured_windows():
    df = pd.DataFrame(
        {
            "cell_id": [0] * 5 + [1] * 2,
            "mars_year": [1] * 7,
            "sol": [0, 1, 2, 3, 4, 0, 1],
        }
    )
    report = run_feasibility_report(df, [3, 5, 10])
    assert report["window_length_sols"].tolist() == [3, 5, 10]
    by_window = report.set_index("window_length_sols")
    assert by_window.loc[3, "feasible"]
    assert by_window.loc[5, "feasible"]
    assert not by_window.loc[10, "feasible"]
    # run-length distribution stats are IDENTICAL across window lengths --
    # they describe the same underlying runs, not the window itself.
    assert by_window.loc[3, "mean"] == by_window.loc[5, "mean"] == by_window.loc[10, "mean"]
