"""Tests for shared figure-provenance stamping: stable figure numbering,
and the PNG/CSV/provenance-JSON triple written per figure."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from martian.evaluation.figure_provenance import (
    FigureProvenance,
    assign_figure_number,
    save_figure_with_provenance,
)


def test_assign_figure_number_stable_across_calls(tmp_path):
    registry_path = tmp_path / "registry.json"
    n1 = assign_figure_number("task_a_model_x_h1", registry_path=registry_path)
    n2 = assign_figure_number("task_a_model_x_h1", registry_path=registry_path)
    assert n1 == n2


def test_assign_figure_number_increments_for_new_keys(tmp_path):
    registry_path = tmp_path / "registry.json"
    n1 = assign_figure_number("figure_one", registry_path=registry_path)
    n2 = assign_figure_number("figure_two", registry_path=registry_path)
    assert n1 != n2
    major1, minor1 = n1.split(".")
    major2, minor2 = n2.split(".")
    assert major1 == major2 == "4"
    assert int(minor2) == int(minor1) + 1


def test_assign_figure_number_continues_from_existing_registry(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"existing_key": "4.7"}), encoding="utf-8")
    n = assign_figure_number("new_key", registry_path=registry_path)
    assert n == "4.8"


def test_save_figure_with_provenance_writes_all_three_files(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    provenance = FigureProvenance(
        figure_number="4.9", caption="Test figure", units="dimensionless", n=3,
        split="test", horizon_sols=1, data_source="SIMULATED (test)",
        script="tests/test_figure_provenance.py", config_paths=["configs/evaluation.yaml"],
    )
    source_df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})

    paths = save_figure_with_provenance(fig, tmp_path / "myfigure", provenance, source_df=source_df)

    assert paths["png"].exists()
    assert paths["csv"].exists()
    assert paths["provenance"].exists()

    loaded = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert loaded["figure_number"] == "4.9"
    assert loaded["caption"] == "Test figure"
    assert loaded["n"] == 3
    assert loaded["split"] == "test"
    assert loaded["horizon_sols"] == 1
    assert loaded["source_csv"] == str(paths["csv"])

    loaded_csv = pd.read_csv(paths["csv"])
    assert loaded_csv["x"].tolist() == [1, 2, 3]


def test_save_figure_with_provenance_closes_the_figure(tmp_path):
    n_open_before = len(plt.get_fignums())
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    provenance = FigureProvenance(
        figure_number="4.10", caption="c", units="u", n=2, split="test", horizon_sols=1,
        data_source="SIMULATED (test)", script="s", config_paths=[],
    )
    save_figure_with_provenance(fig, tmp_path / "fig2", provenance)
    assert len(plt.get_fignums()) == n_open_before  # figure must be closed, not leaked


def test_save_figure_without_source_df_skips_csv(tmp_path):
    fig, ax = plt.subplots()
    provenance = FigureProvenance(
        figure_number="4.11", caption="c", units="u", n=0, split="test", horizon_sols=1,
        data_source="SIMULATED (test)", script="s", config_paths=[],
    )
    paths = save_figure_with_provenance(fig, tmp_path / "fig3", provenance)
    assert paths["csv"] is None
    assert not (tmp_path / "fig3.csv").exists()
