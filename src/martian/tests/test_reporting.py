"""Tests for reporting outputs: QC summary content, figure data-source
stamping, seasonal/local-time table structure, and immutable QC-log writing."""
import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest

from martian.data_sources.manifest import read_manifest
from martian.preprocessing.aggregation import aggregate_by_cell_sol, load_aggregation_config
from martian.preprocessing.grid import assign_cells, load_primary_grid
from martian.preprocessing.quality_control import load_qc_config, run_quality_control
from martian.preprocessing.reporting import (
    cell_sol_count_distribution,
    local_time_coverage,
    qc_summary_report,
    seasonal_coverage,
    spatial_coverage_map,
    write_qc_log,
)

DATA_SOURCE_LABEL = "SIMULATED (test fixture)"


def _retrieval_row(**overrides) -> dict:
    row = {
        "latitude": 10.0,
        "longitude": 100.0,
        "utc_timestamp": pd.Timestamp("2021-02-09", tz="UTC"),
        "mars_year": 1,
        "sol": 1,
        "solar_longitude_ls": 45.0,
        "local_true_solar_time": 14.0,
        "dust_optical_depth": 0.2,
        "dust_optical_depth_uncertainty": 0.02,
        "water_ice_optical_depth": 0.1,
        "water_ice_optical_depth_uncertainty": 0.01,
        "surface_temperature_k": 220.0,
        "surface_temperature_k_uncertainty": 5.0,
        "surface_pressure_mbar": 6.1,
        "surface_emissivity": 0.9,
        "emission_angle_deg": 10.0,
        "quality_flag": 0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def qc_report():
    df = pd.DataFrame(
        [
            _retrieval_row(longitude=lon, sol=sol, solar_longitude_ls=ls, local_true_solar_time=ltst)
            for lon, sol, ls, ltst in [
                (10.0, 1, 30.0, 6.0),
                (20.0, 1, 30.0, 8.0),
                (200.0, 2, 200.0, 20.0),
            ]
        ]
        + [_retrieval_row(quality_flag=1, longitude=300.0)]  # deliberately rejected row
    )
    return run_quality_control(df, load_qc_config())


@pytest.fixture
def aggregated_and_gridded(qc_report):
    agg_config = load_aggregation_config()
    grid = load_primary_grid()
    accepted = qc_report.accepted
    gridded = assign_cells(accepted, grid)
    aggregated = aggregate_by_cell_sol(
        gridded,
        min_obs_per_cell_sol=agg_config["primary"]["min_obs_per_cell_sol"],
        min_pressure_level_coverage_frac=agg_config["primary"]["min_pressure_level_coverage_frac"],
        config=agg_config,
    )
    return grid, accepted, gridded, aggregated


def test_qc_summary_report_contains_counts_and_data_source(qc_report):
    report_text = qc_summary_report(qc_report, DATA_SOURCE_LABEL)
    summary = qc_report.summary()
    assert DATA_SOURCE_LABEL in report_text
    assert str(summary["n_accepted"]) in report_text
    assert str(summary["n_rejected"]) in report_text
    assert "missing_required_variable" in report_text


def test_spatial_coverage_map_is_stamped(aggregated_and_gridded):
    grid, accepted, gridded, aggregated = aggregated_and_gridded
    fig = spatial_coverage_map(gridded, grid, DATA_SOURCE_LABEL)
    stamped_texts = [t.get_text() for t in fig.texts]
    assert any(DATA_SOURCE_LABEL in t for t in stamped_texts)


def test_cell_sol_count_distribution_is_stamped(aggregated_and_gridded):
    _, _, _, aggregated = aggregated_and_gridded
    fig = cell_sol_count_distribution(aggregated, DATA_SOURCE_LABEL)
    stamped_texts = [t.get_text() for t in fig.texts]
    assert any(DATA_SOURCE_LABEL in t for t in stamped_texts)


def test_seasonal_coverage_table_counts_sum_to_input_length(aggregated_and_gridded):
    _, accepted, _, _ = aggregated_and_gridded
    fig, table = seasonal_coverage(accepted, DATA_SOURCE_LABEL, ls_bin_width_deg=10.0)
    assert table["n_accepted_retrievals"].sum() == len(accepted)
    assert list(table.columns) == ["ls_bin_start_deg", "ls_bin_end_deg", "n_accepted_retrievals"]
    stamped_texts = [t.get_text() for t in fig.texts]
    assert any(DATA_SOURCE_LABEL in t for t in stamped_texts)


def test_local_time_coverage_table_counts_sum_to_input_length(aggregated_and_gridded):
    _, accepted, _, _ = aggregated_and_gridded
    fig, table = local_time_coverage(accepted, DATA_SOURCE_LABEL, ltst_bin_width_hours=1.0)
    assert table["n_accepted_retrievals"].sum() == len(accepted)
    assert list(table.columns) == ["ltst_bin_start_hr", "ltst_bin_end_hr", "n_accepted_retrievals"]


def test_write_qc_log_writes_immutable_manifest(tmp_path, qc_report):
    path = write_qc_log(tmp_path, qc_report, data_source_label=DATA_SOURCE_LABEL, extra={"seed": 42})
    assert path.exists()
    payload = read_manifest(path)
    assert payload["data_source"] == DATA_SOURCE_LABEL
    assert payload["summary"]["n_input"] == qc_report.summary()["n_input"]
    assert payload["extra"]["seed"] == 42
    assert len(payload["rule_results"]) == 8
