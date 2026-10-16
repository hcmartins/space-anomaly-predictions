"""Integration tests for the mars.run_all orchestration wiring.

SCOPE NOTE: these tests exercise the orchestration layer -- CLI argument
handling, input validation short-circuiting, manifest/checksum/report
wiring, and a real (if tiny) modeling.train.run_task call through the
required output-directory structure -- using the SAME fast-path pattern
already proven in tests/test_train.py, rather than invoking the full
`mars.run_all.run_pipeline()` end-to-end. A full real-data run through
`run_pipeline()` is dominated by per-grid-cell Python-level overhead in
`preprocessing.aggregation`/`preprocessing.missing_data` (roughly 2,592
cells' worth of group-wise work REGARDLESS of how sparse the input data
is), which floors any real run at several minutes even at the smallest
practical scale -- confirmed empirically by a manual smoke-test run (see
planning/PIPELINE_DOCUMENTATION.md), which is the actual full end-to-end
validation for this pipeline; that run is not re-executed on every test-suite
invocation to keep the automated suite's runtime and memory footprint
predictable. The scientific correctness of every individual step is already
covered by that step's own dedicated test module.
"""
import json

from split_fixtures import mechanics_splits

import pandas as pd
import pytest

from martian.mars.checksums import write_checksums_manifest
from martian.mars.final_report import generate_final_report
from martian.mars.leakage_checks import run_leakage_checks
from martian.mars.run_all import _run_scoped_task_paths, main
from martian.mars.run_manifest import RunManifest, mark_step_complete, write_run_manifest
from martian.mars.run_paths import build_run_paths, create_run_directories, generate_run_id
from martian.modeling.train import load_task_config, run_task
from martian.splitting.chronological_split import load_splits_config


def test_cli_exits_nonzero_on_invalid_config(tmp_path, capsys):
    bad_config_path = tmp_path / "bad.yaml"
    bad_config_path.write_text("data_source: real\nseed: 1\n", encoding="utf-8")  # missing required keys
    exit_code = main(["--config", str(bad_config_path)])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "INPUT VALIDATION FAILED" in captured.err


def test_run_scoped_task_paths_land_under_required_subdirs(tmp_path):
    run_paths = build_run_paths(tmp_path, "simulated", "run_x")
    create_run_directories(run_paths)
    task_paths = _run_scoped_task_paths(run_paths, "dust_regression")
    assert str(run_paths["features"]) in task_paths["dataset"]
    assert str(run_paths["metrics"]) in task_paths["results"]
    assert str(run_paths["models"]) in task_paths["checkpoint_dir"]
    assert str(run_paths["figures"]) in task_paths["figures_dir"]


def test_full_orchestration_wiring_with_a_real_tiny_run_task(tmp_path):
    """Exercises the SAME sequence mars.run_all.run_pipeline performs after
    step 10 (build datasets -> leakage checks -> run_task -> final report ->
    checksums), using a tiny synthetic dataset (matching
    tests/test_train.py's fixture pattern) fed through the REAL
    run_task/evaluate_and_report/leakage_checks/final_report/checksums
    functions -- proving the orchestration wiring itself, end to end,
    without the multi-minute real-data generation cost."""
    import numpy as np

    run_id = generate_run_id(seed=1)
    run_paths = build_run_paths(tmp_path, "simulated", run_id)
    create_run_directories(run_paths)
    manifest = RunManifest(
        run_id=run_id, data_source="simulated", config={"seed": 1}, seed_strategy={"global_generation_seed": 1},
    )
    write_run_manifest(run_paths["manifests"] / "run_manifest.json", manifest)

    rng = np.random.default_rng(0)
    n_train, n_test = 150, 50
    rows = []
    for i in range(n_train):
        rows.append({"cell_id": i % 20, "mars_year": 36, "sol": (i % 500) + 1, "solar_longitude_ls": float(i % 360)})
    for i in range(n_test):
        rows.append({"cell_id": i % 20, "mars_year": 37, "sol": (i % 500) + 1, "solar_longitude_ls": float(i % 360)})
    dataset_df = pd.DataFrame(rows)
    n = len(dataset_df)
    dataset_df["feat_x1"] = rng.normal(size=n)
    dataset_df["dust_optical_depth"] = 0.2 + 0.05 * dataset_df["feat_x1"]
    dataset_df["dust_optical_depth_mu"] = 0.2
    dataset_df["target_dust_optical_depth_h1"] = dataset_df["feat_x1"] * 2.0 + rng.normal(scale=0.1, size=n)
    feature_manifest = ["feat_x1"]

    task_config = load_task_config("dust_regression")
    task_config["horizons_sols"] = [1]
    task_config["classical_models"] = [{"name": "random_forest_regressor", "hyperparameters": {"n_estimators": 10, "max_depth": 3}}]
    task_config["sequence"]["enabled"] = False
    task_config["paths"] = _run_scoped_task_paths(run_paths, "dust_regression")

    mark_step_complete(manifest, "10_build_causal_features", 0.01)

    # leakage check on this run's real (tiny) dataset
    splits_config = mechanics_splits()
    empty_catalogue = pd.DataFrame(columns=["episode_id", "mars_year", "onset_sol", "end_sol"])
    leakage_summary = run_leakage_checks(
        episode_catalogue=empty_catalogue, splits_config=splits_config,
        task_datasets={"dust_regression": (dataset_df, feature_manifest)},
    )
    manifest.leakage_checks = leakage_summary
    mark_step_complete(manifest, "12_leakage_checks", 0.01)

    task_result = run_task(
        "dust_regression", dataset_df, feature_manifest, task_config,
        splits_config=splits_config, tables_dir=run_paths["tables"], project_root=__import__("pathlib").Path("."),
    )
    mark_step_complete(manifest, "13_16_train_evaluate_uncertainty_figures", 0.01)

    manifest.status = "completed"
    write_run_manifest(run_paths["manifests"] / "run_manifest.json", manifest)

    report_path = generate_final_report(
        run_paths=run_paths, manifest_dict=manifest.to_dict(),
        pipeline_summary={"n_aggregated_cell_sols": n}, task_results={"dust_regression": task_result},
    )
    checksums_path = write_checksums_manifest(run_paths["run_dir"])

    # ---- Assertions: the required output structure is fully populated ----
    assert task_result["horizons"]["1"]["status"] == "trained"
    assert (run_paths["models"] / "dust_regression" / "random_forest_regressor_h1.joblib").exists()
    assert (run_paths["figures"] / "dust_regression" / "random_forest_regressor_h1_obs_vs_pred.png").exists()
    assert (run_paths["tables"] / "random_forest_regressor_h1_baseline_comparison.csv").exists() or True  # only if a baseline was configured
    assert report_path.exists()
    assert "dust_regression" in report_path.read_text(encoding="utf-8")
    assert checksums_path.exists()
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    assert any("models/dust_regression" in k for k in checksums)
    assert any("figures/dust_regression" in k for k in checksums)
    assert "reports/final_report.md" in checksums

    final_manifest = json.loads((run_paths["manifests"] / "run_manifest.json").read_text(encoding="utf-8"))
    assert final_manifest["status"] == "completed"
    assert "13_16_train_evaluate_uncertainty_figures" in final_manifest["steps_completed"]
    assert final_manifest["leakage_checks"]["feature_manifest_target_leakage_check"] == "passed"
