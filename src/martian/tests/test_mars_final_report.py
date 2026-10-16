"""Tests for step 17: the final report -- every number in it must come
directly from the run's own artifacts."""
from martian.mars.final_report import generate_final_report
from martian.mars.run_paths import build_run_paths, create_run_directories


def _minimal_task_result() -> dict:
    return {
        "task_name": "dust_regression",
        "task_type": "regression",
        "seed": 42,
        "horizons": {
            "1": {
                "status": "trained",
                "n_train": 100, "n_val": 10, "n_test": 50,
                "models": {
                    "random_forest_regressor": {
                        "kind": "classical",
                        "metrics": {"mae": 0.02, "r2": 0.5},
                        "checkpoint": "checkpoints/model.joblib",
                        "evaluation": {
                            "splits": {
                                "test": {
                                    "n": 50,
                                    "metrics": {"mae": 0.02, "r2": 0.5},
                                    "ci": {"mae": (0.015, 0.025), "r2": (0.4, 0.6)},
                                },
                            },
                            "baseline_comparison": [{"metric": "mae", "model_beats_baseline": True}],
                        },
                    },
                },
            },
            "3": {"status": "insufficient_data", "n_train": 1, "n_val": 0, "n_test": 0, "models": {}},
        },
    }


def test_generate_final_report_creates_file(tmp_path):
    run_paths = build_run_paths(tmp_path, "simulated", "run_x")
    create_run_directories(run_paths)
    manifest_dict = {
        "run_id": "run_x", "data_source": "simulated", "started_at_utc": "2026-01-01T00:00:00Z",
        "finished_at_utc": "2026-01-01T00:10:00Z", "status": "completed",
        "seed_strategy": {"global_generation_seed": 42},
        "step_timings_seconds": {"3_qc_and_gridding": 1.23},
        "leakage_checks": {"episode_boundary_check": "passed"},
    }
    path = generate_final_report(
        run_paths=run_paths, manifest_dict=manifest_dict,
        pipeline_summary={"n_retrievals_generated": 1000},
        task_results={"dust_regression": _minimal_task_result()},
    )
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "run_x" in content
    assert "n_retrievals_generated" in content
    assert "random_forest_regressor" in content
    assert "mae=0.02" in content
    assert "90% CI" in content
    assert "insufficient_data" in content  # horizon 3's status must be reported, not hidden


def test_generate_final_report_includes_data_source_stamp(tmp_path):
    run_paths = build_run_paths(tmp_path, "simulated", "run_y")
    create_run_directories(run_paths)
    manifest_dict = {
        "run_id": "run_y", "data_source": "simulated", "started_at_utc": "t0", "finished_at_utc": "t1",
        "status": "completed", "seed_strategy": {}, "step_timings_seconds": {}, "leakage_checks": {},
    }
    path = generate_final_report(
        run_paths=run_paths, manifest_dict=manifest_dict, pipeline_summary={}, task_results={},
    )
    content = path.read_text(encoding="utf-8")
    assert "DATA SOURCE" in content
    assert "SIMULATED" in content


def test_generate_final_report_data_source_stamp_reflects_real_runs(tmp_path):
    """Regression test: the bracketed [DATA SOURCE: ...] banner previously
    came from a separately-maintained module constant that always said
    "SIMULATED", even for a real run (contradicting the "Data source: real"
    line right below it) -- found and fixed 2026-08-05. The banner must now
    be derived from manifest_dict['data_source'], not a stale constant."""
    run_paths = build_run_paths(tmp_path, "real", "run_z")
    create_run_directories(run_paths)
    manifest_dict = {
        "run_id": "run_z", "data_source": "real", "started_at_utc": "t0", "finished_at_utc": "t1",
        "status": "completed", "seed_strategy": {}, "step_timings_seconds": {}, "leakage_checks": {},
    }
    path = generate_final_report(
        run_paths=run_paths, manifest_dict=manifest_dict, pipeline_summary={}, task_results={},
    )
    content = path.read_text(encoding="utf-8")
    assert "[DATA SOURCE: REAL" in content
    assert "SIMULATED" not in content
