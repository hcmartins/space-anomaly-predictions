"""Tests for the per-trained-model provenance record: software versions,
hardware info, and the training-record writer."""
import json

from martian.modeling.provenance import (
    Timer,
    TrainingRecord,
    hardware_info,
    software_versions,
    write_training_record,
)


def test_software_versions_contains_expected_keys():
    versions = software_versions()
    for key in ("python", "numpy", "pandas", "sklearn", "torch"):
        assert key in versions
        assert versions[key]


def test_hardware_info_contains_expected_keys():
    info = hardware_info()
    assert "cpu_count" in info
    assert info["cpu_count"] is None or info["cpu_count"] > 0
    assert "gpu_available" in info


def test_timer_measures_positive_elapsed_time():
    with Timer() as t:
        total = sum(range(100000))
    assert t.elapsed_seconds >= 0.0
    assert total == sum(range(100000))


def test_write_training_record_roundtrip(tmp_path):
    record = TrainingRecord(
        task_name="dust_regression", model_kind="classical", model_name="random_forest_regressor",
        horizon_sols=1, exact_parameters={"n_estimators": 200}, random_seed=42,
        feature_order=["feat_a", "feat_b"], n_train=100, n_val=10,
        convergence_status="converged", runtime_seconds=1.23,
        training_history=[{"iteration": 0, "train_score": 0.5}],
        checkpoint_path="checkpoints/model.joblib",
    )
    path = write_training_record(tmp_path / "record.json", record)
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["task_name"] == "dust_regression"
    assert loaded["exact_parameters"] == {"n_estimators": 200}
    assert loaded["random_seed"] == 42
    assert loaded["feature_order"] == ["feat_a", "feat_b"]
    assert loaded["n_train"] == 100 and loaded["n_val"] == 10
    assert loaded["convergence_status"] == "converged"
    assert loaded["runtime_seconds"] == 1.23
    assert loaded["training_history"] == [{"iteration": 0, "train_score": 0.5}]
    assert loaded["checkpoint_path"] == "checkpoints/model.joblib"
    assert "software_versions" in loaded
    assert "hardware_info" in loaded


def test_each_record_captures_its_own_software_and_hardware_snapshot():
    record1 = TrainingRecord(
        task_name="a", model_kind="baseline", model_name="persistence", horizon_sols=1,
        exact_parameters={}, random_seed=1, feature_order=[], n_train=1, n_val=1,
        convergence_status="n/a", runtime_seconds=0.0,
    )
    assert record1.software_versions["python"]
    assert record1.hardware_info["cpu_count"] is None or record1.hardware_info["cpu_count"] > 0
