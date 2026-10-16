"""Tests for the run manifest (step 2 / experiment-manifest deliverable):
written at run start, updated after every step, accurate even for a run
that fails partway."""
import json

from martian.mars.run_manifest import RunManifest, mark_step_complete, write_run_manifest


def test_run_manifest_roundtrip(tmp_path):
    manifest = RunManifest(
        run_id="run_x", data_source="simulated", config={"seed": 42}, seed_strategy={"global_generation_seed": 42},
    )
    path = write_run_manifest(tmp_path / "manifest.json", manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "run_x"
    assert loaded["data_source"] == "simulated"
    assert loaded["status"] == "running"
    assert loaded["config"] == {"seed": 42}


def test_run_manifest_captures_software_and_hardware_info():
    manifest = RunManifest(run_id="run_x", data_source="simulated", config={}, seed_strategy={})
    assert manifest.software_versions["python"]
    assert "cpu_count" in manifest.hardware_info


def test_mark_step_complete_appends_and_times():
    manifest = RunManifest(run_id="run_x", data_source="simulated", config={}, seed_strategy={})
    mark_step_complete(manifest, "1_validate_inputs", 0.05)
    mark_step_complete(manifest, "2_write_manifest", 0.01)
    assert manifest.steps_completed == ["1_validate_inputs", "2_write_manifest"]
    assert manifest.step_timings_seconds["1_validate_inputs"] == 0.05
    assert manifest.step_timings_seconds["2_write_manifest"] == 0.01


def test_manifest_reflects_partial_progress_on_failure(tmp_path):
    manifest = RunManifest(run_id="run_x", data_source="simulated", config={}, seed_strategy={})
    mark_step_complete(manifest, "1_validate_inputs", 0.05)
    manifest.status = "failed"
    manifest.error = "boom"
    path = write_run_manifest(tmp_path / "manifest.json", manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["status"] == "failed"
    assert loaded["error"] == "boom"
    assert loaded["steps_completed"] == ["1_validate_inputs"]
