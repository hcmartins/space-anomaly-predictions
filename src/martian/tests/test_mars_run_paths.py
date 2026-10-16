"""Tests for the run-scoped output directory structure:
outputs/<data_source>/<run_id>/{manifests,data,labels,features,splits,
models,metrics,figures,tables,reports}/"""
from martian.mars.run_paths import (
    REQUIRED_SUBDIRS,
    build_run_paths,
    create_run_directories,
    generate_run_id,
)


def test_generate_run_id_contains_seed():
    run_id = generate_run_id(seed=42)
    assert "seed42" in run_id
    assert run_id.startswith("run_")


def test_generate_run_id_unique_across_calls():
    id1 = generate_run_id(seed=1)
    id2 = generate_run_id(seed=1)
    assert id1 != id2  # timestamp-based, never collides even with the same seed


def test_build_run_paths_matches_required_structure():
    paths = build_run_paths("outputs", "simulated", "run_x")
    assert set(REQUIRED_SUBDIRS) == {
        "manifests", "data", "labels", "features", "splits",
        "models", "metrics", "figures", "tables", "reports",
    }
    for subdir in REQUIRED_SUBDIRS:
        assert paths[subdir] == paths["run_dir"] / subdir
    assert str(paths["run_dir"]).replace("\\", "/").endswith("outputs/simulated/run_x")


def test_create_run_directories_creates_every_subdir(tmp_path):
    paths = build_run_paths(tmp_path, "simulated", "run_y")
    create_run_directories(paths)
    for subdir in REQUIRED_SUBDIRS:
        assert paths[subdir].is_dir()


def test_different_run_ids_never_collide(tmp_path):
    paths1 = build_run_paths(tmp_path, "simulated", generate_run_id(seed=1))
    paths2 = build_run_paths(tmp_path, "simulated", generate_run_id(seed=1))
    assert paths1["run_dir"] != paths2["run_dir"]
