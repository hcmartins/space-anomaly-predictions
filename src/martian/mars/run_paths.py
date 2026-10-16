"""Run-scoped output directory structure:

    outputs/<data_source>/<run_id>/{manifests,data,labels,features,splits,
                                    models,metrics,figures,tables,reports}/

Every `mars.run_all` run gets its own `run_id` subdirectory -- never
overwrites a previous run's outputs, so any past run remains reproducible
and inspectable after a later run completes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

REQUIRED_SUBDIRS = (
    "manifests", "data", "labels", "features", "splits",
    "models", "metrics", "figures", "tables", "reports",
)


def generate_run_id(seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run_{timestamp}_seed{seed}"


def build_run_paths(output_root: Path, data_source: str, run_id: str) -> dict[str, Path]:
    run_dir = Path(output_root) / data_source / run_id
    paths = {"run_dir": run_dir}
    for subdir in REQUIRED_SUBDIRS:
        paths[subdir] = run_dir / subdir
    return paths


def create_run_directories(paths: dict[str, Path]) -> None:
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
