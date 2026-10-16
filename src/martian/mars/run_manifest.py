"""Step 2: the run manifest -- also serves as this pipeline's "experiment
manifest" documentation deliverable. Written once at the start of a run and
updated (rewritten in place) after every subsequent step completes, so a
run that fails partway still leaves an accurate record of what actually
finished.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from modeling.provenance import hardware_info, software_versions


@dataclass
class RunManifest:
    run_id: str
    data_source: str
    config: dict
    seed_strategy: dict
    started_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    software_versions: dict = field(default_factory=software_versions)
    hardware_info: dict = field(default_factory=hardware_info)
    steps_completed: list = field(default_factory=list)
    step_timings_seconds: dict = field(default_factory=dict)
    leakage_checks: dict | None = None
    status: str = "running"
    finished_at_utc: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def write_run_manifest(path: Path, manifest: RunManifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def mark_step_complete(manifest: RunManifest, step_name: str, elapsed_seconds: float) -> None:
    manifest.steps_completed.append(step_name)
    manifest.step_timings_seconds[step_name] = elapsed_seconds
