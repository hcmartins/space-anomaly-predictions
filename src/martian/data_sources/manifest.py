"""Immutable run-manifest writer (task requirement 4).

A manifest, once written for a given run_id, is never modified in place.
Re-running the same run_id raises rather than overwriting; a new run always gets
a new run_id (UTC timestamp + prefix). The written file is additionally marked
read-only on disk as a second, filesystem-level guard.

Honest limitation: chmod-based read-only is a best-effort guard against
accidental in-place edits by the same tooling, not a cryptographic tamper-proof
guarantee — a sufficiently privileged process can still remove the file and
recreate one with the same name. This is documented here rather than overstated.
"""
from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ManifestAlreadyExistsError(RuntimeError):
    """Raised when a write is attempted against a run_id whose manifest already
    exists. Manifests are immutable: generate a new run_id instead of retrying."""


def new_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{ts}"


def write_immutable_manifest(manifest_dir: Path, run_id: str, payload: dict[str, Any]) -> Path:
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{run_id}.manifest.json"

    if path.exists():
        raise ManifestAlreadyExistsError(
            f"Manifest for run_id={run_id!r} already exists at {path}. Manifests "
            f"are immutable; generate a new run_id (see new_run_id()) instead of "
            f"overwriting."
        )

    full_payload = {
        "run_id": run_id,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    path.write_text(json.dumps(full_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    # Filesystem-level immutability guard — best-effort, see module docstring.
    os.chmod(path, stat.S_IREAD)

    return path


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
