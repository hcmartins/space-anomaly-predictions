"""Step 18: write a checksum for every output file in the run directory --
tampering or accidental modification of any output after the run completes
becomes detectable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CHECKSUM_MANIFEST_NAME = "checksums.json"


def compute_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums_manifest(run_dir: Path, *, output_subdir: str = "reports") -> Path:
    """Walks every file under `run_dir` (excluding the checksum manifest
    itself, if one already exists from a prior partial run) and writes
    {relative_path: sha256} to `<run_dir>/<output_subdir>/checksums.json`."""
    run_dir = Path(run_dir)
    output_path = run_dir / output_subdir / CHECKSUM_MANIFEST_NAME

    checksums: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_dir() or path == output_path:
            continue
        relative = path.relative_to(run_dir).as_posix()
        checksums[relative] = compute_sha256(path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
