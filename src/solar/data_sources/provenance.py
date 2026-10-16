"""Provenance stamping and checksums for real ingested data.

Every downloaded file is recorded in a run manifest with its source URL (or
JSOC query), access timestamp, local path, byte size, and SHA-256 checksum
-- so it is always possible to trace a downstream row back to exactly which
real file it came from and when it was fetched. This mirrors the
`data_source` provenance discipline established in `martian/data_sources/`,
adapted for real (not simulated) data: the provenance dict here records what
was ACTUALLY downloaded, not a generator's parameters.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class IngestedFileRecord:
    """One entry per real file successfully or unsuccessfully retrieved."""

    source: str                    # "REAL (NOAA GOES-R L2)" | "REAL (NOAA GOES legacy)" | "REAL (JSOC SHARP export)"
    request_identifier: str        # URL fetched, or JSOC query string
    local_path: str | None         # None if the fetch failed
    status: str                    # "success" | "failed"
    http_status: int | None = None
    error: str | None = None
    byte_size: int | None = None
    sha256: str | None = None
    accessed_at_utc: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "request_identifier": self.request_identifier,
            "local_path": self.local_path,
            "status": self.status,
            "http_status": self.http_status,
            "error": self.error,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "accessed_at_utc": self.accessed_at_utc,
        }


class IngestionManifest:
    """Accumulates IngestedFileRecords for one ingestion run and writes them
    to a single JSON manifest -- never silently drops a failed fetch."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.started_at_utc = _utc_now_iso()
        self.records: list[IngestedFileRecord] = []

    def add(self, record: IngestedFileRecord) -> None:
        self.records.append(record)

    def summary(self) -> dict:
        n_success = sum(1 for r in self.records if r.status == "success")
        n_failed = sum(1 for r in self.records if r.status == "failed")
        return {"n_requested": len(self.records), "n_success": n_success, "n_failed": n_failed}

    def write(self, manifest_dir: Path) -> Path:
        manifest_dir = Path(manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = manifest_dir / f"{self.dataset_name}_ingestion_manifest_{timestamp}.json"
        payload = {
            "dataset_name": self.dataset_name,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": _utc_now_iso(),
            "summary": self.summary(),
            "records": [r.to_dict() for r in self.records],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path
