"""Tests for provenance stamping: a failed fetch must be recorded, never
silently dropped, and every manifest write must be traceable (checksum,
timestamp, per-record status)."""
import json

from data_sources.provenance import IngestedFileRecord, IngestionManifest, compute_sha256


def test_compute_sha256_matches_known_value(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_text("hello world", encoding="utf-8")
    # sha256("hello world") is a well-known test vector
    assert compute_sha256(path) == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_ingested_file_record_to_dict_round_trips_all_fields():
    record = IngestedFileRecord(
        source="REAL (NOAA GOES-R L2)", request_identifier="https://example.com/f.nc",
        local_path="/tmp/f.nc", status="success", http_status=200,
        byte_size=1024, sha256="abc123",
    )
    d = record.to_dict()
    assert d["source"] == "REAL (NOAA GOES-R L2)"
    assert d["status"] == "success"
    assert d["byte_size"] == 1024
    assert "accessed_at_utc" in d


def test_manifest_summary_counts_success_and_failure():
    manifest = IngestionManifest("goes_r")
    manifest.add(IngestedFileRecord(source="s", request_identifier="u1", local_path="p1", status="success"))
    manifest.add(IngestedFileRecord(source="s", request_identifier="u2", local_path=None, status="failed", error="404"))
    manifest.add(IngestedFileRecord(source="s", request_identifier="u3", local_path="p3", status="success"))

    summary = manifest.summary()
    assert summary == {"n_requested": 3, "n_success": 2, "n_failed": 1}


def test_manifest_write_persists_failed_records_not_just_successes(tmp_path):
    manifest = IngestionManifest("sharp")
    manifest.add(IngestedFileRecord(source="s", request_identifier="q1", local_path=None, status="failed", error="timeout"))

    written_path = manifest.write(tmp_path)
    payload = json.loads(written_path.read_text(encoding="utf-8"))

    assert payload["summary"]["n_failed"] == 1
    assert payload["records"][0]["status"] == "failed"
    assert payload["records"][0]["error"] == "timeout"


def test_manifest_write_never_overwrites_previous_run(tmp_path):
    manifest_a = IngestionManifest("goes_r")
    manifest_a.add(IngestedFileRecord(source="s", request_identifier="u", local_path="p", status="success"))
    path_a = manifest_a.write(tmp_path)

    manifest_b = IngestionManifest("goes_r")
    manifest_b.add(IngestedFileRecord(source="s", request_identifier="u2", local_path="p2", status="success"))
    path_b = manifest_b.write(tmp_path)

    assert path_a != path_b
    assert path_a.exists() and path_b.exists()
