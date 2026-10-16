"""Tests for step 18: writing a checksum for every output file."""
import hashlib
import json

from martian.mars.checksums import compute_sha256, write_checksums_manifest


def test_compute_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "file.txt"
    path.write_bytes(b"hello world")
    assert compute_sha256(path) == hashlib.sha256(b"hello world").hexdigest()


def test_compute_sha256_changes_when_content_changes(tmp_path):
    path = tmp_path / "file.txt"
    path.write_bytes(b"version 1")
    checksum1 = compute_sha256(path)
    path.write_bytes(b"version 2")
    checksum2 = compute_sha256(path)
    assert checksum1 != checksum2


def test_write_checksums_manifest_covers_every_file(tmp_path):
    run_dir = tmp_path / "run_x"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "data" / "a.csv").write_text("a", encoding="utf-8")
    (run_dir / "figures" / "b.png").write_bytes(b"\x89PNG")

    output_path = write_checksums_manifest(run_dir)
    checksums = json.loads(output_path.read_text(encoding="utf-8"))

    assert "data/a.csv" in checksums
    assert "figures/b.png" in checksums
    assert checksums["data/a.csv"] == hashlib.sha256(b"a").hexdigest()


def test_write_checksums_manifest_excludes_itself(tmp_path):
    run_dir = tmp_path / "run_y"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "data" / "a.csv").write_text("a", encoding="utf-8")

    output_path = write_checksums_manifest(run_dir)
    checksums = json.loads(output_path.read_text(encoding="utf-8"))
    relative_output = output_path.relative_to(run_dir).as_posix()
    assert relative_output not in checksums


def test_write_checksums_manifest_detects_tampering(tmp_path):
    run_dir = tmp_path / "run_z"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "data" / "a.csv").write_text("original", encoding="utf-8")
    output_path = write_checksums_manifest(run_dir)
    checksums = json.loads(output_path.read_text(encoding="utf-8"))
    original_checksum = checksums["data/a.csv"]

    (run_dir / "data" / "a.csv").write_text("tampered", encoding="utf-8")
    assert compute_sha256(run_dir / "data" / "a.csv") != original_checksum
