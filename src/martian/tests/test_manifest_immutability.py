"""Tests for the immutable run-manifest writer (task requirement 4)."""
import os

import pytest

from martian.data_sources.manifest import (
    ManifestAlreadyExistsError,
    new_run_id,
    read_manifest,
    write_immutable_manifest,
)


def test_write_and_read_roundtrip(tmp_path):
    run_id = "run_test_0001"
    path = write_immutable_manifest(tmp_path, run_id, {"foo": "bar", "data_source": "SIMULATED"})
    assert path.exists()
    payload = read_manifest(path)
    assert payload["foo"] == "bar"
    assert payload["run_id"] == run_id
    assert "written_at_utc" in payload


def test_rewriting_same_run_id_raises(tmp_path):
    run_id = "run_test_0002"
    write_immutable_manifest(tmp_path, run_id, {"foo": "bar"})
    with pytest.raises(ManifestAlreadyExistsError):
        write_immutable_manifest(tmp_path, run_id, {"foo": "baz"})


def test_manifest_file_is_read_only(tmp_path):
    run_id = "run_test_0003"
    path = write_immutable_manifest(tmp_path, run_id, {"foo": "bar"})
    assert not os.access(path, os.W_OK)


def test_new_run_id_has_expected_prefix():
    run_id = new_run_id()
    assert run_id.startswith("run_")
    run_id_custom = new_run_id(prefix="martian_ch4")
    assert run_id_custom.startswith("martian_ch4_")
