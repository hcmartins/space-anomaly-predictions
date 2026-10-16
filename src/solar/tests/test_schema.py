"""Tests for config loading, in particular the SHARP jsoc_email guardrail:
ingestion must refuse to run with no registered email rather than silently
falling back to a placeholder."""
import pytest
import yaml

from data_sources.schema import (
    GOES_NORMALIZED_COLUMNS,
    SHARP_NORMALIZED_COLUMNS,
    load_goes_config,
    load_sharp_config,
)


def test_load_goes_config_reads_real_config_file():
    config = load_goes_config()
    assert "goes_r" in config
    assert "goes_legacy" in config
    assert config["goes_r"]["satellites"] == [16, 17, 18]


def test_load_sharp_config_raises_without_jsoc_email(tmp_path, monkeypatch):
    monkeypatch.delenv("JSOC_EMAIL", raising=False)
    bad_config = tmp_path / "sharp.yaml"
    bad_config.write_text(
        yaml.safe_dump({"jsoc_email": None, "series": "hmi.sharp_cea_720s"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="jsoc_email"):
        load_sharp_config(bad_config)


def test_load_sharp_config_missing_key_also_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("JSOC_EMAIL", raising=False)
    bad_config = tmp_path / "sharp.yaml"
    bad_config.write_text(yaml.safe_dump({"series": "hmi.sharp_cea_720s"}), encoding="utf-8")
    with pytest.raises(ValueError, match="jsoc_email"):
        load_sharp_config(bad_config)


def test_load_sharp_config_succeeds_with_email(tmp_path):
    good_config = tmp_path / "sharp.yaml"
    good_config.write_text(
        yaml.safe_dump({"jsoc_email": "researcher@example.com", "series": "hmi.sharp_cea_720s"}),
        encoding="utf-8",
    )
    config = load_sharp_config(good_config)
    assert config["jsoc_email"] == "researcher@example.com"


def test_normalized_column_lists_have_no_duplicates():
    assert len(GOES_NORMALIZED_COLUMNS) == len(set(GOES_NORMALIZED_COLUMNS))
    assert len(SHARP_NORMALIZED_COLUMNS) == len(set(SHARP_NORMALIZED_COLUMNS))
