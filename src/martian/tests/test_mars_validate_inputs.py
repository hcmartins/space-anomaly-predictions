"""Tests for Step 1: input validation -- must collect every problem found,
never just the first, before any expensive work starts. Real-data-only
pipeline: no data_source selection, no simulation density knobs."""
import pytest

from martian.mars.validate_inputs import InputValidationError, validate_inputs


def _valid_config(tmp_path, **overrides) -> dict:
    config = {
        "seed": 42,
        "mars_year_range": [36, 37],
        "output_root": str(tmp_path / "outputs"),
    }
    config.update(overrides)
    return config


def test_valid_config_passes(tmp_path):
    validate_inputs(_valid_config(tmp_path))  # must not raise


def test_missing_required_key_raises(tmp_path):
    config = _valid_config(tmp_path)
    del config["seed"]
    with pytest.raises(InputValidationError, match="seed"):
        validate_inputs(config)


def test_non_integer_seed_raises(tmp_path):
    config = _valid_config(tmp_path, seed="forty-two")
    with pytest.raises(InputValidationError, match="seed"):
        validate_inputs(config)


def test_invalid_mars_year_range_raises(tmp_path):
    config = _valid_config(tmp_path, mars_year_range=[3, 1])  # start > end
    with pytest.raises(InputValidationError, match="mars_year_range"):
        validate_inputs(config)


def test_missing_task_config_file_raises(tmp_path):
    config = _valid_config(tmp_path, tasks=["not_a_real_task"], configs_dir=str(tmp_path))
    with pytest.raises(InputValidationError, match="not_a_real_task"):
        validate_inputs(config)


def test_multiple_problems_all_reported(tmp_path):
    config = _valid_config(tmp_path, seed="bad", mars_year_range=[3, 1])
    with pytest.raises(InputValidationError) as exc_info:
        validate_inputs(config)
    message = str(exc_info.value)
    assert "seed" in message
    assert "mars_year_range" in message


def test_unwritable_output_root_raises(tmp_path, monkeypatch):
    # Simulate an unwritable output_root by pointing at a path whose parent
    # is a FILE (not a directory), which Path.mkdir cannot create into.
    blocker = tmp_path / "blocker_file"
    blocker.write_text("x", encoding="utf-8")
    config = _valid_config(tmp_path, output_root=str(blocker / "outputs"))
    with pytest.raises(InputValidationError, match="not writable"):
        validate_inputs(config)
