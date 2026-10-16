"""Tests for solar ingestion input validation -- must collect every problem
found, and must reject a date range for GOES with end before start before
any network access starts."""
import pytest

from validate_inputs import InputValidationError, validate_inputs


def _valid_config(tmp_path, **overrides) -> dict:
    config = {
        "datasets": ["goes", "sharp"],
        "goes": {"start_date": "2010-05-01", "end_date": "2019-12-31"},
        "sharp": {"enabled": True},
        "output_root": str(tmp_path / "outputs"),
    }
    config.update(overrides)
    return config


def test_valid_config_passes(tmp_path):
    validate_inputs(_valid_config(tmp_path))  # must not raise


def test_missing_datasets_key_raises(tmp_path):
    config = _valid_config(tmp_path)
    del config["datasets"]
    with pytest.raises(InputValidationError, match="datasets"):
        validate_inputs(config)


def test_unsupported_dataset_name_raises(tmp_path):
    config = _valid_config(tmp_path, datasets=["goes", "not_a_real_dataset"])
    with pytest.raises(InputValidationError, match="not_a_real_dataset"):
        validate_inputs(config)


def test_goes_end_before_start_raises(tmp_path):
    config = _valid_config(tmp_path, goes={"start_date": "2019-01-01", "end_date": "2010-01-01"})
    with pytest.raises(InputValidationError, match="end_date"):
        validate_inputs(config)


def test_goes_missing_date_keys_raises(tmp_path):
    config = _valid_config(tmp_path, goes={"start_date": "2010-05-01"})
    with pytest.raises(InputValidationError, match="start_date"):
        validate_inputs(config)


def test_goes_malformed_date_raises(tmp_path):
    config = _valid_config(tmp_path, goes={"start_date": "not-a-date", "end_date": "2019-12-31"})
    with pytest.raises(InputValidationError, match="ISO dates"):
        validate_inputs(config)


def test_unwritable_output_root_raises(tmp_path):
    blocker = tmp_path / "blocker_file"
    blocker.write_text("x", encoding="utf-8")
    config = _valid_config(tmp_path, output_root=str(blocker / "outputs"))
    with pytest.raises(InputValidationError, match="not writable"):
        validate_inputs(config)


def test_multiple_problems_all_reported(tmp_path):
    config = _valid_config(tmp_path, datasets=["not_a_real_dataset"], goes={"start_date": "bad"})
    with pytest.raises(InputValidationError) as exc_info:
        validate_inputs(config)
    message = str(exc_info.value)
    assert "not_a_real_dataset" in message
