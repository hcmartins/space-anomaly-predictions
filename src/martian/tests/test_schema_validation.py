"""Unit tests for the schema-validation mechanism (data_sources/schema.py),
covering the four requested failure categories: missing columns, invalid units,
duplicate observations, and inconsistent versions.

Uses a small hand-built schema fixture (not the full EMIRS schema) so these
tests exercise `validate_schema` in isolation. The full EMIRS schema is
exercised separately in test_simulated_emirs_provider.py.
"""
import pandas as pd
import pytest

from martian.data_sources.schema import FieldSpec, SchemaValidationError, validate_schema


def _make_specs():
    return [
        FieldSpec("a", "float64", "K", True, True, "test field", "TEST", valid_range=(0, 100)),
        FieldSpec("b", "string", None, True, True, "test field", "TEST"),
        FieldSpec("version", "string", None, True, True, "test field", "TEST"),
    ]


def test_valid_dataframe_passes():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "version": ["v1", "v1"]})
    validate_schema(df, _make_specs(), "test_provider")  # must not raise


def test_missing_required_column_raises():
    df = pd.DataFrame({"a": [1.0], "version": ["v1"]})  # missing 'b'
    with pytest.raises(SchemaValidationError, match="Missing required columns"):
        validate_schema(df, _make_specs(), "test_provider")


def test_out_of_range_value_raises():
    df = pd.DataFrame({"a": [1.0, 200.0], "b": ["x", "y"], "version": ["v1", "v1"]})
    with pytest.raises(SchemaValidationError, match="outside declared valid range"):
        validate_schema(df, _make_specs(), "test_provider")


def test_declared_unit_mismatch_raises():
    df = pd.DataFrame({"a": [1.0], "b": ["x"], "version": ["v1"]})
    df.attrs["units"] = {"a": "degrees"}  # schema declares 'K' for field 'a'
    with pytest.raises(SchemaValidationError, match="declares unit"):
        validate_schema(df, _make_specs(), "test_provider")


def test_duplicate_observations_raises():
    df = pd.DataFrame({"a": [1.0, 1.0], "b": ["x", "x"], "version": ["v1", "v1"]})
    with pytest.raises(SchemaValidationError, match="duplicate observation"):
        validate_schema(df, _make_specs(), "test_provider", check_duplicates_on=["a", "b"])


def test_no_false_positive_duplicates_when_rows_differ():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "version": ["v1", "v1"]})
    validate_schema(df, _make_specs(), "test_provider", check_duplicates_on=["a", "b"])  # must not raise


def test_inconsistent_versions_raises():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "version": ["v1", "v2"]})
    with pytest.raises(SchemaValidationError, match="Inconsistent 'version'"):
        validate_schema(df, _make_specs(), "test_provider")


def test_multiple_violations_all_reported_together():
    # Missing 'b' AND inconsistent version AND out-of-range 'a' — all three must
    # appear in a single raised error, not just the first one found.
    df = pd.DataFrame({"a": [1.0, 999.0], "version": ["v1", "v2"]})
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_schema(df, _make_specs(), "test_provider")
    msg = str(excinfo.value)
    assert "Missing required columns" in msg
    assert "outside declared valid range" in msg
    assert "Inconsistent 'version'" in msg
