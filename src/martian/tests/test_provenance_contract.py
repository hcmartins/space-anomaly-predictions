"""Tests for the per-record provenance contract (task requirement 1)."""
import pandas as pd
import pytest

from martian.data_sources.base import BaseProvider, ProvenanceContractError
from martian.data_sources.provenance import (
    REQUIRED_PROVENANCE_COLUMNS,
    ProviderIdentity,
    attach_batch_provenance,
    compute_row_checksum,
)


class _DummyProvider(BaseProvider):
    """Minimal concrete BaseProvider for testing the enforcement mechanism in
    isolation from any real provider's generation logic."""

    def describe(self):
        return {"data_source": "SIMULATED", "product_id": "X", "processing_level": "L0", "version": "v0"}


def test_required_provenance_columns_enforced_on_complete_frame():
    provider = _DummyProvider()
    good_df = pd.DataFrame({c: ["x"] for c in REQUIRED_PROVENANCE_COLUMNS})
    out = provider._enforce_provenance_contract(good_df, caller="test")
    assert set(REQUIRED_PROVENANCE_COLUMNS).issubset(out.columns)


def test_missing_provenance_column_raises():
    provider = _DummyProvider()
    bad_df = pd.DataFrame({"data_source": ["x"]})  # missing everything else
    with pytest.raises(ProvenanceContractError, match="missing required provenance"):
        provider._enforce_provenance_contract(bad_df, caller="test")


def test_checksum_deterministic_and_content_sensitive():
    a = compute_row_checksum([1, "x", 3.0])
    b = compute_row_checksum([1, "x", 3.0])
    c = compute_row_checksum([1, "x", 4.0])
    assert a == b
    assert a != c


def test_attach_batch_provenance_populates_all_required_columns():
    df = pd.DataFrame({"value": [1, 2, 3]})
    identity = ProviderIdentity(data_source="SIMULATED", product_id="X", processing_level="L0", version="v0")
    n = len(df)
    out = attach_batch_provenance(
        df,
        identity,
        original_filenames=pd.Series(["f1", "f2", "f3"]),
        quality_flags=pd.Series([0, 0, 1]),
        retrieval_timestamps=pd.Series([pd.Timestamp.now(tz="UTC")] * n),
        checksum_source_columns=["value"],
    )
    for col in REQUIRED_PROVENANCE_COLUMNS:
        assert col in out.columns
    assert out["checksum"].nunique() == 3  # three distinct 'value' contents


def test_attach_batch_provenance_rejects_unknown_checksum_columns():
    df = pd.DataFrame({"value": [1, 2]})
    identity = ProviderIdentity(data_source="SIMULATED", product_id="X", processing_level="L0", version="v0")
    with pytest.raises(KeyError):
        attach_batch_provenance(
            df,
            identity,
            original_filenames=pd.Series(["f1", "f2"]),
            quality_flags=pd.Series([0, 0]),
            retrieval_timestamps=pd.Series([pd.Timestamp.now(tz="UTC")] * 2),
            checksum_source_columns=["does_not_exist"],
        )


def test_attach_batch_provenance_handles_empty_dataframe():
    df = pd.DataFrame({"value": pd.Series([], dtype="float64")})
    identity = ProviderIdentity(data_source="SIMULATED", product_id="X", processing_level="L0", version="v0")
    out = attach_batch_provenance(
        df,
        identity,
        original_filenames=pd.Series([], dtype="object"),
        quality_flags=pd.Series([], dtype="int64"),
        retrieval_timestamps=pd.Series([], dtype="datetime64[ns, UTC]"),
        checksum_source_columns=["value"],
    )
    assert len(out) == 0
    for col in REQUIRED_PROVENANCE_COLUMNS:
        assert col in out.columns
