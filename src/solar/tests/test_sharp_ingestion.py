"""Tests for real SHARP ingestion via a fake drms.Client (no real JSOC
network access happens during the test suite)."""
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from data_sources import sharp_ingestion
from data_sources.schema import SHARP_NORMALIZED_COLUMNS, load_sharp_config


class FakeDrmsClient:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.queries: list[tuple[str, str]] = []

    def query(self, query, key):
        self.queries.append((query, key))
        if self._error is not None:
            raise self._error
        return self._result


def _sharp_config(tmp_path):
    config = load_sharp_config()
    config["jsoc_email"] = "researcher@example.com"
    config["output"]["raw_cache_dir"] = str(tmp_path / "raw")
    config["output"]["normalized_output_dir"] = str(tmp_path / "normalized")
    config["output"]["manifest_dir"] = str(tmp_path / "manifests")
    return config


def test_build_query_string_all_harpnums():
    config = {
        "series": "hmi.sharp_cea_720s",
        "time_range": {"start": "2010.05.01_00:00:00_TAI", "end": "2010.05.02_00:00:00_TAI"},
        "harpnum_filter": None,
    }
    query = sharp_ingestion.build_query_string(config)
    assert query == "hmi.sharp_cea_720s[][2010.05.01_00:00:00_TAI-2010.05.02_00:00:00_TAI]"


def test_build_query_string_with_harpnum_filter():
    config = {
        "series": "hmi.sharp_cea_720s",
        "time_range": {"start": "2010.05.01_00:00:00_TAI", "end": "2010.05.02_00:00:00_TAI"},
        "harpnum_filter": [377, 401],
    }
    query = sharp_ingestion.build_query_string(config)
    assert query == "hmi.sharp_cea_720s[377,401][2010.05.01_00:00:00_TAI-2010.05.02_00:00:00_TAI]"


def test_ingest_sharp_success_normalizes_and_writes_raw_csv(tmp_path):
    config = _sharp_config(tmp_path)
    raw_result = pd.DataFrame([
        {"HARPNUM": 401, "T_REC": "2010.05.01_00:00:00_TAI", "NOAA_AR": 11069,
         "LAT_FWT": 10.0, "LON_FWT": 5.0, "QUALITY": 0, "USFLUX": 1.2e22},
    ])
    client = FakeDrmsClient(result=raw_result)

    df, manifest = sharp_ingestion.ingest_sharp(config=config, client=client)

    assert len(df) == 1
    assert list(df.columns) == SHARP_NORMALIZED_COLUMNS
    assert df.iloc[0]["USFLUX"] == 1.2e22
    assert manifest.summary()["n_success"] == 1
    assert len(client.queries) == 1


def test_ingest_sharp_missing_keywords_become_nan_not_dropped(tmp_path):
    config = _sharp_config(tmp_path)
    # Real response is missing TOTUSJZ etc. -- must be NaN-filled, not silently omitted.
    raw_result = pd.DataFrame([{"HARPNUM": 401, "T_REC": "2010.05.01_00:00:00_TAI", "USFLUX": 1.0}])
    client = FakeDrmsClient(result=raw_result)

    df, manifest = sharp_ingestion.ingest_sharp(config=config, client=client)
    assert "TOTUSJZ" in df.columns
    assert pd.isna(df.iloc[0]["TOTUSJZ"])


def test_ingest_sharp_query_failure_recorded_not_raised(tmp_path):
    config = _sharp_config(tmp_path)
    client = FakeDrmsClient(error=RuntimeError("JSOC unavailable"))

    df, manifest = sharp_ingestion.ingest_sharp(config=config, client=client)
    assert len(df) == 0
    assert manifest.summary()["n_failed"] == 1
    assert "JSOC unavailable" in manifest.records[0].error


def test_ingest_sharp_empty_result_recorded_as_failed(tmp_path):
    config = _sharp_config(tmp_path)
    client = FakeDrmsClient(result=pd.DataFrame())

    df, manifest = sharp_ingestion.ingest_sharp(config=config, client=client)
    assert len(df) == 0
    assert manifest.summary()["n_failed"] == 1


class SequencedFakeDrmsClient:
    """Returns/raises a different (result, error) pair per call, in order --
    lets a test simulate one chunk failing while its neighbours succeed."""

    def __init__(self, outcomes: list[tuple]):
        self._outcomes = outcomes
        self.queries: list[tuple[str, str]] = []

    def query(self, query, key):
        self.queries.append((query, key))
        result, error = self._outcomes[len(self.queries) - 1]
        if error is not None:
            raise error
        return result


class SizeLimitedFakeDrmsClient:
    """Simulates JSOC's real undocumented size limit: fails any query whose
    time window exceeds max_days, succeeds (one row) otherwise -- drives the
    REAL adaptive-bisection recursion in `ingest_sharp_chunked` without a
    test having to predict the exact call order/count by hand."""

    _RANGE_RE = re.compile(
        r"\[(\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2}_TAI)-(\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2}_TAI)\]$"
    )

    def __init__(self, max_days: float):
        self.max_days = max_days
        self.queries: list[str] = []

    def query(self, query, key):
        self.queries.append(query)
        start_str, end_str = self._RANGE_RE.search(query).groups()
        start = datetime.strptime(start_str, sharp_ingestion._TAI_FORMAT)
        end = datetime.strptime(end_str, sharp_ingestion._TAI_FORMAT)
        duration_days = (end - start).total_seconds() / 86400
        if duration_days > self.max_days:
            raise RuntimeError("simulated JSOC size-limit error [status=6]")
        return pd.DataFrame([{"HARPNUM": 401, "T_REC": start_str, "USFLUX": 1.0}])


def test_month_chunks_single_month_range_is_one_chunk():
    chunks = sharp_ingestion.month_chunks("2017.09.04_00:00:00_TAI", "2017.09.10_23:59:59_TAI")
    assert chunks == [("2017.09.04_00:00:00_TAI", "2017.09.10_23:59:59_TAI")]


def test_month_chunks_splits_at_calendar_month_boundaries():
    chunks = sharp_ingestion.month_chunks("2010.05.15_00:00:00_TAI", "2010.07.10_23:59:59_TAI")
    assert chunks == [
        ("2010.05.15_00:00:00_TAI", "2010.05.31_23:59:59_TAI"),
        ("2010.06.01_00:00:00_TAI", "2010.06.30_23:59:59_TAI"),
        ("2010.07.01_00:00:00_TAI", "2010.07.10_23:59:59_TAI"),
    ]


def test_month_chunks_handles_december_to_january_year_rollover():
    chunks = sharp_ingestion.month_chunks("2010.12.15_00:00:00_TAI", "2011.01.10_23:59:59_TAI")
    assert chunks == [
        ("2010.12.15_00:00:00_TAI", "2010.12.31_23:59:59_TAI"),
        ("2011.01.01_00:00:00_TAI", "2011.01.10_23:59:59_TAI"),
    ]


def test_month_chunks_start_after_end_raises():
    with pytest.raises(ValueError):
        sharp_ingestion.month_chunks("2012.01.01_00:00:00_TAI", "2010.01.01_00:00:00_TAI")


def test_bisect_tai_range_splits_in_half():
    left, right = sharp_ingestion._bisect_tai_range("2017.01.01_00:00:00_TAI", "2017.01.05_00:00:00_TAI")
    assert left == ("2017.01.01_00:00:00_TAI", "2017.01.03_00:00:00_TAI")
    assert right == ("2017.01.03_00:00:00_TAI", "2017.01.05_00:00:00_TAI")


def test_bisect_tai_range_returns_none_at_the_floor():
    assert sharp_ingestion._bisect_tai_range("2017.01.01_00:00:00_TAI", "2017.01.01_12:00:00_TAI") is None


def test_ingest_sharp_chunked_combines_all_month_chunks_into_one_dataframe(tmp_path):
    config = _sharp_config(tmp_path)
    config["time_range"] = {"start": "2010.05.01_00:00:00_TAI", "end": "2010.06.24_23:59:59_TAI"}
    row = {"HARPNUM": 401, "T_REC": "2010.05.01_00:00:00_TAI", "USFLUX": 1.0}
    client = FakeDrmsClient(result=pd.DataFrame([row]))  # same 1-row result every call, never exceeds any limit

    df, manifest = sharp_ingestion.ingest_sharp_chunked(config=config, client=client)

    assert len(client.queries) == 2  # one query per calendar-month chunk (May, June)
    assert len(df) == 2  # one row per chunk, concatenated
    assert manifest.summary()["n_requested"] == 2
    assert manifest.summary()["n_success"] == 2


def test_ingest_sharp_chunked_short_range_reuses_ingest_sharp_behavior(tmp_path):
    # A range within one calendar month chunks to exactly one call --
    # chunking must be a no-op for a pilot-sized window, not add spurious splits.
    config = _sharp_config(tmp_path)
    config["time_range"] = {"start": "2017.09.04_00:00:00_TAI", "end": "2017.09.10_23:59:59_TAI"}
    client = FakeDrmsClient(result=pd.DataFrame([{"HARPNUM": 401, "T_REC": "2017.09.04_00:00:00_TAI", "USFLUX": 1.0}]))

    df, manifest = sharp_ingestion.ingest_sharp_chunked(config=config, client=client)
    assert len(client.queries) == 1
    assert len(df) == 1


def test_ingest_sharp_chunked_bisects_a_chunk_that_exceeds_the_real_size_limit(tmp_path):
    # One calendar month (~30 days) always exceeds this fake 10-day limit,
    # so ingest_sharp_chunked must recursively bisect it until every actual
    # sub-query stays within bounds -- the real behavior this module exists
    # to handle (see the module docstring's Feb/Oct 2014 findings).
    config = _sharp_config(tmp_path)
    config["time_range"] = {"start": "2017.09.01_00:00:00_TAI", "end": "2017.09.30_23:59:59_TAI"}
    client = SizeLimitedFakeDrmsClient(max_days=10)

    df, manifest = sharp_ingestion.ingest_sharp_chunked(config=config, client=client)

    assert len(client.queries) > 1  # the single-month chunk got bisected
    assert len(df) > 0  # every eventual sub-query succeeded
    assert manifest.summary()["n_failed"] == 0  # no query was left permanently oversized


def test_ingest_sharp_chunked_gives_up_at_the_bisection_floor(tmp_path):
    # A client that fails EVERY query, regardless of size -- bisection must
    # stop at _MIN_CHUNK and record a final failure, not recurse forever.
    config = _sharp_config(tmp_path)
    config["time_range"] = {"start": "2017.09.01_00:00:00_TAI", "end": "2017.09.05_00:00:00_TAI"}
    client = SizeLimitedFakeDrmsClient(max_days=0)  # every query, even 1 day, "exceeds" this

    df, manifest = sharp_ingestion.ingest_sharp_chunked(config=config, client=client)

    assert len(df) == 0
    assert manifest.summary()["n_failed"] > 0
    assert manifest.summary()["n_success"] == 0


def test_ingest_sharp_chunked_zero_row_chunk_is_final_not_bisected(tmp_path):
    # A legitimately empty month (no tracked active region at all) must be
    # recorded as one failure, not bisected into extra, pointless queries --
    # a smaller window cannot produce different real data here.
    config = _sharp_config(tmp_path)
    config["time_range"] = {"start": "2017.09.01_00:00:00_TAI", "end": "2017.09.30_23:59:59_TAI"}
    client = FakeDrmsClient(result=pd.DataFrame())  # always returns zero rows

    df, manifest = sharp_ingestion.ingest_sharp_chunked(config=config, client=client)

    assert len(client.queries) == 1  # NOT bisected -- one query for the whole month chunk
    assert len(df) == 0
    assert manifest.summary()["n_failed"] == 1
    assert manifest.records[0].error == "query returned zero records"
