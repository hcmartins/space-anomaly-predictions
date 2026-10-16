"""Tests for real GOES XRS ingestion, using injected fake HTTP sessions so
no real network access happens during the test suite. The actual
`ingest_goes*` functions are exercised end-to-end (only the network layer
and netCDF parsing -- which need a real file/xarray -- are stubbed)."""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data_sources import goes_ingestion
from data_sources.schema import GOES_NORMALIZED_COLUMNS, load_goes_config


class FakeResponse:
    def __init__(self, status_code=200, text="", content=b"fake netcdf bytes", headers=None):
        self.status_code = status_code
        self.text = text
        self._content = content
        # Real responses report Content-Length; default here matches _content's
        # real length so tests aren't opting into the truncated-download path
        # unless they explicitly ask for it via `headers`.
        self.headers = headers if headers is not None else {"Content-Length": str(len(content))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1 << 16):
        yield self._content


class FakeSession:
    """Maps exact URLs to canned FakeResponses; unmapped URLs 404."""

    def __init__(self, responses: dict[str, FakeResponse]):
        self._responses = responses
        self.get_calls: list[str] = []

    def get(self, url, timeout=None, headers=None, stream=None):
        self.get_calls.append(url)
        return self._responses.get(url, FakeResponse(status_code=404))


def _minimal_goes_config(tmp_path):
    config = load_goes_config()
    config["output"]["raw_cache_dir"] = str(tmp_path / "raw")
    config["output"]["normalized_output_dir"] = str(tmp_path / "normalized")
    config["output"]["manifest_dir"] = str(tmp_path / "manifests")
    config["request"]["max_retries"] = 2
    config["request"]["retry_backoff_seconds"] = 0
    return config


def test_resolve_directory_listing_extracts_matching_filenames():
    html = (
        '<a href="sci_xrsf-l2-avg1m_g16_d20180101_v2-2-0.nc">f1</a>'
        '<a href="sci_xrsf-l2-avg1m_g16_d20180101_v2-1-0.nc">f2</a>'
        '<a href="unrelated_file.nc">other</a>'
    )
    session = FakeSession({"http://example.com/idx/": FakeResponse(text=html)})
    matches = goes_ingestion.resolve_directory_listing(
        session, "http://example.com/idx/", r'href="(sci_xrsf-l2-avg1m_g16_d20180101[\w\.\-]*\.nc)"'
    )
    assert matches == sorted({
        "sci_xrsf-l2-avg1m_g16_d20180101_v2-2-0.nc",
        "sci_xrsf-l2-avg1m_g16_d20180101_v2-1-0.nc",
    })


def test_download_success_writes_file_and_checksum(tmp_path):
    session = FakeSession({"http://example.com/f.nc": FakeResponse(status_code=200, content=b"abc")})
    config = _minimal_goes_config(tmp_path)
    dest = tmp_path / "raw" / "f.nc"
    record = goes_ingestion._download(session, "http://example.com/f.nc", dest, config)
    assert record.status == "success"
    assert record.sha256 is not None
    assert dest.exists()
    assert dest.read_bytes() == b"abc"


def test_download_truncated_stream_is_rejected_not_saved_as_success(tmp_path):
    # Content-Length says 100 bytes but the stream only delivers 3 -- a real
    # network stall/truncation must be caught here, not silently accepted
    # and left for the netCDF parser to choke on with an opaque low-level
    # error (found via a real flaky-network smoke-test run).
    session = FakeSession({
        "http://example.com/f.nc": FakeResponse(status_code=200, content=b"abc", headers={"Content-Length": "100"})
    })
    config = _minimal_goes_config(tmp_path)
    config["request"]["max_retries"] = 1
    dest = tmp_path / "raw" / "f.nc"
    record = goes_ingestion._download(session, "http://example.com/f.nc", dest, config)
    assert record.status == "failed"
    assert "incomplete download" in record.error
    assert not dest.exists()


def test_download_skips_network_call_when_file_already_cached(tmp_path):
    # dest_path is only ever left on disk after a full, size-verified
    # success (see the truncated-stream test above) -- so a pre-existing
    # file is safe to treat as already fetched. Added after a real
    # multi-hour background ingestion run was killed partway through by an
    # unrelated environment issue; without this, restarting it would have
    # silently re-downloaded every already-cached file from scratch.
    dest = tmp_path / "raw" / "f.nc"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"already downloaded")
    session = FakeSession({"http://example.com/f.nc": FakeResponse(status_code=200, content=b"different bytes")})
    config = _minimal_goes_config(tmp_path)

    record = goes_ingestion._download(session, "http://example.com/f.nc", dest, config)

    assert record.status == "success"
    assert record.sha256 is not None
    assert session.get_calls == []  # no network call was made at all
    assert dest.read_bytes() == b"already downloaded"  # untouched, not overwritten


def test_download_failure_retries_then_records_failed(tmp_path):
    session = FakeSession({})  # every URL 404s
    config = _minimal_goes_config(tmp_path)
    dest = tmp_path / "raw" / "missing.nc"
    record = goes_ingestion._download(session, "http://example.com/missing.nc", dest, config)
    assert record.status == "failed"
    assert record.http_status == 404
    # max_retries=2 -> exactly 2 attempts, not silently retried forever or only once
    assert len(session.get_calls) == 2


def test_ingest_goes_r_records_failure_when_no_directory_match(tmp_path, monkeypatch):
    config = _minimal_goes_config(tmp_path)
    config["goes_r"]["satellites"] = [16]
    session = FakeSession({})  # directory listing itself 404s

    df, manifest = goes_ingestion.ingest_goes_r(session, config, date(2018, 1, 1), date(2018, 1, 1))
    assert len(df) == 0
    assert manifest.summary()["n_failed"] == 1


def test_ingest_goes_r_success_path_with_stubbed_parse(tmp_path, monkeypatch):
    config = _minimal_goes_config(tmp_path)
    config["goes_r"]["satellites"] = [16]
    index_url = f"{config['goes_r']['base_url'].format(sat=16)}/2018/01/"
    filename = "sci_xrsf-l2-avg1m_g16_d20180101_v2-2-0.nc"
    html = f'<a href="{filename}">f</a>'
    session = FakeSession({
        index_url: FakeResponse(text=html),
        index_url + filename: FakeResponse(content=b"fake"),
    })

    def fake_parse(path, variables, satellite, series, time_variable):
        return pd.DataFrame([{
            "time_utc": pd.Timestamp("2018-01-01", tz="UTC"), "satellite": satellite,
            "satellite_series": series, "xrsa_flux_w_m2": 1e-8, "xrsb_flux_w_m2": 1e-7,
            "xrsa_quality": 0, "xrsb_quality": 0, "source_file": path.name,
        }])[GOES_NORMALIZED_COLUMNS]

    monkeypatch.setattr(goes_ingestion, "_parse_goes_netcdf", fake_parse)

    df, manifest = goes_ingestion.ingest_goes_r(session, config, date(2018, 1, 1), date(2018, 1, 1))
    assert len(df) == 1
    assert df.iloc[0]["satellite"] == 16
    assert manifest.summary()["n_success"] == 1


def test_ingest_goes_legacy_builds_monthly_url_and_records_failure_on_404(tmp_path):
    config = _minimal_goes_config(tmp_path)
    config["goes_legacy"]["satellites"] = [15]
    session = FakeSession({})  # every URL 404s

    df, manifest = goes_ingestion.ingest_goes_legacy(session, config, date(2015, 3, 1), date(2015, 3, 31))
    assert len(df) == 0
    assert manifest.summary()["n_requested"] == 1
    assert manifest.summary()["n_failed"] == 1


def test_ingest_goes_routes_dates_to_correct_archive(tmp_path, monkeypatch):
    config = _minimal_goes_config(tmp_path)
    calls = []

    def fake_ingest_r(session, cfg, start, end):
        calls.append(("goes_r", start, end))
        return pd.DataFrame(columns=GOES_NORMALIZED_COLUMNS), goes_ingestion.IngestionManifest("goes_r")

    def fake_ingest_legacy(session, cfg, start, end):
        calls.append(("goes_legacy", start, end))
        return pd.DataFrame(columns=GOES_NORMALIZED_COLUMNS), goes_ingestion.IngestionManifest("goes_legacy")

    monkeypatch.setattr(goes_ingestion, "ingest_goes_r", fake_ingest_r)
    monkeypatch.setattr(goes_ingestion, "ingest_goes_legacy", fake_ingest_legacy)

    goes_ingestion.ingest_goes(date(2016, 1, 1), date(2018, 1, 1), config=config, session=FakeSession({}))

    kinds = {c[0] for c in calls}
    assert kinds == {"goes_r", "goes_legacy"}
    legacy_call = next(c for c in calls if c[0] == "goes_legacy")
    r_call = next(c for c in calls if c[0] == "goes_r")
    assert legacy_call[1] == date(2016, 1, 1)  # requested start, before goes_r begins
    assert r_call[2] == date(2018, 1, 1)  # requested end
