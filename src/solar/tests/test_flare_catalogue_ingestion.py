"""Tests for real flare-catalogue ingestion, using an injected fake HTTP
session (same house style as tests/test_goes_ingestion.py) -- no real
network access during the test suite."""
from pathlib import Path

import pandas as pd

from data_sources import flare_catalogue_ingestion
from data_sources.schema import FLARE_CATALOGUE_NORMALIZED_COLUMNS, load_flare_catalogue_config

_REAL_HEADER = (
    "time,start_time,end_time,flare_id,xrsb_irrad,flare_class,xrsb_irrad_source,"
    "background_irrad,integrated_irrad_peak,integrated_irrad_end,flare_loc_swpc_hgs_lon,"
    "flare_loc_swpc_hgs_lat,flare_loc_swpc_source,flare_loc_xrs_hgs_lon,flare_loc_xrs_hgs_lat,"
    "flare_loc_xrs_hpc_x,flare_loc_xrs_hpc_y,flare_loc_xrs_source,sequential_flare_num,"
    "event_id_swpc,peak_saturated,active_region"
)
_ROW_WITH_AR = (
    "2017-09-04 05:49:00,2017-09-04 05:40:00,2017-09-04 05:55:00,201709040540,"
    "1.8e-05,M1.8,GOES-16,4e-06,1e-03,1.5e-03,,,,,,,,,1,,0,2673"
)
_ROW_WITHOUT_AR = (
    "2017-09-06 12:02:00,2017-09-06 11:52:00,2017-09-06 12:10:00,201709061152,"
    "1.464e-03,X14.6,GOES-16,8e-06,6e-01,9e-01,,,,29.8,-11.0,466.7,-284.0,GOES-16_XRS,1,,0,"
)
_ROW_OUTSIDE_DATE_RANGE = (
    "2005-01-01 00:00:00,2005-01-01 00:00:00,2005-01-01 00:10:00,200501010000,"
    "1e-06,C1.0,GOES-12,1e-07,1e-05,1e-05,,,,,,,,,1,,0,"
)


class FakeResponse:
    def __init__(self, status_code=200, text="", content=b"", headers=None):
        self.status_code = status_code
        self.text = text
        self._content = content
        self.headers = headers if headers is not None else {"Content-Length": str(len(content))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1 << 16):
        yield self._content


class FakeSession:
    def __init__(self, responses: dict[str, FakeResponse]):
        self._responses = responses

    def get(self, url, timeout=None, headers=None, stream=None):
        return self._responses.get(url, FakeResponse(status_code=404))


def _config(tmp_path):
    config = load_flare_catalogue_config()
    config["output"]["raw_cache_dir"] = str(tmp_path / "raw")
    config["request"]["max_retries"] = 1
    config["request"]["retry_backoff_seconds"] = 0
    return config


def test_ingest_flare_catalogue_normalizes_and_filters_date_range(tmp_path):
    csv_body = "\n".join([_REAL_HEADER, _ROW_WITH_AR, _ROW_WITHOUT_AR, _ROW_OUTSIDE_DATE_RANGE])
    config = _config(tmp_path)
    filename = "sci_xrsf-l2-flrpt_geo_s19950103_e20260724_v1-0-1.csv"
    index_html = f'<a href="{filename}">f</a>'
    session = FakeSession({
        config["index_url"]: FakeResponse(text=index_html),
        config["index_url"] + filename: FakeResponse(content=csv_body.encode()),
    })

    df, manifest = flare_catalogue_ingestion.ingest_flare_catalogue(session, config)

    assert manifest.summary()["n_success"] == 1
    assert list(df.columns) == FLARE_CATALOGUE_NORMALIZED_COLUMNS
    # The 2005 row is outside configs/flare_catalogue.yaml's real
    # date_range (2010-05-01..2025-07-24) and must be filtered out.
    assert len(df) == 2
    assert set(df["flare_id"]) == {"201709040540", "201709061152"}

    with_ar = df.loc[df["flare_id"] == "201709040540"].iloc[0]
    assert with_ar["active_region"] == "2673"  # not "2673.0"
    assert with_ar["flare_class"] == "M1.8"

    without_ar = df.loc[df["flare_id"] == "201709061152"].iloc[0]
    assert without_ar["active_region"] == ""  # unassigned, never fabricated


def test_ingest_flare_catalogue_records_failure_on_no_directory_match(tmp_path):
    config = _config(tmp_path)
    session = FakeSession({})  # directory listing itself 404s

    df, manifest = flare_catalogue_ingestion.ingest_flare_catalogue(session, config)

    assert len(df) == 0
    assert manifest.summary()["n_failed"] == 1


def test_ingest_flare_catalogue_picks_highest_version_when_multiple_present(tmp_path):
    csv_body = "\n".join([_REAL_HEADER, _ROW_WITH_AR])
    config = _config(tmp_path)
    old_filename = "sci_xrsf-l2-flrpt_geo_s19950103_e20260101_v1-0-0.csv"
    new_filename = "sci_xrsf-l2-flrpt_geo_s19950103_e20260724_v1-0-1.csv"
    index_html = f'<a href="{old_filename}">f1</a><a href="{new_filename}">f2</a>'
    session = FakeSession({
        config["index_url"]: FakeResponse(text=index_html),
        config["index_url"] + new_filename: FakeResponse(content=csv_body.encode()),
        # old_filename deliberately has NO stubbed response -- if the code
        # picked it, the download would 404 and the test would fail.
    })

    df, manifest = flare_catalogue_ingestion.ingest_flare_catalogue(session, config)

    assert manifest.summary()["n_success"] == 1
    assert Path(manifest.records[0].local_path).name == new_filename
