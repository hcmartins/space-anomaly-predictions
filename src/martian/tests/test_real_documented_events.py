"""Tests for data_sources/real_documented_events.py: the real Guha et al.
(2024) EMM/EXI MY36 dust-storm catalogue loader.

Uses a small hand-built xlsx fixture matching the real source's exact
column names (confirmed against the real downloaded file, 2026-08-05) --
no live network access at test time.
"""
import pandas as pd
import pytest

from martian.data_sources.real_documented_events import (
    load_real_documented_events,
    load_real_documented_events_config,
)


def _make_catalogue_xlsx(path):
    df = pd.DataFrame(
        {
            "Image": [
                "emm_exi_l2a_20211129T223059_0140_xos1_f635_p_v03-01",
                "emm_exi_l2a_20211129T230500_0140_xos1_f635_p_v03-01",
                "emm_exi_l2a_20211201T183837_0141_xos1_f635_p_v03-01",
            ],
            "Storm no.": [1, 1, 2],
            "Centroid Latitude": [-10.0, -11.0, -12.0],
            "Centroid Longitude": [-100.0, -101.0, -96.0],
            "MarsLs": [134.62, 134.65, 135.52],
            "MarsUT": [22.10, 22.5, 17.05],
            "MarsLTST_ZeroLongitude": [22.57, 22.9, 17.52],
            "MarsLTST_CentroidLongitude": [15.90, 16.2, 10.85],
            "Size": [33781.1, 39683.2, 47421.8],
            "Month": ["November", "November", "December"],
            "Year": [2021, 2021, 2021],
        }
    )
    df.to_excel(path, index=False)


def _config(tmp_path, catalogue_path):
    return {
        "catalogue_path": str(catalogue_path),
        "citation": "Guha et al. (2024)",
        "source_doi": "10.5281/zenodo.10275974",
        "licence": "CC BY 4.0",
        "covers_mars_year": 36,
    }


def test_load_real_documented_events_collapses_by_storm_number(tmp_path):
    xlsx_path = tmp_path / "catalogue.xlsx"
    _make_catalogue_xlsx(xlsx_path)
    config = _config(tmp_path, xlsx_path)

    events = load_real_documented_events(config)

    assert len(events) == 2  # 3 detections, 2 distinct storm numbers
    event1 = events.loc[events["event_id"] == 1].iloc[0]
    assert event1["n_detections"] == 2
    assert event1["onset_sol"] <= event1["end_sol"]


def test_load_real_documented_events_converts_longitude_to_0_360(tmp_path):
    xlsx_path = tmp_path / "catalogue.xlsx"
    _make_catalogue_xlsx(xlsx_path)
    config = _config(tmp_path, xlsx_path)

    events = load_real_documented_events(config)

    assert (events["centroid_lon"] >= 0).all()
    assert (events["centroid_lon"] <= 360).all()
    # -100.0 W -> 260.0 E
    event1 = events.loc[events["event_id"] == 1].iloc[0]
    assert event1["centroid_lon"] == pytest.approx(260.0, abs=1.0)


def test_load_real_documented_events_attaches_citation_metadata(tmp_path):
    xlsx_path = tmp_path / "catalogue.xlsx"
    _make_catalogue_xlsx(xlsx_path)
    config = _config(tmp_path, xlsx_path)

    events = load_real_documented_events(config)

    assert events.attrs["source_doi"] == "10.5281/zenodo.10275974"
    assert "Guha" in events.attrs["citation"]


def test_load_real_documented_events_raises_clear_error_when_file_missing(tmp_path):
    config = _config(tmp_path, tmp_path / "does_not_exist.xlsx")
    with pytest.raises(FileNotFoundError, match="not found"):
        load_real_documented_events(config)


def test_load_real_documented_events_config_reads_real_config_file():
    config = load_real_documented_events_config()
    assert config["covers_mars_year"] == 36
    assert "catalogue_path" in config


def test_load_real_documented_events_handles_mars_year_boundary_straddling_storm(tmp_path):
    """Real bug, confirmed against the real downloaded file (2026-08-05,
    storm 98): a storm can genuinely straddle a Mars-Year boundary. Naively
    taking min/max sol across ALL detections would read a post-boundary
    sol=1 row as an earlier onset than a pre-boundary sol=665 row -- a
    spurious ~665-sol span. onset/end must be computed only from the
    detections on the SAME side of the boundary as the event's assigned
    (majority) mars_year."""
    df = pd.DataFrame(
        {
            "Image": [
                # 4 detections late in MY36 (real Dec 2022 dates -> high sol)
                "emm_exi_l2a_20221223T115952_0311_xos1_f635_p_v06-02",
                "emm_exi_l2a_20221223T143627_0311_xos1_f635_p_v06-02",
                "emm_exi_l2a_20221223T171221_0311_xos1_f635_p_v06-02",
                "emm_exi_l2a_20221223T194901_0311_xos1_f635_p_v06-02",
                # 3 detections just after the MY36->MY37 rollover (sol resets near 0)
                "emm_exi_l2a_20221227T134237_0313_xos1_f635_p_v06-04",
                "emm_exi_l2a_20221227T162914_0313_xos1_f635_p_v06-04",
                "emm_exi_l2a_20221227T191551_0313_xos1_f635_p_v06-04",
            ],
            "Storm no.": [98] * 7,
            "Centroid Latitude": [-10.0] * 7,
            "Centroid Longitude": [-100.0] * 7,
            "MarsLs": [300.0] * 7,
            "MarsUT": [12.0] * 7,
            "MarsLTST_ZeroLongitude": [12.0] * 7,
            "MarsLTST_CentroidLongitude": [12.0] * 7,
            "Size": [50000.0] * 7,
            "Month": ["December"] * 7,
            "Year": [2022] * 7,
        }
    )
    xlsx_path = tmp_path / "catalogue.xlsx"
    df.to_excel(xlsx_path, index=False)
    config = _config(tmp_path, xlsx_path)

    events = load_real_documented_events(config)

    assert len(events) == 1
    event = events.iloc[0]
    # majority (4 of 7) detections are MY36 -> event assigned to MY36
    assert event["mars_year"] == 36
    # span must stay small (same-year detections only), not ~665 sols
    assert (event["end_sol"] - event["onset_sol"]) < 10


def test_load_real_documented_events_tolerates_trailing_space_in_image_column(tmp_path):
    """Real bug, confirmed against the real downloaded file (2026-08-05): its
    'Image' column header has a trailing space ('Image ')."""
    df = pd.DataFrame(
        {
            "Image ": ["emm_exi_l2a_20211129T223059_0140_xos1_f635_p_v03-01"],
            "Storm no.": [1],
            "Centroid Latitude": [-10.0],
            "Centroid Longitude": [-100.0],
            "MarsLs": [134.62],
            "MarsUT": [22.10],
            "MarsLTST_ZeroLongitude": [22.57],
            "MarsLTST_CentroidLongitude": [15.90],
            "Size": [33781.1],
            "Month": ["November"],
            "Year": [2021],
        }
    )
    xlsx_path = tmp_path / "catalogue.xlsx"
    df.to_excel(xlsx_path, index=False)
    config = _config(tmp_path, xlsx_path)

    events = load_real_documented_events(config)
    assert len(events) == 1
