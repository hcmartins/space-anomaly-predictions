"""Tests for the descriptive (reporting-only) storm-taxonomy classifier."""
import pandas as pd

from martian.labeling.storm_taxonomy import (
    classify_episode,
    classify_episode_catalogue,
    load_storm_taxonomy_config,
)


def test_classify_local():
    cls = classify_episode(peak_area_cells=5, duration_sols=2, longitude_sectors_spanned=1)
    assert cls == "local"


def test_classify_regional():
    cls = classify_episode(peak_area_cells=50, duration_sols=10, longitude_sectors_spanned=1)
    assert cls == "regional"


def test_classify_planet_encircling():
    cls = classify_episode(peak_area_cells=250, duration_sols=25, longitude_sectors_spanned=3)
    assert cls == "planet_encircling"


def test_planet_encircling_requires_all_three_conditions():
    # Large area and long duration but only 1 longitude sector spanned --
    # must NOT be classified planet_encircling.
    cls = classify_episode(peak_area_cells=250, duration_sols=25, longitude_sectors_spanned=1)
    assert cls != "planet_encircling"


def test_classify_unclassified_when_no_band_matches():
    # area=300 (too big for local/regional), duration=5 (too short for
    # planet_encircling) -- matches no band.
    cls = classify_episode(peak_area_cells=300, duration_sols=5, longitude_sectors_spanned=1)
    assert cls == "unclassified"


def test_classify_episode_catalogue_attaches_column():
    catalogue = pd.DataFrame(
        [
            {"episode_id": 0, "peak_area_cells": 5, "duration_sols": 2, "longitude_sectors_spanned": 1},
            {"episode_id": 1, "peak_area_cells": 250, "duration_sols": 25, "longitude_sectors_spanned": 3},
        ]
    )
    out = classify_episode_catalogue(catalogue)
    assert out["taxonomy_class"].tolist() == ["local", "planet_encircling"]


def test_classify_episode_catalogue_empty_input():
    empty = pd.DataFrame(columns=["episode_id", "peak_area_cells", "duration_sols", "longitude_sectors_spanned"])
    out = classify_episode_catalogue(empty)
    assert "taxonomy_class" in out.columns
    assert len(out) == 0


def test_config_loaded_and_used_at_runtime(tmp_path):
    config = load_storm_taxonomy_config()
    custom = dict(config)
    # Shrink the "local" band's max_area_cells from 20 to 5 -- a case that
    # was "local" (area=10 <= 20) under the default config now fits NEITHER
    # band (10 > 5, and still < regional's min_area_cells=21) -> unclassified.
    custom["local"] = {"max_area_cells": 5, "max_duration_sols": 3}
    custom_path = tmp_path / "custom_storm_taxonomy.yaml"
    import yaml

    custom_path.write_text(yaml.safe_dump(custom), encoding="utf-8")
    loaded = load_storm_taxonomy_config(custom_path)

    default_cls = classify_episode(peak_area_cells=10, duration_sols=2, longitude_sectors_spanned=1, config=config)
    custom_cls = classify_episode(peak_area_cells=10, duration_sols=2, longitude_sectors_spanned=1, config=loaded)
    assert default_cls == "local"
    assert custom_cls == "unclassified"
