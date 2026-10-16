"""Tests for the machine-readable label-specification JSON generator."""
import json

from martian.labeling.label_spec import build_label_specification, write_label_specification_json


def test_specification_contains_all_labels():
    spec = build_label_specification()
    labels = spec["labels"]
    assert set(labels.keys()) == {
        "1_dust_anomaly",
        "2_atmospheric_anomaly",
        "3_storm_episode",
        "4_horizon_targets",
    }


def test_specification_records_exact_thresholds():
    spec = build_label_specification()
    assert spec["thresholds"]["tau_primary"] == 2.0
    assert spec["thresholds"]["tau_sensitivity"] == [1.5, 2.0, 2.5]
    assert spec["thresholds"]["persistence_sols_primary"] == 2
    assert spec["thresholds"]["min_area_cells_primary"] == 3
    assert spec["thresholds"]["sensitivity"]["persistence_sols"] == [1, 2, 3]
    assert spec["thresholds"]["sensitivity"]["min_area_cells"] == [1, 3, 5]


def test_specification_records_sidedness():
    spec = build_label_specification()
    assert spec["labels"]["1_dust_anomaly"]["sidedness"] == "one_sided_high"


def test_write_label_specification_json_roundtrip(tmp_path):
    path = write_label_specification_json(tmp_path / "sub" / "label_specification.json")
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["thresholds"]["tau_primary"] == 2.0
    assert "4_horizon_targets" in loaded["labels"]
