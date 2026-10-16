"""Tests for the A1-A7 ablation registry (Table 5.2, Section 5.6.1)."""
import pytest

from evaluation.ablation import decisive_comparisons, get_configuration, load_ablation_config


def test_load_ablation_config_has_all_seven_configurations():
    config = load_ablation_config()
    ids = {entry["id"] for entry in config["configurations"]}
    assert ids == {"A1", "A2", "A3", "A4", "A5", "A6", "A7"}


def test_get_configuration_returns_the_right_entry():
    entry = get_configuration("A7")
    assert entry["name"] == "full_model"
    assert entry["use_fusion_pathway"] is True
    assert entry["use_alignment_loss"] is True


def test_get_configuration_raises_for_unknown_id():
    with pytest.raises(KeyError):
        get_configuration("A99")


def test_a4_is_the_sharpest_control_shuffled_not_noise():
    entry = get_configuration("A4")
    assert entry["other_domain_embedding"] == "within_batch_shuffled"


def test_decisive_comparisons_include_a7_vs_a2_a3_a4():
    comparisons = decisive_comparisons()
    flat_pairs = [c for c in comparisons if len(c) == 2]
    assert ["A7", "A2"] in flat_pairs
    assert ["A7", "A3"] in flat_pairs
    assert ["A7", "A4"] in flat_pairs
