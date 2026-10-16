"""Tests for failure-case cataloguing (Section 5.6.3)."""
import numpy as np
import pandas as pd

from evaluation.failure_cases import catalogue_failure_cases


def test_catalogue_selects_missed_events_and_false_alarms_only():
    y_true = [1, 1, 0, 0, 1, 0]
    y_prob_fused = [0.1, 0.9, 0.9, 0.1, 0.6, 0.2]   # row0: missed event; row2: false alarm; others correct
    y_prob_unimodal = [0.1, 0.9, 0.9, 0.1, 0.6, 0.2]
    alpha = np.array([[0.5, 0.5]] * 6)

    catalogue = catalogue_failure_cases(
        y_true, y_prob_fused, y_prob_unimodal, alpha,
        threshold_fused=0.5, threshold_unimodal=0.5, false_alarm_probability_threshold=0.5,
    )
    assert len(catalogue) == 2
    assert set(catalogue["failure_type"]) == {"missed_event", "false_alarm"}


def test_shared_with_unimodal_true_when_both_models_miss_the_same_event():
    y_true = [1]
    y_prob_fused = [0.1]      # fused misses (below threshold)
    y_prob_unimodal = [0.1]   # unimodal ALSO misses -- a representation problem, not a fusion problem
    alpha = np.array([[0.5, 0.5]])

    catalogue = catalogue_failure_cases(
        y_true, y_prob_fused, y_prob_unimodal, alpha,
        threshold_fused=0.5, threshold_unimodal=0.5, false_alarm_probability_threshold=0.5,
    )
    assert catalogue.iloc[0]["shared_with_unimodal"] == True  # noqa: E712


def test_shared_with_unimodal_false_when_only_fused_misses_candidate_negative_transfer():
    y_true = [1]
    y_prob_fused = [0.1]      # fused misses
    y_prob_unimodal = [0.9]   # unimodal catches it -- introduced by fusion
    alpha = np.array([[0.5, 0.5]])

    catalogue = catalogue_failure_cases(
        y_true, y_prob_fused, y_prob_unimodal, alpha,
        threshold_fused=0.5, threshold_unimodal=0.5, false_alarm_probability_threshold=0.5,
    )
    assert catalogue.iloc[0]["shared_with_unimodal"] == False  # noqa: E712


def test_catalogue_carries_through_metadata_columns():
    y_true = [1]
    y_prob_fused = [0.1]
    y_prob_unimodal = [0.1]
    alpha = np.array([[0.7, 0.3]])
    metadata = pd.DataFrame({"run_id": ["run_001"], "date": ["2020-01-01"], "regime": ["quiet"]})

    catalogue = catalogue_failure_cases(
        y_true, y_prob_fused, y_prob_unimodal, alpha,
        threshold_fused=0.5, threshold_unimodal=0.5, false_alarm_probability_threshold=0.5,
        metadata=metadata,
    )
    assert catalogue.iloc[0]["run_id"] == "run_001"
    assert catalogue.iloc[0]["alpha_solar_at_issuance"] == 0.7


def test_catalogue_empty_when_no_failures():
    y_true = [1, 0]
    y_prob_fused = [0.9, 0.1]
    y_prob_unimodal = [0.9, 0.1]
    alpha = np.array([[0.5, 0.5], [0.5, 0.5]])
    catalogue = catalogue_failure_cases(
        y_true, y_prob_fused, y_prob_unimodal, alpha,
        threshold_fused=0.5, threshold_unimodal=0.5, false_alarm_probability_threshold=0.5,
    )
    assert len(catalogue) == 0
