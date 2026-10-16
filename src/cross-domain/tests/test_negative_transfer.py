"""Hand-computed tests for negative-transfer analysis (Section 5.6.2)."""
import numpy as np

from evaluation.negative_transfer import negative_transfer_by_regime, true_skill_statistic


def test_true_skill_statistic_hand_computed():
    # 4 positives, 4 negatives; perfect classifier at threshold 0.5.
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_prob = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
    tss = true_skill_statistic(y_true, y_prob, threshold=0.5)
    assert tss == 1.0   # POD=1, FAR=0


def test_true_skill_statistic_random_classifier_near_zero():
    y_true = [1, 0, 1, 0]
    y_prob = [0.5, 0.5, 0.5, 0.5]   # threshold 0.5 -> all predicted positive
    tss = true_skill_statistic(y_true, y_prob, threshold=0.5)
    assert tss == 0.0   # POD=1, FAR=1 -> TSS=0


def test_negative_transfer_by_regime_hand_computed():
    y_true = [1, 1, 0, 0, 1, 1, 0, 0]
    regime = ["A", "A", "A", "A", "B", "B", "B", "B"]
    # Regime A: fused perfect, unimodal random -> positive delta.
    y_prob_fused = [0.9, 0.9, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5]
    y_prob_unimodal = [0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.1, 0.1]
    result = negative_transfer_by_regime(
        y_true, y_prob_fused, y_prob_unimodal, regime, threshold_fused=0.5, threshold_unimodal=0.5
    )
    result = result.set_index("regime")
    assert result.loc["A", "delta"] > 0
    assert result.loc["A", "negative_transfer"] == False  # noqa: E712 -- may be numpy bool, not `is`-comparable
    # Regime B: fused random, unimodal perfect -> negative delta.
    assert result.loc["B", "delta"] < 0
    assert result.loc["B", "negative_transfer"] == True  # noqa: E712


def test_negative_transfer_reports_nan_not_fabricated_for_single_class_regime():
    y_true = [0, 0, 0]
    regime = ["C", "C", "C"]
    result = negative_transfer_by_regime(
        y_true, [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], regime, threshold_fused=0.5, threshold_unimodal=0.5
    )
    assert np.isnan(result.loc[0, "delta"])
    assert result.loc[0, "negative_transfer"] is None
