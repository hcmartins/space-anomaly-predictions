"""Hand-computed tests for the total fusion objective (Eq. 5.8)."""
import numpy as np
import torch

from training.objective import (
    compute_class_weights,
    martian_task_loss,
    total_objective,
    weight_decay_penalty,
    weighted_binary_cross_entropy,
)


def test_compute_class_weights_hand_computed():
    y = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])   # 4 negatives, 2 positives
    w_pos, w_neg = compute_class_weights(y)
    assert w_pos == 6 / (2 * 2)
    assert w_neg == 6 / (2 * 4)


def test_weighted_binary_cross_entropy_hand_computed():
    y_prob = torch.tensor([0.8, 0.3])
    y_true = torch.tensor([1.0, 0.0])
    loss = weighted_binary_cross_entropy(y_prob, y_true, w_pos=2.0, w_neg=0.5)
    expected = -(2.0 * np.log(0.8) + 0.5 * np.log(0.7)) / 2
    assert abs(loss.item() - expected) < 1e-6


def test_martian_task_loss_is_plain_unweighted_bce():
    y_prob = torch.tensor([0.8, 0.3])
    y_true = torch.tensor([1.0, 0.0])
    loss = martian_task_loss(y_prob, y_true)
    expected = -(np.log(0.8) + np.log(0.7)) / 2
    assert abs(loss.item() - expected) < 1e-6


def test_weight_decay_penalty_hand_computed():
    params = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    penalty = weight_decay_penalty(params)
    assert abs(penalty.item() - (1.0 + 4.0 + 9.0)) < 1e-6


def test_weight_decay_penalty_empty_parameters_returns_zero():
    penalty = weight_decay_penalty([])
    assert penalty.item() == 0.0


def test_total_objective_combines_all_terms_with_correct_weights():
    forward_output = {
        "y_hat_solar": torch.tensor([0.7, 0.7]),
        "y_hat_martian": torch.tensor([0.6, 0.6]),
        "alignment_loss": torch.tensor(2.0),
        "balance_loss": torch.tensor(3.0),
    }
    y_solar = torch.tensor([1.0, 0.0])
    y_martian = torch.tensor([1.0, 0.0])
    params = [torch.tensor([1.0])]

    result = total_objective(
        forward_output, y_solar, y_martian,
        solar_class_weights=(1.0, 1.0), trainable_parameters=params,
        lambda_align=0.5, lambda_bal=0.1, lambda_wd=0.01,
    )
    expected_total = (
        result["loss_solar"].item() + result["loss_martian"].item()
        + 0.5 * 2.0 + 0.1 * 3.0 + 0.01 * 1.0
    )
    assert abs(result["total"].item() - expected_total) < 1e-5
