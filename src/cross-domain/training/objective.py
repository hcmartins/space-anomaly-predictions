"""Total fusion-stage objective (Eq. 5.8):

L = L_S + L_M + lambda_align * L_align + lambda_bal * Omega_bal + lambda_wd * ||theta_fus||^2
"""
from __future__ import annotations

import torch


def weighted_binary_cross_entropy(y_prob: torch.Tensor, y_true: torch.Tensor, w_pos: float, w_neg: float) -> torch.Tensor:
    """L_S: Chapter 3.7's weighted BCE formula, re-implemented locally here
    (see README.md's decoupling decision) -- identical to
    src/solar/modeling/sequence_models.py::weighted_binary_cross_entropy."""
    eps = 1e-7
    p = y_prob.clamp(eps, 1 - eps)
    loss = -(w_pos * y_true * torch.log(p) + w_neg * (1 - y_true) * torch.log(1 - p))
    return loss.mean()


def compute_class_weights(y: torch.Tensor) -> tuple[float, float]:
    """w+ = N/(2*N+), w- = N/(2*N-) -- Chapter 3.7's formula, train split only."""
    n = len(y)
    n_pos = int((y == 1).sum().item())
    n_neg = n - n_pos
    w_pos = n / (2 * n_pos) if n_pos > 0 else 0.0
    w_neg = n / (2 * n_neg) if n_neg > 0 else 0.0
    return w_pos, w_neg


def martian_task_loss(y_prob: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """L_M: CONFIRMED against Chapter 5 Section 5.4.1's own text ("the
    Martian storm-detection task uses unweighted binary cross-entropy"),
    obtained 2026-08-05. Plain (unweighted) binary cross-entropy -- never
    the same weighted-BCE formula Chapter 3.7 specifies for solar, which
    remains deliberately different per the thesis's own wording."""
    eps = 1e-7
    p = y_prob.clamp(eps, 1 - eps)
    return -(y_true * torch.log(p) + (1 - y_true) * torch.log(1 - p)).mean()


def weight_decay_penalty(parameters) -> torch.Tensor:
    total = None
    for p in parameters:
        term = (p ** 2).sum()
        total = term if total is None else total + term
    return total if total is not None else torch.tensor(0.0)


def total_objective(
    forward_output: dict,
    y_solar: torch.Tensor,
    y_martian: torch.Tensor,
    *,
    solar_class_weights: tuple[float, float],
    trainable_parameters,
    lambda_align: float,
    lambda_bal: float,
    lambda_wd: float,
) -> dict:
    """Returns every term separately (not just the combined sum) so
    training/train_fusion.py can log each one individually -- never
    collapse diagnostic information into a single opaque number."""
    w_pos, w_neg = solar_class_weights
    loss_solar = weighted_binary_cross_entropy(forward_output["y_hat_solar"], y_solar, w_pos, w_neg)
    loss_martian = martian_task_loss(forward_output["y_hat_martian"], y_martian)
    wd = weight_decay_penalty(trainable_parameters)

    total = (
        loss_solar
        + loss_martian
        + lambda_align * forward_output["alignment_loss"]
        + lambda_bal * forward_output["balance_loss"]
        + lambda_wd * wd
    )
    return {
        "total": total,
        "loss_solar": loss_solar,
        "loss_martian": loss_martian,
        "alignment_loss": forward_output["alignment_loss"],
        "balance_loss": forward_output["balance_loss"],
        "weight_decay_penalty": wd,
    }
