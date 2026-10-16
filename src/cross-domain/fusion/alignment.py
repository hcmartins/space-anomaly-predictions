"""Moment-matching alignment loss (Eq. 5.3-5.4):

mu_hat_D = (1/n_D) sum_i h_tilde_D^(i)
sigma_hat_D^2 = (1/n_D) sum_i (h_tilde_D^(i) - mu_hat_D)^2          (Eq. 5.3)

L_align = (1/d)||mu_hat_S - mu_hat_M||^2 + (beta/d)||sigma_hat_S^2 - sigma_hat_M^2||^2   (Eq. 5.4)

[CONFIRM resolved]: Eq. (5.3) is explicitly the BIASED (1/n_D, not
1/(n_D-1)) variance estimator -- implemented literally as written, not a
free engineering choice. No covariance-matching term (Eq. 5.4 constrains
only the marginal mean/variance, per the thesis's own text)."""
from __future__ import annotations

import torch


def batch_moments(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """h: (batch, d). Returns (mean, variance), each shape (d,)."""
    mean = h.mean(dim=0)
    variance = ((h - mean) ** 2).mean(dim=0)
    return mean, variance


def moment_alignment_loss(h_solar: torch.Tensor, h_martian: torch.Tensor, *, variance_weight: float = 1.0) -> torch.Tensor:
    mu_solar, var_solar = batch_moments(h_solar)
    mu_martian, var_martian = batch_moments(h_martian)
    d = h_solar.shape[1]

    mean_term = torch.sum((mu_solar - mu_martian) ** 2) / d
    variance_term = torch.sum((var_solar - var_martian) ** 2) / d
    return mean_term + variance_weight * variance_term
