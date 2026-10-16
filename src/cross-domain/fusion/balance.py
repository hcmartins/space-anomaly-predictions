"""Balance regularization (Eq. 5.7):

Omega_bal = sum_{t in {S,M}} D_KL(alpha_bar_t || u),  u = (1/2, 1/2)

[CONFIRM resolved]: implemented literally as the stated KL divergence to
uniform -- not the negative-entropy or squared-error alternatives the
thesis flags only as fallbacks if the real implementation used something
else."""
from __future__ import annotations

import torch


def balance_regularizer(alpha: torch.Tensor) -> torch.Tensor:
    """alpha: (batch, 2) attention weights for ONE task. Returns
    D_KL(alpha_bar || uniform) for that task; the total Omega_bal in
    Eq. (5.7) is the SUM of this over both tasks' own attention weights,
    computed by the caller (fusion/model.py)."""
    eps = 1e-12
    alpha_bar = alpha.mean(dim=0).clamp(min=eps)   # (2,)
    uniform = torch.full_like(alpha_bar, 0.5)
    return torch.sum(alpha_bar * torch.log(alpha_bar / uniform))
