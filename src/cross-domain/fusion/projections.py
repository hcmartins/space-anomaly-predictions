"""Projection into the shared embedding space (Eq. 5.2):
h_tilde_S = P_S h_S + b_S, h_tilde_M = P_M h_M + b_M.

[CONFIRM resolved, configs/fusion.yaml's projection_nonlinearity=null]: no
non-linearity after the affine map -- Eq. (5.2) as written shows a plain
affine transform, no activation function."""
from __future__ import annotations

import torch
from torch import nn


class DomainProjection(nn.Module):
    def __init__(self, *, input_dim: int, shared_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, shared_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)
