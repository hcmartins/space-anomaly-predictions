"""A2 ("capacity_matched_unimodal", configs/ablation.yaml): a single-domain
model with NO cross-domain component at all (no attention, no alignment,
no auxiliary embedding), but widened so its trainable parameter count
matches that domain's own real share of the full A7 fusion model's
trainable parameters (projection + attention scorer + head for that
task) -- isolating "does fusion help, or would the same parameter budget
help just as much spent on a bigger unimodal model?" (the config's own
"pure capacity effect" framing).

ENGINEERING DECISION: configs/ablation.yaml gives no formula for exactly
how to "widen" A1 into A2 -- only the target property (matched trainable
count). This module widens via one extra hidden layer in the head
(Linear -> Tanh -> Linear) rather than, e.g., enlarging shared_dim,
because shared_dim is Eq. (5.2)'s own fixed cross-domain embedding
contract; changing it would conflate the capacity axis with the
projection-dimensionality axis, two different things. The hidden layer's
width is solved directly from the real Linear-layer parameter-count
formula (not searched), so the resulting count is deterministic and
verifiable, not merely "close enough"."""
from __future__ import annotations

import torch
from torch import nn

from encoders.frozen_encoder import FrozenSequenceEncoder


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _solve_head_hidden_size(*, shared_dim: int, target_head_param_count: int) -> int:
    """Smallest `hidden` such that a Linear(shared_dim, hidden) + Tanh +
    Linear(hidden, 1) head has >= target_head_param_count parameters.
    Params = hidden*(shared_dim + 1) + hidden + (hidden + 1)
           = hidden*(shared_dim + 2) + 1."""
    if target_head_param_count <= 1:
        return 1
    hidden = -(-(target_head_param_count - 1) // (shared_dim + 2))  # ceil division
    return max(1, hidden)


class CapacityMatchedUnimodalModel(nn.Module):
    def __init__(self, encoder: FrozenSequenceEncoder, *, shared_dim: int, target_param_count: int):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Linear(encoder.hidden_size, shared_dim)
        projection_param_count = _count_params(self.projection)
        target_head_param_count = max(0, target_param_count - projection_param_count)
        self.head_hidden_size = _solve_head_hidden_size(shared_dim=shared_dim, target_head_param_count=target_head_param_count)
        self.head = nn.Sequential(nn.Linear(shared_dim, self.head_hidden_size), nn.Tanh(), nn.Linear(self.head_hidden_size, 1))
        self.target_param_count = target_param_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.projection(h)
        return torch.sigmoid(self.head(z)).squeeze(-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())


def domain_share_of_trainable_params(*, projection: nn.Module, scorer, head: nn.Module) -> int:
    """A7's own real per-domain trainable-parameter share -- projection +
    attention scorer + head for ONE task -- the target A2's model is
    widened to match. Not the whole fused model's total, since A2 has no
    cross-domain component to compare against the OTHER domain's own
    parameters."""
    return _count_params(projection) + _count_params(scorer) + _count_params(head)
