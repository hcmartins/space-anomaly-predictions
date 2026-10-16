"""Adaptive weighting: attention scorer, softmax normalization, and the
fused representation (Eq. 5.5-5.6):

a_{t,k} = w_t^T tanh(W_t h_tilde_k + b_t)          (Eq. 5.5, [CONFIRM]-resolved form)
alpha_{t,k} = softmax_k(a_{t,k}),  alpha_{t,S} + alpha_{t,M} = 1
z_t = alpha_{t,S} h_tilde_S + alpha_{t,M} h_tilde_M     (Eq. 5.6)

Non-negativity and sum-to-one are structural (softmax), never a separate
penalty -- matching the thesis's own note."""
from __future__ import annotations

import torch
from torch import nn


class AttentionScorer(nn.Module):
    """One scorer per task t (Eq. 5.5's phi_t) -- a single dense layer with
    tanh, the thesis's own first-listed example form."""

    def __init__(self, *, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(embedding_dim, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (batch, embedding_dim) -> raw score: (batch,)."""
        return self.score(torch.tanh(self.dense(h))).squeeze(-1)


def fuse(own_proj: torch.Tensor, other_proj: torch.Tensor, scorer: AttentionScorer) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (z, alpha) for ONE task's scorer, where `own_proj` is that
    task's own-domain batch (n_own, d) and `other_proj` is the OTHER
    domain's independently-sampled batch (n_other, d).

    ENGINEERING DECISION: Eq. (5.6) is written per-instance, z_t =
    alpha_{t,S} h_tilde_S + alpha_{t,M} h_tilde_M, implicitly assuming one
    h_tilde_M per solar row. But Section 5.5.1's real batch strategy draws
    n_S and n_M INDEPENDENTLY (possibly different sizes, no sample-level
    pairing -- "no such correspondence exists, as the domains are
    unpaired"). The only interpretation compatible with that is that the
    OTHER domain's contribution at a given step is its batch's own
    representative (mean) embedding, broadcast across every row of the
    own-domain batch -- consistent with the alignment loss (Eq. 5.4)
    already operating on batch-level statistics only, never per-sample
    correspondence.
    """
    other_representative = other_proj.mean(dim=0, keepdim=True)   # (1, d) -- this step's batch summary
    other_broadcast = other_representative.expand(own_proj.shape[0], -1)

    score_own = scorer(own_proj)
    score_other = scorer(other_broadcast)
    scores = torch.stack([score_own, score_other], dim=1)   # (n_own, 2)
    alpha = torch.softmax(scores, dim=1)
    z = alpha[:, 0:1] * own_proj + alpha[:, 1:2] * other_broadcast
    return z, alpha
