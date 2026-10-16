"""The combined cross-domain fusion model -- Section 5.4.1's formal
definition end to end: frozen encoders -> projections (Eq. 5.2) -> per-task
attention/fusion (Eq. 5.5-5.6) -> task heads, plus the alignment (Eq. 5.4)
and balance (Eq. 5.7) loss terms needed for the total objective (Eq. 5.8).
"""
from __future__ import annotations

import torch
from torch import nn

from encoders.frozen_encoder import FrozenSequenceEncoder

from .alignment import moment_alignment_loss
from .attention import AttentionScorer, fuse
from .balance import balance_regularizer
from .projections import DomainProjection

_OTHER_DOMAIN_EMBEDDING_MODES = ("real", "moment_matched_noise", "within_batch_shuffled")


def _transform_other(proj: torch.Tensor, mode: str) -> torch.Tensor:
    """Realizes configs/ablation.yaml's `other_domain_embedding` axis (A3/A4's
    controls, Section 5.6.1): what the OTHER domain's projected embedding
    looks like before it reaches `fuse()`.

    "real" (A6/A7, the default): unchanged.
    "moment_matched_noise" (A3): real Gaussian noise, matched per-dimension
    mean/std to this batch's own real embedding -- isolates whether the
    fused pathway benefits from the auxiliary signal's real CONTENT, or
    merely its presence/regularizing effect (the config's own framing).
    "within_batch_shuffled" (A4): a row permutation of the real embedding
    -- the exact empirical marginal distribution is preserved, only the
    (already nonexistent, per Section 5.5's unpaired-batches design)
    row-level association is further scrambled -- the config's own
    "sharpest control" description.
    """
    if mode == "real":
        return proj
    if mode == "moment_matched_noise":
        mean = proj.mean(dim=0, keepdim=True)
        std = proj.std(dim=0, keepdim=True).clamp_min(1e-6)
        return torch.randn_like(proj) * std + mean
    if mode == "within_batch_shuffled":
        perm = torch.randperm(proj.shape[0])
        return proj[perm]
    raise ValueError(f"Unknown other_domain_embedding mode {mode!r}, expected one of {_OTHER_DOMAIN_EMBEDDING_MODES}")


class CrossDomainFusionModel(nn.Module):
    def __init__(
        self,
        *,
        solar_encoder: FrozenSequenceEncoder,
        martian_encoder: FrozenSequenceEncoder,
        shared_dim: int,
        attention_hidden_size: int,
        alignment_variance_weight: float = 1.0,
    ):
        super().__init__()
        self.solar_encoder = solar_encoder
        self.martian_encoder = martian_encoder
        self.projection_solar = DomainProjection(input_dim=solar_encoder.hidden_size, shared_dim=shared_dim)
        self.projection_martian = DomainProjection(input_dim=martian_encoder.hidden_size, shared_dim=shared_dim)
        # One attention scorer PER TASK (Eq. 5.5's phi_t) -- two tasks, t in {S, M}.
        self.scorer_solar_task = AttentionScorer(embedding_dim=shared_dim, hidden_size=attention_hidden_size)
        self.scorer_martian_task = AttentionScorer(embedding_dim=shared_dim, hidden_size=attention_hidden_size)
        self.head_solar = nn.Linear(shared_dim, 1)
        self.head_martian = nn.Linear(shared_dim, 1)
        self.alignment_variance_weight = alignment_variance_weight

        # Fixed-prototype fallback for the missing domain at inference
        # (configs/fusion.yaml's inference_missing_domain_policy) --
        # populated by training/train_fusion.py at the end of Stage 2;
        # zeros until then (predict_*_only is not meaningful before that).
        self.register_buffer("solar_prototype", torch.zeros(shared_dim))
        self.register_buffer("martian_prototype", torch.zeros(shared_dim))

    def trainable_parameters(self):
        """theta_fus (Eq. 5.8): projections + attention scorers + heads.
        Excludes the frozen encoders by construction (their parameters
        have requires_grad=False -- see encoders/frozen_encoder.py)."""
        return [p for p in self.parameters() if p.requires_grad]

    def project(self, x_solar: torch.Tensor, x_martian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_solar = self.solar_encoder(x_solar)
        h_martian = self.martian_encoder(x_martian)
        return self.projection_solar(h_solar), self.projection_martian(h_martian)

    def forward(
        self,
        x_solar: torch.Tensor,
        x_martian: torch.Tensor,
        *,
        use_fusion_pathway: bool = True,
        other_domain_embedding: str = "real",
    ) -> dict:
        """Full Eq. (5.1)-(5.7) forward pass for one (solar batch, martian
        batch) pair. The batches are independently sampled and UNPAIRED
        (Section 5.5) -- there is no per-row correspondence between them;
        alignment/balance operate on batch-level statistics only.

        `use_fusion_pathway`/`other_domain_embedding` realize
        configs/ablation.yaml's A1-A7 axes (Section 5.6.1) -- both default
        to A7's ("full_model") exact behavior, so every existing caller
        (run_fusion.py, this module's own tests) is unaffected."""
        h_solar_proj, h_martian_proj = self.project(x_solar, x_martian)

        # alignment_loss is always computed from the real projections,
        # regardless of use_fusion_pathway -- A5 ("alignment_only") needs
        # it active even though the fusion pathway itself is skipped.
        alignment_loss = moment_alignment_loss(h_solar_proj, h_martian_proj, variance_weight=self.alignment_variance_weight)

        if not use_fusion_pathway:
            # A1/A2/A5: heads read their own-domain projection directly --
            # no cross-domain mixing, no attention weights to report or to
            # regularize toward uniform.
            z_solar, z_martian = h_solar_proj, h_martian_proj
            alpha_solar = None
            alpha_martian = None
            balance_loss = torch.tensor(0.0)
        else:
            # fuse()'s (own, other) argument order means each call's alpha
            # columns are [own, other] -- normalize both to a fixed
            # [alpha_solar, alpha_martian] column order here so downstream
            # consumers (evaluation/failure_cases.py, callers in general)
            # never need to know which task's own-domain order was used.
            other_for_solar = _transform_other(h_martian_proj, other_domain_embedding)
            other_for_martian = _transform_other(h_solar_proj, other_domain_embedding)
            z_solar, alpha_solar = fuse(h_solar_proj, other_for_solar, self.scorer_solar_task)
            z_martian, alpha_martian_own_first = fuse(h_martian_proj, other_for_martian, self.scorer_martian_task)
            alpha_martian = alpha_martian_own_first[:, [1, 0]]   # -> [alpha_solar, alpha_martian]
            balance_loss = balance_regularizer(alpha_solar) + balance_regularizer(alpha_martian)

        y_hat_solar = torch.sigmoid(self.head_solar(z_solar)).squeeze(-1)
        y_hat_martian = torch.sigmoid(self.head_martian(z_martian)).squeeze(-1)

        return {
            "y_hat_solar": y_hat_solar,
            "y_hat_martian": y_hat_martian,
            "alpha_solar": alpha_solar,
            "alpha_martian": alpha_martian,
            "alignment_loss": alignment_loss,
            "balance_loss": balance_loss,
            "h_solar_proj": h_solar_proj,
            "h_martian_proj": h_martian_proj,
        }

    def predict_solar_only(self, x_solar: torch.Tensor) -> torch.Tensor:
        """Domain-only fallback (Section 5.4's graceful degradation), using
        the stored fixed prototype for the unavailable martian embedding."""
        h_solar_proj = self.projection_solar(self.solar_encoder(x_solar))
        martian_proto = self.martian_prototype.unsqueeze(0).expand(h_solar_proj.shape[0], -1)
        z, _alpha = fuse(h_solar_proj, martian_proto, self.scorer_solar_task)
        return torch.sigmoid(self.head_solar(z)).squeeze(-1)

    def predict_martian_only(self, x_martian: torch.Tensor) -> torch.Tensor:
        h_martian_proj = self.projection_martian(self.martian_encoder(x_martian))
        solar_proto = self.solar_prototype.unsqueeze(0).expand(h_martian_proj.shape[0], -1)
        z, _alpha = fuse(h_martian_proj, solar_proto, self.scorer_martian_task)   # own=martian, other=solar prototype
        return torch.sigmoid(self.head_martian(z)).squeeze(-1)
