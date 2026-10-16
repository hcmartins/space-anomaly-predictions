"""Real, independently-trained unimodal baselines for negative-transfer
(Section 5.6.2) and failure-case (Section 5.6.3) comparison.

`encoders/frozen_encoder.py` deliberately discards each domain's own task
head when loading a checkpoint ("ignoring that domain's own task head --
Chapter 5's fusion stage never uses it"). For a genuine unimodal
comparison we want exactly that head back: this module loads ONLY
`head.weight`/`head.bias` from the SAME real checkpoint files (both real
checkpoints have this layer -- solar's `SequenceModel.head`, martian's
`SequenceModel.head`, confirmed by direct inspection of both `.pt` files)
and applies it to the frozen encoder's ALREADY-COMPUTED hidden state, so no
duplicate LSTM forward pass is needed.

This is more faithful than `CrossDomainFusionModel.predict_solar_only`/
`predict_martian_only`, which use the *fusion*-trained head plus a fixed
prototype for the missing domain -- a fusion-degradation mode, not a true
independently-trained single-domain baseline.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class UnimodalHead(nn.Module):
    """A frozen `nn.Linear(hidden_size, 1)` + sigmoid, loaded directly from
    a real checkpoint's own `head.weight`/`head.bias` -- the same real
    weights that domain's own model used to make its own predictions
    before being repurposed as a fusion encoder."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.linear(h)).squeeze(-1)


def load_unimodal_head(checkpoint_path: Path, *, hidden_size: int, checkpoint_format: str) -> UnimodalHead:
    if checkpoint_format not in ("wrapped", "raw_state_dict"):
        raise ValueError(f"Unknown checkpoint_format {checkpoint_format!r}")

    checkpoint = torch.load(Path(checkpoint_path), weights_only=True, map_location="cpu")
    raw_state_dict = checkpoint["state_dict"] if checkpoint_format == "wrapped" else checkpoint

    head = UnimodalHead(hidden_size=hidden_size)
    missing, _unexpected = head.load_state_dict(
        {"linear.weight": raw_state_dict["head.weight"], "linear.bias": raw_state_dict["head.bias"]},
        strict=False,
    )
    if missing:
        raise ValueError(
            f"Checkpoint is missing expected head weights ({missing}) -- confirm this checkpoint "
            "was produced by a SequenceModel with a `head` Linear layer."
        )
    head.eval()
    return head


def load_unimodal_head_for_domain(domain: str, encoders_config: dict) -> UnimodalHead:
    """Convenience wrapper resolving a domain's checkpoint path/hidden_size/
    checkpoint_format straight out of configs/encoders.yaml (already loaded
    as `encoders_config`) -- shared by run_fusion.py and run_ablation.py so
    this resolution logic exists exactly once."""
    from encoders.solar_encoder import CROSS_DOMAIN_ROOT

    config = encoders_config[domain]
    checkpoint_path = (CROSS_DOMAIN_ROOT / config["checkpoint_path"]).resolve()
    return load_unimodal_head(checkpoint_path, hidden_size=config["hidden_size"], checkpoint_format=config["checkpoint_format"])
