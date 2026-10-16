"""Frozen, checkpoint-only sequence encoders (Chapter 5, Eq. 5.1: f_S, f_M).

A minimal, inference-only LSTM matching the shape of both src/solar's and
src/martian's real SequenceModel classes (LSTM -> take the last time
step). Deliberately reconstructed here rather than imported from either
package -- see README.md's "Why encoders are self-contained" section.
Always frozen: parameters have requires_grad=False and the module stays
in eval() mode throughout, matching Chapter 5's own description --
"the encoder parameters theta_S and theta_M are frozen throughout fusion
training... the encoders contribute no trainable parameters to the fusion
stage."
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class FrozenSequenceEncoder(nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns h = the final LSTM hidden state, shape (batch, hidden_size)."""
        with torch.no_grad():
            out, _ = self.lstm(x)
            return out[:, -1, :]


def _load_lstm_state_dict(encoder: FrozenSequenceEncoder, raw_state_dict: dict, prefix: str) -> None:
    """Loads only the `prefix`-matching LSTM weights from a real checkpoint
    (ignoring that domain's own task head -- Chapter 5's fusion stage never
    uses it), remapping e.g. `{prefix}weight_ih_l0` -> `lstm.weight_ih_l0`."""
    remapped = {}
    for key, value in raw_state_dict.items():
        if key.startswith(prefix):
            remapped["lstm." + key[len(prefix):]] = value
    missing, _unexpected = encoder.load_state_dict(remapped, strict=False)
    lstm_missing = [k for k in missing if k.startswith("lstm.")]
    if lstm_missing:
        raise ValueError(
            f"Checkpoint is missing expected LSTM weights {lstm_missing} for prefix {prefix!r} "
            "-- confirm configs/encoders.yaml's state_dict_prefix matches the real checkpoint."
        )


def load_frozen_encoder(
    checkpoint_path: Path,
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    state_dict_prefix: str,
    checkpoint_format: str,
) -> FrozenSequenceEncoder:
    """checkpoint_format: "wrapped" ({"state_dict":..., "hyperparameters":...},
    src/solar's real convention) or "raw_state_dict" (a bare state_dict,
    src/martian's real convention) -- see configs/encoders.yaml."""
    if checkpoint_format not in ("wrapped", "raw_state_dict"):
        raise ValueError(f"Unknown checkpoint_format {checkpoint_format!r}")

    encoder = FrozenSequenceEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    checkpoint = torch.load(Path(checkpoint_path), weights_only=True, map_location="cpu")
    raw_state_dict = checkpoint["state_dict"] if checkpoint_format == "wrapped" else checkpoint
    _load_lstm_state_dict(encoder, raw_state_dict, state_dict_prefix)
    encoder.eval()
    return encoder
