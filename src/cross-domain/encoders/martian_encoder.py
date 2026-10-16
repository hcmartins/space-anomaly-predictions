"""The real martian storm-detection LSTM encoder (Chapter 5's f_M), loaded
from src/martian's real checkpoint -- trained on real EMIRS/MCD data via
`mars.run_all` (outputs/real/run_20260813T195013457439Z_seed42, promoted
2026-08-14; supersedes run_20260805T115705602822Z_seed42, whose val split
was empty -- see src/martian/CHAPTER_4_PARAMETER_VALUES.md Section 13),
not simulated data. `load_martian_encoder` still returns a provenance
string alongside the encoder (the mechanism is domain-agnostic, not
simulation-specific) so callers always have an explicit, checkable record
of which run produced the checkpoint they're using.
"""
from __future__ import annotations

from pathlib import Path

from .frozen_encoder import FrozenSequenceEncoder, load_frozen_encoder
from .solar_encoder import CROSS_DOMAIN_ROOT, load_encoders_config


def load_martian_encoder(config: dict | None = None) -> tuple[FrozenSequenceEncoder, str]:
    """Returns (encoder, data_provenance) -- the provenance string is never
    optional here, unlike solar's loader, precisely because it carries a
    real caveat that must not be dropped by a careless caller."""
    config = (config or load_encoders_config())["martian"]
    checkpoint_path = (CROSS_DOMAIN_ROOT / config["checkpoint_path"]).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Real martian checkpoint not found at {checkpoint_path}. Run "
            "`python -m mars.run_all --config configs/run.yaml` from src/martian "
            "first -- this package never fabricates a substitute."
        )
    encoder = load_frozen_encoder(
        checkpoint_path,
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        state_dict_prefix=config["state_dict_prefix"],
        checkpoint_format=config["checkpoint_format"],
    )
    return encoder, config["data_provenance"]
