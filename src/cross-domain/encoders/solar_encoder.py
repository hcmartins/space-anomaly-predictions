"""The real solar GOES-LSTM encoder (Chapter 5's f_S), loaded from the
real checkpoint trained this session on real GOES data under the real
Cycle 24 (train) / Cycle 25 (test) split -- see
src/solar/planning and configs/encoders.yaml's `solar:` section.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .frozen_encoder import FrozenSequenceEncoder, load_frozen_encoder

CROSS_DOMAIN_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = CROSS_DOMAIN_ROOT / "configs"


def load_encoders_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "encoders.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def load_solar_encoder(config: dict | None = None) -> FrozenSequenceEncoder:
    config = (config or load_encoders_config())["solar"]
    checkpoint_path = (CROSS_DOMAIN_ROOT / config["checkpoint_path"]).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Real solar checkpoint not found at {checkpoint_path}. Run "
            "src/solar/run_pipeline_sequence.py first (see src/solar/README-equivalent "
            "planning docs) -- this package never fabricates a substitute."
        )
    return load_frozen_encoder(
        checkpoint_path,
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        state_dict_prefix=config["state_dict_prefix"],
        checkpoint_format=config["checkpoint_format"],
    )
