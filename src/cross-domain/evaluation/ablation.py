"""The A1-A7 ablation registry (Section 5.6.1, Table 5.2). Structural in
this delivery: loads configs/ablation.yaml's definitions; does not launch
seven real training runs (see README.md's scope notes)."""
from __future__ import annotations

from pathlib import Path

import yaml

CROSS_DOMAIN_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = CROSS_DOMAIN_ROOT / "configs"


def load_ablation_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / "ablation.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def get_configuration(ablation_id: str, config: dict | None = None) -> dict:
    config = config or load_ablation_config()
    for entry in config["configurations"]:
        if entry["id"] == ablation_id:
            return entry
    raise KeyError(f"No ablation configuration with id {ablation_id!r} in configs/ablation.yaml")


def decisive_comparisons(config: dict | None = None) -> list[list[str]]:
    config = config or load_ablation_config()
    return config["decisive_comparisons"]
