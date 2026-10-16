"""Generates the machine-readable label-specification JSON: every formula,
threshold, and config value backing the labels defined in
labeling/anomaly_labels.py, in one place so a reviewer never has to
reverse-engineer a threshold from code. Dust-only -- no real climatological
source exists for surface_temperature_k.
"""
from __future__ import annotations

import json
from pathlib import Path

from .anomaly_labels import load_anomaly_definition_config
from .storm_taxonomy import load_storm_taxonomy_config


def build_label_specification(
    config: dict | None = None,
    storm_taxonomy_config: dict | None = None,
) -> dict:
    config = config or load_anomaly_definition_config()
    storm_taxonomy_config = storm_taxonomy_config or load_storm_taxonomy_config()

    return {
        "z_score": {
            "formula": "z_v(c,s) = (x_v(c,s) - mu_v(c,s)) / max(sigma_v(c,s), epsilon)",
            "epsilon": config["epsilon"],
            "mu_sigma_source": "MCDClimatologyProvider.query() (real, built from Montabone et al. dust climatology data)",
        },
        "labels": {
            "1_dust_anomaly": {
                "definition": "z_dust(c,s) > tau",
                "sidedness": config["dust_exceedance"],
            },
            "2_atmospheric_anomaly": {
                "definition": "dust_anomaly(c,s)",
            },
            "3_storm_episode": {
                "definition": (
                    "atmospheric_anomaly(c,s) cell-sols grouped into spatiotemporal "
                    "episodes (8-neighbour spatial connectivity with longitude "
                    "wraparound, plus same-cell/spatial-neighbour temporal linking "
                    "across |delta_sol|<=1, within one mars_year); a cell-sol's "
                    "storm_episode label is True iff its episode's duration_sols >= "
                    "persistence_sols AND peak_area_cells >= min_area_cells"
                ),
                "connectivity": config["connectivity"],
                "longitude_wraparound": config["longitude_wraparound"],
            },
            "4_horizon_targets": {
                "definition": config["horizon_definition"].strip(),
                "horizons_sols": config["horizons_sols"],
            },
        },
        "thresholds": {
            "tau_primary": config["tau_primary"],
            "tau_sensitivity": config["tau_sensitivity"],
            "persistence_sols_primary": config["persistence_sols_primary"],
            "min_area_cells_primary": config["min_area_cells_primary"],
            "primary_regime_note": (
                "Primary regime (persistence>=2 sols, min_area>=3 cells) is the "
                "examiner-suggested stricter gate, promoted from sensitivity to "
                "primary once real data was available (2026-08-05) -- see "
                "configs/anomaly_definition.yaml's own history comment. The "
                "earlier, maximally-permissive controlled-simulation primary "
                "(persistence>=1 sol, min_area>=1 cell -- under which storm_episode "
                "was definitionally identical to atmospheric_anomaly) is now the "
                "permissive end of the sensitivity sweep instead."
            ),
            "sensitivity": config["sensitivity"],
        },
        "storm_taxonomy": {
            "classification_target": storm_taxonomy_config["classification_target"],
            "note": (
                "Descriptive metadata only, attached to qualifying storm episodes "
                "for reporting -- NEVER the machine-learning target."
            ),
            "local": storm_taxonomy_config["local"],
            "regional": storm_taxonomy_config["regional"],
            "planet_encircling": storm_taxonomy_config["planet_encircling"],
        },
        "source_configs": ["configs/anomaly_definition.yaml", "configs/storm_taxonomy.yaml"],
    }


def write_label_specification_json(path: Path, config: dict | None = None, storm_taxonomy_config: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = build_label_specification(config=config, storm_taxonomy_config=storm_taxonomy_config)
    path.write_text(json.dumps(spec, indent=2, sort_keys=False), encoding="utf-8")
    return path
