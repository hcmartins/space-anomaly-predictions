"""Shared figure-provenance stamping: every evaluation figure gets a figure
number, a descriptive caption, units, sample size, data split, horizon,
data-source label, and script+config provenance -- stamped visibly on the
figure itself AND written to a companion `<name>.provenance.json`, alongside
a companion `<name>.csv` holding the exact data the figure was drawn from.

Figure numbers are assigned from a persistent registry
(`results/figure_registry.json`) keyed by (task, model, horizon, split,
kind), continuing the thesis's existing Chapter 4 figure sequence (Figures
4.3 and 4.7 already exist -- see planning/PREPROCESSING_DOCUMENTATION.md and
configs/storm_taxonomy.yaml) so numbers are stable across re-runs: the same
figure always gets the same number, and a new figure gets the next unused one.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_sources.provenance import stamp_figure

_REGISTRY_PATH_DEFAULT = "results/figure_registry.json"
_FIRST_FIGURE_NUMBER = 8  # continues from the existing Figure 4.7


def _load_registry(registry_path: Path) -> dict:
    if registry_path.exists():
        return json.loads(registry_path.read_text(encoding="utf-8"))
    return {}


def _save_registry(registry_path: Path, registry: dict) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def assign_figure_number(key: str, *, registry_path: Path | None = None, chapter: int = 4) -> str:
    """Returns a stable "4.N" figure number for `key`, assigning the next
    unused N the first time `key` is seen and reusing it on every
    subsequent call (re-running the pipeline never renumbers a figure)."""
    registry_path = registry_path or Path(_REGISTRY_PATH_DEFAULT)
    registry = _load_registry(registry_path)
    if key in registry:
        return registry[key]

    used_numbers = [int(v.split(".")[1]) for v in registry.values() if v.startswith(f"{chapter}.")]
    next_number = max(used_numbers, default=_FIRST_FIGURE_NUMBER - 1) + 1
    figure_id = f"{chapter}.{next_number}"
    registry[key] = figure_id
    _save_registry(registry_path, registry)
    return figure_id


@dataclass
class FigureProvenance:
    figure_number: str
    caption: str
    units: str
    n: int
    split: str
    horizon_sols: int
    data_source: str
    script: str
    config_paths: list = field(default_factory=list)
    source_csv: str | None = None


def save_figure_with_provenance(
    fig,
    base_path: Path,
    provenance: FigureProvenance,
    *,
    source_df: pd.DataFrame | None = None,
) -> dict:
    """Saves `fig` to `<base_path>.png`, `source_df` (if given) to
    `<base_path>.csv`, and the full provenance record to
    `<base_path>.provenance.json`. Also stamps a compact, human-readable
    caption block directly onto the figure so the PNG is self-describing
    even without the JSON."""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = base_path.with_suffix(".png")
    csv_path = base_path.with_suffix(".csv")
    json_path = base_path.with_suffix(".provenance.json")

    caption_block = (
        f"Figure {provenance.figure_number}: {provenance.caption}\n"
        f"Units: {provenance.units} | n={provenance.n} | split={provenance.split} | "
        f"horizon={provenance.horizon_sols}sol\n"
        f"Data source: {provenance.data_source} | Script: {provenance.script}"
    )
    fig.text(0.01, 0.01, caption_block, fontsize=6.5, color="dimgray", ha="left", va="bottom")
    stamp_figure(fig, provenance.data_source)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # figures are created faster than any GUI/event loop would
    # reclaim them here -- across a full multi-task, multi-horizon,
    # multi-model run this is dozens of figures; leaving them open triggers
    # matplotlib's "More than 20 figures" RuntimeWarning and leaks memory.

    provenance_dict = asdict(provenance)
    if source_df is not None:
        source_df.to_csv(csv_path, index=False)
        provenance_dict["source_csv"] = str(csv_path)
    json_path.write_text(json.dumps(provenance_dict, indent=2, default=str), encoding="utf-8")

    return {"png": png_path, "csv": csv_path if source_df is not None else None, "provenance": json_path}
