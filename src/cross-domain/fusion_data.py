"""Shared real-data loading for both run_fusion.py and run_ablation.py:
regenerating (via each domain's own export script, run as a subprocess so
neither solar's nor martian's Python modules are ever imported directly
-- see README.md's "Why encoders are self-contained") and loading the
real sequence-tensor exports neither src/solar nor src/martian persists
on its own. Originally private to run_fusion.py; factored out here
unchanged in behavior so run_ablation.py doesn't duplicate it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from encoders.solar_encoder import CROSS_DOMAIN_ROOT

SOLAR_ROOT = (CROSS_DOMAIN_ROOT / ".." / "solar").resolve()
MARTIAN_ROOT = (CROSS_DOMAIN_ROOT / ".." / "martian").resolve()

# Real GOES+SHARP sequence run (2026-07-29) -- closes the gap between
# Chapter 5's described GOES+SHARP solar input and the earlier GOES-only
# export. See configs/encoders.yaml's solar section for the matching
# checkpoint and the real before/after TSS comparison.
SOLAR_EXPORT_DIR = SOLAR_ROOT / "results" / "real_full_sequence_goes_sharp_20100501_20250724" / "fusion_export"
# Real mars.run_all run against real EMIRS/MCD data (2026-08-13), k=2/m=3
# primary regime, real Table 4.6 catalogue validation, chronological-split
# leakage fix + sentinel-row exclusion + missing-data sensitivity selection
# -- see src/martian/CHAPTER_4_PARAMETER_VALUES.md Section 13. Supersedes
# the 2026-08-05 run (run_20260805T115705602822Z_seed42), whose exported
# val split was empty (the same split bug fixed in that run's Section 13)
# -- see CHAPTER_5_PARAMETER_VALUES.md for the cross-domain before/after.
# This path must always name a run directory that still exists under
# src/martian/outputs/real/.
# 2026-10-04: harmonised labels + masked horizons + MY37-validation split
# (martian_rerun_20261004.md). Supersedes run_20260813T195013457439Z_seed42,
# whose checkpoints are backed up in
# src/martian/checkpoints/task_storm_detection_superseded_run_20260813T195013457439Z/.
MARTIAN_EXPORT_DIR = MARTIAN_ROOT / "outputs" / "real" / "run_20261004T080014968911Z_seed42" / "fusion_export"

FUSION_SEED = 44   # matches this project's existing storm_detection task seed convention

_REQUIRED_EXPORT_FILES = [f"{name}_{split}.npy" for name in ("X", "y") for split in ("train", "val", "test")] + [
    "regime_test.npy"
]


def ensure_export(
    export_dir: Path, *, package_root: Path, script_name: str = "export_fusion_sequences.py",
    script_args: list[str] | None = None,
) -> None:
    if all((export_dir / f).exists() for f in _REQUIRED_EXPORT_FILES):
        return
    script_args = script_args or []
    print(f"--- Exporting real sequence data via {script_name} {' '.join(script_args)} (cwd={package_root}) ---", flush=True)
    subprocess.run([sys.executable, script_name, *script_args], cwd=str(package_root), check=True)
    missing = [f for f in _REQUIRED_EXPORT_FILES if not (export_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"{script_name} did not produce expected files: {missing}")


def ensure_both_exports() -> None:
    ensure_export(
        SOLAR_EXPORT_DIR, package_root=SOLAR_ROOT,
        script_args=[
            "--base-run-name", "real_full_20100501_20250724",
            "--sequence-run-name", "real_full_sequence_goes_sharp_20100501_20250724",
            "--include-sharp",
        ],
    )
    ensure_export(
        MARTIAN_EXPORT_DIR, package_root=MARTIAN_ROOT,
        script_args=["--run-dir", str(MARTIAN_EXPORT_DIR.parent)],
    )


def load_domain_arrays(export_dir: Path) -> dict:
    arrays = {
        f"{name}_{split}": np.load(export_dir / f"{name}_{split}.npy")
        for name in ("X", "y")
        for split in ("train", "val", "test")
    }
    # regime_test may be a string/object array (martian's quartile-bucketed
    # season labels) rather than numeric (solar's calendar years).
    arrays["regime_test"] = np.load(export_dir / "regime_test.npy", allow_pickle=True)
    return arrays


def load_fusion_config(config_path: Path | None = None) -> dict:
    config_path = config_path or CROSS_DOMAIN_ROOT / "configs" / "fusion.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
