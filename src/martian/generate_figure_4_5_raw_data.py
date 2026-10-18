"""One-off real-data extraction for Figure 4.5 (raw-vs-QC'd preprocessing
comparison): loads the full real EMIRS archive (mirroring
mars/run_all.py's Step 3 exactly -- same provider, same QC config, same
grid), then narrows to ONE representative cell and writes its raw (pre-QC)
and QC'd retrievals to a small CSV that evaluation/preprocessing_figures.py
consumes -- so the ~15-minute full-archive load only has to happen once,
not on every figure-regeneration run.

Run manually:  python generate_figure_4_5_raw_data.py
"""
from __future__ import annotations

from pathlib import Path

from data_sources.real_emirs_provider import RealEMIRSObservationProvider
from preprocessing.grid import assign_cells, load_primary_grid
from preprocessing.quality_control import load_qc_config, run_quality_control

REPRESENTATIVE_CELL_ID = 1393  # see evaluation/preprocessing_figures.py::REPRESENTATIVE_CELL_ID
OUT_DIR = Path("outputs/real/run_20261004T080014968911Z_seed42/figures/preprocessing")
OUT_PATH = OUT_DIR / f"raw_and_qc_retrievals_cell_{REPRESENTATIVE_CELL_ID}.csv"


def main() -> None:
    print("Loading real EMIRS archive (this mirrors mars/run_all.py Step 3, ~15 min)...")
    provider = RealEMIRSObservationProvider()
    raw = provider.load(mars_year_range=(36, 37), seed=42)
    print(f"Loaded {len(raw)} raw retrievals.")

    grid = load_primary_grid()
    raw_gridded = assign_cells(raw, grid)

    cell_subset = raw_gridded.loc[raw_gridded["cell_id"] == REPRESENTATIVE_CELL_ID].reset_index(drop=True)
    print(f"Representative cell {REPRESENTATIVE_CELL_ID}: {len(cell_subset)} raw retrievals.")

    # QC rules are all per-row or local-to-the-given-frame (duplicate
    # detection uses df.duplicated on key columns that always include
    # geolocation, so restricting to one cell's rows before running QC
    # gives IDENTICAL per-row accept/reject outcomes as running QC on the
    # full 1.38M-row archive first and filtering after -- see
    # CHAPTER_4_PARAMETER_VALUES.md's Figure 4.5 writeup for the check).
    qc_report = run_quality_control(cell_subset, load_qc_config())
    print(qc_report.summary())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qc_report.df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(qc_report.df)} rows, qc_status + qc_rejection_rule columns included).")


if __name__ == "__main__":
    main()
