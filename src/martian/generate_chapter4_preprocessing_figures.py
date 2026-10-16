"""Generates real Figures 4.5-4.8 from a completed real run's own saved
outputs -- CHAPTER_4_PARAMETER_VALUES.md open item 3
(evaluation/preprocessing_figures.py implements the actual figure logic;
this script just wires it to one run directory and writes the results with
provenance, via evaluation.figure_provenance.save_figure_with_provenance).

Figure 4.5 additionally needs a representative cell's RAW (pre-QC)
retrievals, which are not saved by the main pipeline run (they would be
~1.4M rows across the whole archive) -- see
generate_figure_4_5_raw_data.py, which must be run once first (it writes
figures/preprocessing/raw_and_qc_retrievals_cell_<id>.csv into the run
directory; this script reads that file back in).

Run manually:
    python generate_figure_4_5_raw_data.py   # once, ~15 min, writes the raw-retrieval CSV
    python generate_chapter4_preprocessing_figures.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from data_sources.real_mcd_provider import RealMCDClimatologyProvider
from data_sources.schema import CONFIG_DIR
from evaluation.figure_provenance import FigureProvenance, save_figure_with_provenance
from evaluation.preprocessing_figures import (
    figure_4_5_raw_qc_comparison,
    figure_4_6_missingness_gapfill_distribution,
    figure_4_7_spatiotemporal_structure,
    figure_4_8_seasonal_series,
    select_representative_cell,
)
from preprocessing.grid import load_primary_grid
from splitting.chronological_split import assign_split, load_splits_config

RUN_DIR = Path("outputs/real/run_20260813T195013457439Z_seed42")
DATA_SOURCE_LABEL = (
    "REAL (Martian Chapter 4 mars.run_all pipeline) -- Figures 4.5-4.8 generated post-hoc "
    "2026-08-12 from this run's own saved outputs, not part of the original 2026-08-05 run"
)


def _load_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "preprocessing_figures.yaml").read_text(encoding="utf-8"))


def main() -> None:
    config = _load_config()
    grid = load_primary_grid()
    splits_config = load_splits_config()

    aggregated = pd.read_csv(RUN_DIR / "data" / "aggregated_cell_sol.csv")
    filled = pd.read_csv(RUN_DIR / "data" / "trailing_filled_cell_sol.csv")
    labeled = pd.read_csv(RUN_DIR / "labels" / "labeled_cell_sol.csv")
    episode_catalogue = pd.read_csv(RUN_DIR / "labels" / "episode_catalogue.csv")

    split_series = assign_split(aggregated, splits_config)
    excluded = tuple(config["representative_cell"]["excluded_cell_ids"])
    representative_cell_id = select_representative_cell(aggregated, split_series, excluded_cell_ids=excluded)
    configured_cell_id = config["representative_cell"]["cell_id"]
    if representative_cell_id != configured_cell_id:
        raise ValueError(
            f"Recomputed representative cell ({representative_cell_id}) does not match "
            f"configs/preprocessing_figures.yaml's recorded value ({configured_cell_id}) -- "
            f"the config comment/selection_rule text would then be describing the wrong cell; "
            f"update the config rather than silently using a different cell."
        )

    out_dir = RUN_DIR / "figures" / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)
    mars_year = splits_config["train_mars_year"]

    # ---- Figure 4.5 ----
    raw_qc_path = out_dir / f"raw_and_qc_retrievals_cell_{representative_cell_id}.csv"
    if raw_qc_path.exists():
        raw_qc_df = pd.read_csv(raw_qc_path)
        agg_cell = aggregated.loc[(aggregated["cell_id"] == representative_cell_id) & (aggregated["mars_year"] == mars_year)]
        filled_cell = filled.loc[(filled["cell_id"] == representative_cell_id) & (filled["mars_year"] == mars_year)]
        fig, source_df = figure_4_5_raw_qc_comparison(
            raw_qc_df, agg_cell, filled_cell, cell_id=representative_cell_id, mars_year=mars_year
        )
        save_figure_with_provenance(
            fig, out_dir / "figure_4_5_raw_qc_comparison",
            FigureProvenance(
                figure_number="4.5",
                caption=(
                    f"Preprocessing comparison for representative cell {representative_cell_id} "
                    f"(MY{mars_year}, training partition): raw per-retrieval CDOD by QC outcome, "
                    f"QC'd and gap-treated cell-sol series, and retained observation count per sol."
                ),
                units="dimensionless CDOD (1075 cm^-1)",
                n=len(raw_qc_df),
                split="train",
                horizon_sols=0,
                data_source=DATA_SOURCE_LABEL,
                script="generate_chapter4_preprocessing_figures.py",
                config_paths=["configs/preprocessing_figures.yaml"],
            ),
            source_df=source_df,
        )
        print(f"Wrote Figure 4.5 (cell {representative_cell_id}, n_raw={len(raw_qc_df)}).")
    else:
        print(f"SKIPPED Figure 4.5: {raw_qc_path} not found -- run generate_figure_4_5_raw_data.py first.")

    # ---- Figure 4.6 ----
    mcd = RealMCDClimatologyProvider()
    sol_min = int(splits_config["train_sol_range"][0])
    sol_max = int(splits_config["train_sol_range"][1])
    fig, source_df = figure_4_6_missingness_gapfill_distribution(
        aggregated, filled, mcd,
        representative_cell_id=representative_cell_id, grid=grid,
        mars_year=mars_year, sol_min=sol_min, sol_max=sol_max,
    )
    save_figure_with_provenance(
        fig, out_dir / "figure_4_6_missingness_gapfill_distribution",
        FigureProvenance(
            figure_number="4.6",
            caption=(
                f"Column-dust-optical-depth preprocessing, MY{mars_year} training partition: "
                f"cell-sol missingness map and trailing-fill map (latitude band containing cell "
                f"{representative_cell_id}), and pre/post-fill CDOD distribution against the "
                f"Montabone gridded climatology mean."
            ),
            units="dimensionless CDOD",
            n=int(len(aggregated.loc[aggregated['mars_year'] == mars_year])),
            split="train",
            horizon_sols=0,
            data_source=DATA_SOURCE_LABEL,
            script="generate_chapter4_preprocessing_figures.py",
            config_paths=["configs/preprocessing_figures.yaml", "configs/mcd.yaml"],
        ),
        source_df=source_df,
    )
    print("Wrote Figure 4.6.")

    # ---- Figure 4.7 ----
    labeled_train = labeled.loc[(labeled["mars_year"] == mars_year) & (labeled["sol"] >= sol_min) & (labeled["sol"] <= sol_max)]
    fig, source_dict = figure_4_7_spatiotemporal_structure(labeled_train, episode_catalogue, grid, config=config)
    save_figure_with_provenance(
        fig, out_dir / "figure_4_7_spatiotemporal_structure",
        FigureProvenance(
            figure_number="4.7",
            caption=(
                f"Temporal and spatial structure of the Martian dust field, MY{mars_year} training "
                f"partition: (a) qualifying storm_episode-centred composite of CDOD z-score "
                f"(sols -{config['episode_composite']['window_sols_before']}..+"
                f"{config['episode_composite']['window_sols_after']}); (b) pooled panel ACF/PACF "
                f"of the cell-sol CDOD z-score series; (c) CDOD z-score semivariogram by latitude "
                f"band and Ls quartile."
            ),
            units="dimensionless CDOD z-score",
            n=len(labeled_train),
            split="train",
            horizon_sols=0,
            data_source=DATA_SOURCE_LABEL,
            script="generate_chapter4_preprocessing_figures.py",
            config_paths=["configs/preprocessing_figures.yaml"],
        ),
        source_df=source_dict["semivariogram"],
    )
    source_dict["episode_composite"].to_csv(out_dir / "figure_4_7_episode_composite.csv", index=False)
    source_dict["acf_pacf"].to_csv(out_dir / "figure_4_7_acf_pacf.csv", index=False)
    print(f"Wrote Figure 4.7 (n_train_rows={len(labeled_train)}, "
          f"n_qualifying_episodes={int(episode_catalogue['qualifies'].sum())}).")

    # ---- Figure 4.8 ----
    fig, source_df = figure_4_8_seasonal_series(aggregated, cell_id=representative_cell_id, mars_year=mars_year)
    save_figure_with_provenance(
        fig, out_dir / "figure_4_8_seasonal_series",
        FigureProvenance(
            figure_number="4.8",
            caption=(
                f"Sol-aggregated column-dust-optical-depth time series for representative cell "
                f"{representative_cell_id}, MY{mars_year}, aligned to solar-longitude markers."
            ),
            units="dimensionless CDOD",
            n=len(source_df),
            split="train",
            horizon_sols=0,
            data_source=DATA_SOURCE_LABEL,
            script="generate_chapter4_preprocessing_figures.py",
            config_paths=["configs/preprocessing_figures.yaml"],
        ),
        source_df=source_df,
    )
    print(f"Wrote Figure 4.8 (cell {representative_cell_id}, n={len(source_df)}).")


if __name__ == "__main__":
    main()
