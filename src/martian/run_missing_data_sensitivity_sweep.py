"""Missing-data sensitivity sweep (CHAPTER_4_PARAMETER_VALUES.md Section 11
item 7): compares n_min (accepted retrievals per cell-sol), causal
trailing-fill gap, and feature-window completeness fraction, using a fast
gradient-boosting proxy model evaluated on TRAIN/VAL ONLY (never test) --
per the researcher's 2026-08-13 confirmation. Selection metric: val TSS
(storm detection, max) and val MAE (dust regression, min), averaged across
horizons 1/3/5, combined by rank sum across both tasks.

Grid, exactly the task's own specification: n_min in {1, 2} x
max_trailing_fill_gap_sols in {0, 1, 2} x min_window_completeness_frac in
{None, 0.6, 0.8} = 18 combinations.

Ingestion + QC + gridding is run ONCE (the expensive, ~15 min step) and
reused across all 18 combinations; only aggregation, fill, labeling, and
feature construction (all cheap) vary per combination.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from data_sources.real_emirs_provider import RealEMIRSObservationProvider
from data_sources.real_mcd_provider import RealMCDClimatologyProvider
from labeling.anomaly_labels import (
    compute_exceedance,
    compute_z_scores,
    load_anomaly_definition_config,
    run_full_labeling_pipeline,
)
from modeling.classical_models import make_gradient_boosting_classifier, make_gradient_boosting_regressor
from modeling.datasets import build_dust_regression_dataset, build_storm_detection_dataset
from modeling.metrics import classification_metrics, regression_metrics
from preprocessing.aggregation import aggregate_by_cell_sol, load_aggregation_config
from preprocessing.causal_filters import apply_causal_savgol, load_causal_filter_config
from preprocessing.features import build_features, load_features_config
from preprocessing.grid import assign_cells, load_primary_grid
from preprocessing.harmonisation import harmonise_dust_to_reference
from preprocessing.missing_data import apply_trailing_fill, load_missing_data_config
from preprocessing.quality_control import load_qc_config, run_quality_control
from splitting.chronological_split import assign_split, load_splits_config

SEED = 42
HORIZONS = (1, 3, 5)
N_MIN_GRID = [1, 2]
FILL_GAP_GRID = [0, 1, 2]
WINDOW_COMPLETENESS_GRID = [None, 0.6, 0.8]


def main() -> None:
    t0 = time.perf_counter()
    print("Step 1/2: real ingestion + QC + gridding (once, reused across all 18 combinations)...")
    provider = RealEMIRSObservationProvider()
    raw = provider.load(mars_year_range=(36, 37), seed=SEED)
    qc_report = run_quality_control(raw, load_qc_config())
    grid = load_primary_grid()
    gridded = assign_cells(qc_report.accepted, grid)
    gridded, _ = harmonise_dust_to_reference(gridded)  # same as mars/run_all.py step 4
    print(f"  n_qc_accepted={len(qc_report.accepted)} ({time.perf_counter() - t0:.0f}s elapsed)")

    mcd = RealMCDClimatologyProvider()
    anomaly_config = load_anomaly_definition_config()
    agg_config_base = load_aggregation_config()
    missing_config_base = load_missing_data_config()
    causal_filter_config = load_causal_filter_config()
    features_config_base = load_features_config()
    splits_config = load_splits_config()

    rows: list[dict] = []
    combo_id = 0
    for n_min in N_MIN_GRID:
        agg_config = dict(agg_config_base)
        aggregated = aggregate_by_cell_sol(
            gridded, min_obs_per_cell_sol=n_min,
            min_pressure_level_coverage_frac=agg_config["primary"]["min_pressure_level_coverage_frac"],
            config=agg_config,
        )
        labeled_df, episode_catalogue = run_full_labeling_pipeline(aggregated, mcd, grid, config=anomaly_config)

        for max_gap in FILL_GAP_GRID:
            filled = apply_trailing_fill(aggregated, max_gap_sols=max_gap, config=missing_config_base)
            filled = apply_causal_savgol(filled, config=causal_filter_config)
            filled_z = compute_z_scores(filled, mcd, epsilon=anomaly_config["epsilon"])
            filled_exceedance = compute_exceedance(filled_z, tau=anomaly_config["tau_primary"], config=anomaly_config)

            for window_completeness in WINDOW_COMPLETENESS_GRID:
                combo_id += 1
                t_combo = time.perf_counter()
                features_config = dict(features_config_base)
                features_config["min_window_completeness_frac"] = window_completeness
                features_df = build_features(filled_exceedance, grid, config=features_config)

                dust_dataset, dust_features = build_dust_regression_dataset(aggregated, labeled_df, features_df, horizons_sols=HORIZONS)
                storm_dataset, storm_features = build_storm_detection_dataset(labeled_df, features_df, horizons_sols=HORIZONS)

                dust_dataset["split"] = assign_split(dust_dataset, splits_config)
                storm_dataset["split"] = assign_split(storm_dataset, splits_config)

                mae_by_horizon = {}
                tss_by_horizon = {}
                for h in HORIZONS:
                    # dust regression
                    target_col = f"target_dust_optical_depth_h{h}"
                    d = dust_dataset.dropna(subset=[target_col])
                    train = d.loc[d["split"] == "train"]
                    val = d.loc[d["split"] == "val"]
                    if len(train) >= 10 and len(val) >= 5:
                        train_median = train[dust_features].median().fillna(0.0)  # 0.0 fallback if a column is entirely NaN in train
                        X_train = train[dust_features].fillna(train_median)
                        X_val = val[dust_features].fillna(train_median)
                        model = make_gradient_boosting_regressor(
                            {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05}, SEED
                        )
                        model.fit(X_train, train[target_col])
                        pred = model.predict(X_val)
                        mae_by_horizon[h] = regression_metrics(val[target_col].to_numpy(), pred)["mae"]
                    else:
                        mae_by_horizon[h] = float("nan")

                    # storm detection
                    target_col_s = f"target_storm_horizon_h{h}"
                    s = storm_dataset
                    train_s = s.loc[s["split"] == "train"]
                    val_s = s.loc[s["split"] == "val"]
                    if len(train_s) >= 10 and len(val_s) >= 5 and train_s[target_col_s].nunique() > 1:
                        train_median_s = train_s[storm_features].median().fillna(0.0)
                        X_train_s = train_s[storm_features].fillna(train_median_s)
                        X_val_s = val_s[storm_features].fillna(train_median_s)
                        model_s = make_gradient_boosting_classifier(
                            {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05}, SEED
                        )
                        model_s.fit(X_train_s, train_s[target_col_s])
                        prob = model_s.predict_proba(X_val_s)[:, 1]
                        tss_by_horizon[h] = classification_metrics(val_s[target_col_s].to_numpy(), prob)["tss"]
                    else:
                        tss_by_horizon[h] = float("nan")

                row = {
                    "combo_id": combo_id,
                    "n_min": n_min,
                    "max_trailing_fill_gap_sols": max_gap,
                    "min_window_completeness_frac": window_completeness,
                    "n_aggregated_cell_sols": len(aggregated),
                    "n_dust_train": int((dust_dataset["split"] == "train").sum()),
                    "n_dust_val": int((dust_dataset["split"] == "val").sum()),
                    "n_storm_train": int((storm_dataset["split"] == "train").sum()),
                    "n_storm_val": int((storm_dataset["split"] == "val").sum()),
                    "val_mae_h1": mae_by_horizon[1], "val_mae_h3": mae_by_horizon[3], "val_mae_h5": mae_by_horizon[5],
                    "val_tss_h1": tss_by_horizon[1], "val_tss_h3": tss_by_horizon[3], "val_tss_h5": tss_by_horizon[5],
                    "val_mae_mean": float(np.nanmean([mae_by_horizon[h] for h in HORIZONS])),
                    "val_tss_mean": float(np.nanmean([tss_by_horizon[h] for h in HORIZONS])),
                    "elapsed_s": round(time.perf_counter() - t_combo, 1),
                }
                rows.append(row)
                print(f"  combo {combo_id}/18: n_min={n_min} gap={max_gap} completeness={window_completeness} "
                      f"-> val_mae_mean={row['val_mae_mean']:.4f} val_tss_mean={row['val_tss_mean']:.4f} "
                      f"({row['elapsed_s']}s)")

    results = pd.DataFrame(rows)
    results["rank_mae"] = results["val_mae_mean"].rank(ascending=True)  # lower MAE better
    results["rank_tss"] = results["val_tss_mean"].rank(ascending=False)  # higher TSS better
    results["combined_rank_sum"] = results["rank_mae"] + results["rank_tss"]
    results = results.sort_values("combined_rank_sum")

    out_path = "outputs/real/missing_data_sensitivity_sweep_20260813.csv"
    results.to_csv(out_path, index=False)
    print()
    print(f"Wrote {out_path}")
    print()
    print("=== Full results, sorted by combined rank (best first) ===")
    print(results[[
        "combo_id", "n_min", "max_trailing_fill_gap_sols", "min_window_completeness_frac",
        "val_mae_mean", "val_tss_mean", "rank_mae", "rank_tss", "combined_rank_sum",
    ]].to_string(index=False))

    best = results.iloc[0]
    print()
    print("=== SELECTED PRIMARY CONFIGURATION (lowest combined rank sum, train/val evidence only) ===")
    print(f"n_min = {int(best['n_min'])}")
    print(f"max_trailing_fill_gap_sols = {int(best['max_trailing_fill_gap_sols'])}")
    print(f"min_window_completeness_frac = {best['min_window_completeness_frac']}")
    print(f"val_mae_mean = {best['val_mae_mean']:.4f} (rank {best['rank_mae']:.0f}/18)")
    print(f"val_tss_mean = {best['val_tss_mean']:.4f} (rank {best['rank_tss']:.0f}/18)")
    print()
    print(f"Total wall time: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
