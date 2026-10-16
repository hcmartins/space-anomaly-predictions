"""The one reproducible entry point for the whole Martian Chapter 4 pipeline.

    python -m mars.run_all --config configs/run.yaml

Every step below reuses an already-tested library function from
data_sources/, preprocessing/, labeling/, splitting/, modeling/, or
evaluation/ -- this module's job is orchestration, run isolation
(outputs/real/<run_id>/...), and reproducibility bookkeeping, not new
scientific logic. Real data only: RealEMIRSObservationProvider requires real
downloaded EMIRS FITS files (configs/real_emirs_ingestion.yaml) and
RealMCDClimatologyProvider requires a precomputed real climatology file
(configs/mcd.yaml) -- this pipeline never falls back to simulated data. See
planning/PIPELINE_DOCUMENTATION.md for the step-by-step mapping and
planning/README.md (repo root README.md) for exact reproduction commands.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

from data_sources.real_documented_events import load_real_documented_events, load_real_documented_events_config
from data_sources.real_emirs_provider import RealEMIRSObservationProvider
from data_sources.real_mcd_provider import RealMCDClimatologyProvider
from labeling.anomaly_labels import (
    compute_exceedance,
    compute_z_scores,
    load_anomaly_definition_config,
    run_full_labeling_pipeline,
)
from labeling.catalogue_validation import load_catalogue_validation_config, run_catalogue_validation, write_table_4_6
from labeling.catalogue_tolerance_tuning import (
    load_catalogue_tolerance_tuning_config,
    tune_and_validate_tolerance,
    write_table_4_6b,
)
from labeling.storm_taxonomy import classify_episode_catalogue, load_storm_taxonomy_config
from preprocessing.aggregation import aggregate_by_cell_sol, load_aggregation_config
from preprocessing.causal_filters import apply_causal_savgol, load_causal_filter_config
from preprocessing.features import build_features, load_features_config
from preprocessing.grid import assign_cells, load_primary_grid
from preprocessing.harmonisation import harmonise_dust_to_reference, load_harmonisation_config
from preprocessing.missing_data import apply_trailing_fill, load_missing_data_config
from preprocessing.quality_control import load_qc_config, run_quality_control
from preprocessing.reporting import qc_summary_report, write_qc_log
from splitting.chronological_split import (
    assign_split,
    find_episodes_crossing_split_boundary,
    load_splits_config,
    write_split_manifest,
)

from .checksums import write_checksums_manifest
from .final_report import generate_final_report
from .leakage_checks import LeakageCheckFailed, run_leakage_checks
from .run_manifest import RunManifest, mark_step_complete, write_run_manifest
from .run_paths import build_run_paths, create_run_directories, generate_run_id
from .validate_inputs import InputValidationError, validate_inputs

MARTIAN_ROOT = Path(__file__).resolve().parent.parent
DATA_SOURCE = "real"
DATA_SOURCE_LABEL = "REAL (Martian Chapter 4 mars.run_all pipeline)"


class _StepTimer:
    """Context manager that both times a step and marks it complete in the
    run manifest -- used so every one of the 14 numbered steps is
    individually timed and recorded, never silently merged into "the run
    took N seconds" with no per-step breakdown."""

    def __init__(self, manifest: RunManifest, step_name: str):
        self.manifest = manifest
        self.step_name = step_name

    def __enter__(self):
        print(f"--- Step: {self.step_name} ---")
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        elapsed = time.perf_counter() - self._start
        if exc_info[0] is None:
            mark_step_complete(self.manifest, self.step_name, elapsed)
            print(f"    done in {elapsed:.2f}s")
        else:
            print(f"    FAILED after {elapsed:.2f}s: {exc_info[1]}")


def load_run_config(config_path: Path) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def run_pipeline(config: dict) -> Path:
    from modeling.datasets import build_dust_regression_dataset, build_storm_detection_dataset
    from modeling.train import load_task_config, run_task

    # ---- Step 1: validate inputs ----
    validate_inputs(config)

    seed = config["seed"]
    mars_year_range = tuple(config["mars_year_range"])
    tasks = config.get("tasks", ["dust_regression", "storm_detection"])

    run_id = generate_run_id(seed)
    run_paths = build_run_paths(Path(config["output_root"]), DATA_SOURCE, run_id)
    create_run_directories(run_paths)

    seed_strategy = {
        "global_generation_seed": seed,
        "task_seeds": {"dust_regression": 42, "storm_detection": 44},
        "bootstrap_seed": "see configs/evaluation.yaml bootstrap.seed",
        "note": "see planning/SEED_STRATEGY.md for the full rationale",
    }
    manifest = RunManifest(run_id=run_id, data_source=DATA_SOURCE, config=config, seed_strategy=seed_strategy)

    # ---- Step 2: write run manifest ----
    manifest_path = run_paths["manifests"] / "run_manifest.json"
    write_run_manifest(manifest_path, manifest)

    try:
        pipeline_summary: dict = {}

        # ---- Step 3: QC and gridding ----
        with _StepTimer(manifest, "3_qc_and_gridding"):
            provider = RealEMIRSObservationProvider()
            raw = provider.load(mars_year_range=mars_year_range, seed=seed)
            qc_report = run_quality_control(raw, load_qc_config())
            grid = load_primary_grid()
            gridded = assign_cells(qc_report.accepted, grid)
            pipeline_summary["n_retrievals_generated"] = len(raw)
            pipeline_summary["n_qc_accepted"] = qc_report.summary()["n_accepted"]
            pipeline_summary["n_qc_rejected"] = qc_report.summary()["n_rejected"]
            write_qc_log(run_paths["manifests"], qc_report, data_source_label=DATA_SOURCE_LABEL, run_id=f"{run_id}_qc")
            (run_paths["tables"] / "qc_summary.md").write_text(
                qc_summary_report(qc_report, DATA_SOURCE_LABEL), encoding="utf-8"
            )

        # ---- Step 4: aggregate cell-sols ----
        with _StepTimer(manifest, "4_aggregate_cell_sols"):
            # Per-retrieval physical harmonisation to the Montabone reference
            # convention (configs/harmonisation.yaml) BEFORE aggregation.
            gridded, harmonisation_summary = harmonise_dust_to_reference(gridded, load_harmonisation_config())
            pipeline_summary["dust_harmonisation"] = harmonisation_summary
            agg_config = load_aggregation_config()
            aggregated = aggregate_by_cell_sol(
                gridded,
                min_obs_per_cell_sol=agg_config["primary"]["min_obs_per_cell_sol"],
                min_pressure_level_coverage_frac=agg_config["primary"]["min_pressure_level_coverage_frac"],
                config=agg_config,
            )
            pipeline_summary["n_aggregated_cell_sols"] = len(aggregated)

        # ---- Step 5: treat missing data causally ----
        with _StepTimer(manifest, "5_causal_missing_data"):
            missing_config = load_missing_data_config()
            filled = apply_trailing_fill(
                aggregated, max_gap_sols=missing_config["primary"]["max_trailing_fill_gap_sols"], config=missing_config
            )
            pipeline_summary["n_trailing_filled_rows"] = len(filled)

        # ---- Step 6: apply causal filtering ----
        with _StepTimer(manifest, "6_causal_filtering"):
            causal_filter_config = load_causal_filter_config()
            filled_with_sg = apply_causal_savgol(filled, config=causal_filter_config)
            filled_with_sg.to_csv(run_paths["data"] / "filled_with_causal_savgol.csv", index=False)

        # ---- Step 7: calculate climatological deviations ----
        with _StepTimer(manifest, "7_climatological_deviations"):
            mcd = RealMCDClimatologyProvider()
            anomaly_config = load_anomaly_definition_config()
            filled_z = compute_z_scores(filled, mcd, epsilon=anomaly_config["epsilon"])
            filled_exceedance = compute_exceedance(filled_z, tau=anomaly_config["tau_primary"], config=anomaly_config)
            aggregated.to_csv(run_paths["data"] / "aggregated_cell_sol.csv", index=False)
            filled.to_csv(run_paths["data"] / "trailing_filled_cell_sol.csv", index=False)

        # ---- Step 8: build labels and episodes ----
        with _StepTimer(manifest, "8_build_labels_and_episodes"):
            labeled_df, episode_catalogue = run_full_labeling_pipeline(aggregated, mcd, grid, config=anomaly_config)
            taxonomy_config = load_storm_taxonomy_config()
            episode_catalogue = classify_episode_catalogue(episode_catalogue, taxonomy_config)
            pipeline_summary["n_atmospheric_anomaly"] = int(labeled_df["atmospheric_anomaly"].sum())
            pipeline_summary["n_storm_episode"] = int(labeled_df["storm_episode"].sum())
            pipeline_summary["n_candidate_episodes"] = len(episode_catalogue)

            splits_config = load_splits_config()
            crossing = find_episodes_crossing_split_boundary(episode_catalogue, splits_config)
            if len(crossing):
                crossing_ids = set(crossing["episode_id"])
                exclude_mask = labeled_df["episode_id"].isin(crossing_ids)
                excluded_keys = labeled_df.loc[exclude_mask, ["cell_id", "mars_year", "sol"]]
                labeled_df = labeled_df.loc[~exclude_mask].reset_index(drop=True)
                aggregated = aggregated.merge(
                    excluded_keys.assign(_exclude=True), on=["cell_id", "mars_year", "sol"], how="left"
                )
                aggregated = aggregated.loc[aggregated["_exclude"].isna()].drop(columns=["_exclude"]).reset_index(drop=True)
                episode_catalogue = episode_catalogue.loc[
                    ~episode_catalogue["episode_id"].isin(crossing_ids)
                ].reset_index(drop=True)
                pipeline_summary["n_episodes_excluded_for_split_boundary_crossing"] = len(crossing_ids)
            else:
                pipeline_summary["n_episodes_excluded_for_split_boundary_crossing"] = 0

            labeled_df.to_csv(run_paths["labels"] / "labeled_cell_sol.csv", index=False)
            episode_catalogue.to_csv(run_paths["labels"] / "episode_catalogue.csv", index=False)

        # ---- Step 9: validate labels against the real documented dust-storm
        # catalogue (Table 4.6, Section 4.4.1) -- OPTIONAL: if the real
        # catalogue file isn't present, this is honestly recorded as
        # "NOT PERFORMED" rather than failing the whole run or silently
        # skipping without a trace. See labeling/catalogue_validation.py and
        # data_sources/real_documented_events.py.
        with _StepTimer(manifest, "9_validate_labels_against_documented_catalogue"):
            try:
                documented_events_config = load_real_documented_events_config()
                documented_events = load_real_documented_events(documented_events_config)
                catalogue_validation_config = load_catalogue_validation_config()
                catalogue_result = run_catalogue_validation(
                    labeled_df, episode_catalogue, documented_events, grid, config=catalogue_validation_config
                )
                write_table_4_6(catalogue_result, run_paths["tables"], data_source_label=DATA_SOURCE_LABEL)
                catalogue_result["matched_pairs"].to_csv(run_paths["labels"] / "catalogue_matched_pairs.csv", index=False)
                catalogue_result["per_documented_event"].to_csv(run_paths["labels"] / "catalogue_documented_events.csv", index=False)
                catalogue_result["per_episode"].to_csv(run_paths["labels"] / "catalogue_episode_status.csv", index=False)
                pipeline_summary["catalogue_validation"] = catalogue_result["summary"]

                # ---- Step 9b: blind dev/test tuning of the Table 4.6
                # matching tolerance (CHAPTER_4_PARAMETER_VALUES.md open
                # item 4) -- searches (temporal_tolerance_sols,
                # spatial_region_radius_cells) on a DEV half of the
                # documented events only, then reports Table 4.6b's
                # hit/miss/POD/precision on the held-out TEST half under the
                # frozen, dev-selected tolerance. Additive: the original
                # Table 4.6 above (full catalogue, untuned tolerance) is
                # unaffected. Gated by the same catalogue availability as
                # Step 9 -- if the FileNotFoundError above didn't fire,
                # documented_events is already loaded and real.
                tuning_config = load_catalogue_tolerance_tuning_config()
                tuning_result = tune_and_validate_tolerance(
                    labeled_df, episode_catalogue, documented_events, grid,
                    config=tuning_config,
                    unmatched_episode_label=catalogue_validation_config.get("unmatched_episode_label", "false_positive_candidate"),
                )
                write_table_4_6b(tuning_result, run_paths["tables"], data_source_label=DATA_SOURCE_LABEL)
                pipeline_summary["catalogue_tolerance_tuning"] = {
                    "selected_tolerance": tuning_result["selected_tolerance"],
                    "split": tuning_result["split"],
                    "test_validation_summary": tuning_result["test_validation"]["summary"],
                }
            except FileNotFoundError as exc:
                pipeline_summary["catalogue_validation"] = {"status": "NOT PERFORMED", "reason": str(exc)}
                pipeline_summary["catalogue_tolerance_tuning"] = {"status": "NOT PERFORMED", "reason": str(exc)}

        # ---- Step 10: build causal features ----
        with _StepTimer(manifest, "10_build_causal_features"):
            features_config = load_features_config()
            features_df = build_features(filled_exceedance, grid, config=features_config)
            n_feature_cols = len([c for c in features_df.columns if c.startswith("feat_")])
            pipeline_summary["n_engineered_feature_columns"] = n_feature_cols

        # ---- Step 11: construct chronological splits ----
        with _StepTimer(manifest, "11_chronological_splits"):
            split_tag = assign_split(labeled_df, splits_config)
            write_split_manifest(
                run_paths["splits"] / "split_manifest.json",
                labeled_df.assign(split=split_tag),
                label_cols=["atmospheric_anomaly", "storm_episode"],
            )
            pipeline_summary["split_sizes"] = split_tag.value_counts().to_dict()

        # Per-task datasets (needed before leakage checks and training).
        dataset_builders = {
            "dust_regression": lambda: build_dust_regression_dataset(aggregated, labeled_df, features_df),
            "storm_detection": lambda: build_storm_detection_dataset(labeled_df, features_df),
        }
        task_datasets = {name: dataset_builders[name]() for name in tasks}

        # ---- Step 12: run leakage tests ----
        with _StepTimer(manifest, "12_leakage_checks"):
            try:
                leakage_summary = run_leakage_checks(
                    episode_catalogue=episode_catalogue, splits_config=splits_config, task_datasets=task_datasets
                )
                manifest.leakage_checks = leakage_summary
            except LeakageCheckFailed as exc:
                manifest.leakage_checks = {"status": "FAILED", "detail": str(exc)}
                manifest.status = "failed"
                manifest.error = str(exc)
                write_run_manifest(manifest_path, manifest)
                raise

        # ---- Step 13: train baselines/models, evaluate every task/horizon,
        # compute uncertainty intervals, generate figures/tables. These four
        # are fused in a single per-model call
        # (evaluation.evaluate_task.evaluate_and_report, invoked from inside
        # modeling.train.run_task immediately after each model is fit) so
        # the already-fitted model/imputer never needs reloading from a
        # checkpoint -- see planning/EVALUATION_DOCUMENTATION.md.
        with _StepTimer(manifest, "13_train_evaluate_uncertainty_figures"):
            task_results = {}
            for task_name in tasks:
                dataset_df, feature_manifest = task_datasets[task_name]
                task_config = load_task_config(task_name)
                task_config["paths"] = _run_scoped_task_paths(run_paths, task_name)
                task_results[task_name] = run_task(
                    task_name, dataset_df, feature_manifest, task_config,
                    features_df=features_df,
                    episode_catalogue=episode_catalogue if task_name == "storm_detection" else None,
                    splits_config=splits_config,
                    tables_dir=run_paths["tables"] / task_name,  # task-scoped: shared model names across tasks would otherwise collide
                    project_root=Path("."),  # task_config["paths"] are already absolute
                    data_source_label=DATA_SOURCE_LABEL,
                )

        # ---- Step 14: generate final report ----
        with _StepTimer(manifest, "14_final_report"):
            manifest.status = "completed"
            manifest.finished_at_utc = _utc_now()
            write_run_manifest(manifest_path, manifest)
            generate_final_report(
                run_paths=run_paths, manifest_dict=manifest.to_dict(),
                pipeline_summary=pipeline_summary, task_results=task_results,
            )

        # ---- Step 15: write checksums for every output ----
        with _StepTimer(manifest, "15_checksums"):
            checksums_path = write_checksums_manifest(run_paths["run_dir"])

        write_run_manifest(manifest_path, manifest)
        print(f"\nRun complete: {run_paths['run_dir']}")
        print(f"Final report: {run_paths['reports'] / 'final_report.md'}")
        print(f"Checksums: {checksums_path}")
        return run_paths["run_dir"]

    except (InputValidationError, LeakageCheckFailed):
        raise
    except Exception as exc:
        manifest.status = "failed"
        manifest.error = str(exc)
        manifest.finished_at_utc = _utc_now()
        write_run_manifest(manifest_path, manifest)
        raise


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _run_scoped_task_paths(run_paths: dict, task_name: str) -> dict:
    # NOTE: "results".parent is used by modeling.train.run_task as the
    # per-model evaluation/provenance results_dir (evaluation/evaluate_task.py
    # writes files there named only by model_name+horizon, e.g.
    # "random_forest_regressor_h1_evaluation.json" -- with NO task_name in
    # that filename). Since tasks share baseline/classical model names
    # (e.g. "climatology", "persistence"), "results" MUST resolve to a
    # task-specific subdirectory (metrics/<task_name>/), never the flat
    # metrics/ directory directly, or two tasks would silently overwrite each
    # other's evaluation files (found via a real full-scale run).
    task_metrics_dir = run_paths["metrics"] / task_name
    return {
        "dataset": str(run_paths["features"] / f"{task_name}_dataset.csv"),
        "feature_manifest": str(run_paths["features"] / f"{task_name}_feature_manifest.json"),
        "results": str(task_metrics_dir / f"{task_name}_metrics.json"),
        "predictions": str(task_metrics_dir / f"{task_name}_predictions.csv"),
        "checkpoint_dir": str(run_paths["models"] / task_name),
        "figures_dir": str(run_paths["figures"] / task_name),
        "sequence_feasibility_report": str(task_metrics_dir / f"{task_name}_sequence_feasibility_report.csv"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full Martian Chapter 4 pipeline end-to-end.")
    parser.add_argument("--config", required=True, help="Path to a mars run config YAML (e.g. configs/run.yaml).")
    args = parser.parse_args(argv)

    config = load_run_config(args.config)
    try:
        run_pipeline(config)
    except InputValidationError as exc:
        print(f"INPUT VALIDATION FAILED:\n{exc}", file=sys.stderr)
        return 2
    except LeakageCheckFailed as exc:
        print(f"LEAKAGE CHECK FAILED -- run aborted, no downstream artifacts trusted:\n{exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"RUN FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
