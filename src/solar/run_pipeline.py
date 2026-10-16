"""The reproducible entry point for the real, full end-to-end solar pipeline:
QC -> alignment -> gap-filling -> features -> labeling -> dataset join ->
split+embargo -> baseline/classical-model training.

    cd src/solar
    python -m run_pipeline --goes-csv <path> --sharp-csv <path> --run-name <name>

(Run with cwd = src/solar -- same reason as ingest.py, see its module
docstring: unprefixed intra-package imports like `from preprocessing...`
only resolve this way.)

Every step reuses an already-tested library function from preprocessing/,
labeling/, modeling/, splitting/ -- this module's job is orchestration and
writing each stage's output to results/<run-name>/ immediately after that
stage completes, so a run that dies partway can be inspected/resumed from
the last completed stage rather than silently losing everything (the exact
failure mode `planning/REAL_FULL_RANGE_RUN.md` documents biting the GOES
ingestion run, before it had any per-file resume capability).

Uses the real, tracked configs (configs/{alignment,missing_data,features,
labeling,splits,task_flare_forecast}.yaml) unmodified -- unlike
planning/REAL_PILOT_RUN_2017-09.md's pilot run, which deliberately used
in-memory config overrides sized for a 7-day window. This entry point is
meant for real full-range (or any real-data) runs against those real
configs as they stand.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from labeling.flare_labels import build_primary_label, load_labeling_config
from modeling.datasets import assign_split_with_embargo, build_flare_forecast_dataset, summarize_targets
from modeling.train import load_task_config, run_task
from preprocessing.alignment import build_aligned_timeline, load_alignment_config
from preprocessing.features import build_features, load_features_config
from preprocessing.missing_data import apply_trailing_fill, load_missing_data_config
from preprocessing.quality_control import load_qc_config, run_quality_control

SOLAR_ROOT = Path(__file__).resolve().parent


class _StepTimer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        print(f"--- {self.name} ---", flush=True)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        elapsed = time.perf_counter() - self._start
        if exc_info[0] is None:
            print(f"    done in {elapsed:.1f}s", flush=True)
        else:
            print(f"    FAILED after {elapsed:.1f}s: {exc_info[1]}", flush=True)


def run_pipeline(goes_csv: Path, sharp_csv: Path, run_name: str) -> dict:
    run_dir = SOLAR_ROOT / "results" / run_name
    checkpoint_dir = SOLAR_ROOT / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"run_name": run_name, "goes_csv": str(goes_csv), "sharp_csv": str(sharp_csv)}

    with _StepTimer("1. Load real ingested GOES + SHARP CSVs"):
        goes_raw = pd.read_csv(goes_csv)
        sharp_raw = pd.read_csv(sharp_csv)
        summary["n_goes_raw"] = len(goes_raw)
        summary["n_sharp_raw"] = len(sharp_raw)

    with _StepTimer("2. Quality control"):
        qc_config = load_qc_config()
        goes_qc = run_quality_control(goes_raw, "goes", qc_config)
        sharp_qc = run_quality_control(sharp_raw, "sharp", qc_config)
        summary["goes_qc"] = goes_qc.summary()
        summary["sharp_qc"] = sharp_qc.summary()
        goes_qc.accepted.to_csv(run_dir / "goes_qc_accepted.csv", index=False)
        sharp_qc.accepted.to_csv(run_dir / "sharp_qc_accepted.csv", index=False)

    with _StepTimer("3. Disk-level alignment (SHARP cadence)"):
        alignment_config = load_alignment_config()
        aligned, goes_disk = build_aligned_timeline(goes_qc.accepted, sharp_qc.accepted, alignment_config)
        summary["n_aligned_rows"] = len(aligned)
        summary["n_goes_disk_rows"] = len(goes_disk)
        aligned.to_csv(run_dir / "aligned.csv", index=False)
        goes_disk.to_csv(run_dir / "goes_disk.csv", index=False)

    with _StepTimer("4. Causal gap-filling"):
        missing_config = load_missing_data_config()
        filled = apply_trailing_fill(aligned, config=missing_config)
        summary["n_filled_rows"] = len(filled)
        summary["n_locf_filled"] = int((~filled["is_observed"]).sum())
        filled.to_csv(run_dir / "filled.csv", index=False)

    with _StepTimer("5. Causal feature engineering"):
        features_config = load_features_config()
        features_df = build_features(filled, config=features_config)
        n_feature_cols = len([c for c in features_df.columns if c.startswith("feat_")])
        summary["n_feature_columns"] = n_feature_cols
        features_df.to_csv(run_dir / "features.csv", index=False)

    with _StepTimer("6. Forward-looking labeling"):
        labeling_config = load_labeling_config()
        labeled_df = build_primary_label(aligned, goes_disk, config=labeling_config)
        summary["forecast_horizon_hours"] = labeling_config["forecast_horizon_hours"]
        summary["label_primary_defined"] = int(labeled_df["label_primary"].notna().sum())
        summary["label_primary_positive"] = int(labeled_df["label_primary"].sum())
        labeled_df.to_csv(run_dir / "labeled.csv", index=False)

    with _StepTimer("7. Dataset join (features + labels)"):
        dataset, feature_manifest, target_cols = build_flare_forecast_dataset(features_df, labeled_df)
        summary["n_dataset_rows"] = len(dataset)
        summary["n_features_in_manifest"] = len(feature_manifest)
        (run_dir / "feature_manifest.json").write_text(
            json.dumps({"feature_manifest": feature_manifest, "target_columns": target_cols}, indent=2),
            encoding="utf-8",
        )

    with _StepTimer("8. Chronological split + embargo"):
        dataset_with_split = assign_split_with_embargo(dataset)
        summary["split_sizes"] = dataset_with_split["split"].value_counts().to_dict()
        target_summary = summarize_targets(dataset_with_split, target_cols)
        summary["target_summary"] = target_summary.to_dict(orient="records")
        dataset_with_split.to_csv(run_dir / "dataset_with_split.csv", index=False)

    with _StepTimer("9. Baseline + classical model training"):
        task_config = load_task_config("flare_forecast")
        task_config["paths"] = {
            "dataset": f"results/{run_name}/train_dataset.csv",
            "feature_manifest": f"results/{run_name}/train_feature_manifest.json",
            "results": f"results/{run_name}/metrics.json",
            "checkpoint_dir": f"checkpoints/{run_name}",
        }
        task_results = run_task(dataset_with_split, feature_manifest, task_config, project_root=SOLAR_ROOT)
        summary["task_results"] = task_results

    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nPipeline complete. Summary: {run_dir / 'run_summary.json'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the real, full end-to-end solar pipeline.")
    parser.add_argument("--goes-csv", required=True, help="Path to a normalized GOES XRS CSV (data_sources/schema.py's GOES_NORMALIZED_COLUMNS).")
    parser.add_argument("--sharp-csv", required=True, help="Path to a normalized SHARP CSV (data_sources/schema.py's SHARP_NORMALIZED_COLUMNS).")
    parser.add_argument("--run-name", required=True, help="Subdirectory name under results/ and checkpoints/ for this run's outputs.")
    args = parser.parse_args(argv)

    try:
        run_pipeline(Path(args.goes_csv), Path(args.sharp_csv), args.run_name)
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"RUN FAILED: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
