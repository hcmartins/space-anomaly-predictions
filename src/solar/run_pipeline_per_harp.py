"""Per-active-region (HARP-level) real end-to-end pipeline -- Phase 2 of
scoping out Chapter 3.2.1's per-HARP forecasting task. Mirrors
run_pipeline.py's orchestration pattern (same stage-by-stage
checkpointing to results/<run-name>/), but for the per-HARP path:
QC (reused, unchanged) -> harp_alignment.build_per_harp_timeline ->
gap-fill/features via groupby("HARPNUM").apply() over the UNCHANGED
apply_trailing_fill/build_features -> harp_attribution (Phase 1, reused
unchanged) -> harp_catalogue_labels (Phase 1's real attribution as the
label source) -> harp_datasets join -> assign_split_with_embargo (reused
unchanged) -> run_task (reused unchanged).

    cd src/solar
    python -m run_pipeline_per_harp --goes-csv <path> --sharp-csv <path> \\
        --sharp-noaa-ars-csv <path> --flare-catalogue-csv <path> --run-name <name>

SHARP is split across two real CSVs because configs/sharp.yaml's
NOAA_ARS keyword was added AFTER the original full-scale
physical-keyword ingestion (see planning/FLARE_CATALOGUE_DOCUMENTATION.md)
-- rather than re-running that ~4-hour ingestion, this merges the
lighter NOAA_ARS-only patch (data_sources/sharp_ingestion.py's
ingest_sharp_chunked with `keywords: []`) onto the existing
physical-keyword CSV by (HARPNUM, T_REC).

Does NOT touch run_pipeline.py's disk-integrated path or its real,
already-reported results -- writes to a separate results/<run-name>/
directory.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd

from labeling.flare_labels import load_labeling_config
from labeling.harp_catalogue_labels import build_harp_catalogue_labels
from modeling.datasets import assign_split_with_embargo, summarize_targets
from modeling.harp_datasets import build_harp_flare_forecast_dataset
from modeling.train import load_task_config, run_task
from preprocessing.features import build_features, load_features_config
from preprocessing.harp_alignment import build_per_harp_timeline
from preprocessing.harp_attribution import attribute_flares_to_harps, attribution_quality_report, build_harp_ar_index
from preprocessing.missing_data import apply_trailing_fill, load_missing_data_config
from preprocessing.quality_control import load_qc_config, run_quality_control

SOLAR_ROOT = Path(__file__).resolve().parent


def _apply_per_harp_group(df: pd.DataFrame, func, **kwargs) -> pd.DataFrame:
    """Applies `func` (apply_trailing_fill or build_features, UNCHANGED)
    once per HARPNUM group, re-attaching the HARPNUM column afterward.

    Needed because pandas >=2.2 (confirmed on 3.0.3, 2026-07-26) no
    longer passes the grouping column into a groupby().apply() callback
    at all (include_groups=True was removed entirely, not just
    deprecated -- ValueError: "include_groups=True is no longer
    allowed."), so `func`'s output would otherwise silently lose
    HARPNUM. `g.name` still reliably gives the group's HARPNUM value even
    though the column itself is stripped from `g`."""

    def wrapped(g: pd.DataFrame) -> pd.DataFrame:
        harpnum = g.name
        result = func(g, **kwargs)
        result = result.copy()
        result["HARPNUM"] = harpnum
        return result

    return df.groupby("HARPNUM", group_keys=False).apply(wrapped)


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


def run_pipeline_per_harp(
    goes_csv: Path, sharp_csv: Path, sharp_noaa_ars_csv: Path, flare_catalogue_csv: Path, run_name: str
) -> dict:
    run_dir = SOLAR_ROOT / "results" / run_name
    checkpoint_dir = SOLAR_ROOT / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"run_name": run_name}

    with _StepTimer("1. Load real GOES + SHARP (physical + NOAA_ARS patch, merged) + flare catalogue"):
        goes_raw = pd.read_csv(goes_csv)
        sharp_physical = pd.read_csv(sharp_csv)
        sharp_ars = pd.read_csv(sharp_noaa_ars_csv, usecols=["HARPNUM", "T_REC", "NOAA_ARS"])
        sharp_raw = sharp_physical.merge(sharp_ars, on=["HARPNUM", "T_REC"], how="left")
        flares = pd.read_csv(flare_catalogue_csv, parse_dates=["time_utc_peak", "time_utc_start", "time_utc_end"])
        # A CSV round-trip of a column mixing numeric-looking strings
        # ("2673") with blanks infers float64 (blanks -> NaN), so a naive
        # .astype(str) would give "2673.0", never matching harp_ar_index's
        # "2673" -- strip the float artifact back to the real string form
        # flare_catalogue_ingestion.py originally produced.
        flares["active_region"] = flares["active_region"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
        summary["n_goes_raw"] = len(goes_raw)
        summary["n_sharp_raw"] = len(sharp_raw)
        summary["n_flares_raw"] = len(flares)

    with _StepTimer("2. Quality control (reused, unchanged)"):
        qc_config = load_qc_config()
        goes_qc = run_quality_control(goes_raw, "goes", qc_config)
        sharp_qc = run_quality_control(sharp_raw, "sharp", qc_config)
        summary["goes_qc"] = goes_qc.summary()
        summary["sharp_qc"] = sharp_qc.summary()

    with _StepTimer("3. Per-HARP alignment (limb exclusion + GOES trailing state)"):
        aligned, goes_disk = build_per_harp_timeline(goes_qc.accepted, sharp_qc.accepted)
        summary["n_aligned_rows"] = len(aligned)
        summary["n_distinct_harps"] = int(aligned["HARPNUM"].nunique())
        aligned.to_csv(run_dir / "aligned_per_harp.csv", index=False)

    with _StepTimer("4. NOAA AR -> HARPNUM attribution (Phase 1, reused unchanged)"):
        # Computed here (before gap-filling/features, not after) specifically
        # so `aligned` only needs to stay alive until step 6 (labeling) --
        # harp_ar_index itself is small (HARPNUM/time_utc/active_region
        # only), not the ~30-column full aligned timeline. Real full-scale
        # run 2026-07-26 hit ArrayMemoryError at the step 8 join with
        # aligned/filled/attributed/features_df/labeled_df all still
        # alive simultaneously (~3.9M rows each) -- this reordering plus
        # the explicit `del`s below fix that at its real, measured cause.
        harp_ar_index = build_harp_ar_index(aligned, time_col="time_utc")
        attributed = attribute_flares_to_harps(flares, harp_ar_index)
        summary["attribution_quality"] = attribution_quality_report(attributed)
        attributed.to_csv(run_dir / "flares_attributed.csv", index=False)
        del harp_ar_index, flares
        gc.collect()

    with _StepTimer("5. Causal gap-filling per HARP group (reuses apply_trailing_fill unchanged)"):
        missing_config = load_missing_data_config()
        filled = _apply_per_harp_group(aligned, apply_trailing_fill, config=missing_config)
        summary["n_filled_rows"] = len(filled)
        filled.to_csv(run_dir / "filled_per_harp.csv", index=False)

    with _StepTimer("6. Causal feature engineering per HARP group (reuses build_features unchanged)"):
        features_config = load_features_config()
        features_df = _apply_per_harp_group(filled, build_features, config=features_config)
        n_feature_cols = len([c for c in features_df.columns if c.startswith("feat_")])
        summary["n_feature_columns"] = n_feature_cols
        features_df.to_csv(run_dir / "features_per_harp.csv", index=False)
        # Freed immediately (not carried through step 7) -- labeling below
        # doesn't need it, and at real full-scale (~3.9M rows, ~260 cols)
        # every large object held "just in case" measurably raises peak
        # memory; reloaded fresh from disk right before step 8's join,
        # which is the only step that actually needs it again.
        del filled, features_df
        gc.collect()

    with _StepTimer("7. Forward-looking per-HARP catalogue labeling"):
        labeling_config = load_labeling_config()
        labeled_df = build_harp_catalogue_labels(aligned, attributed, config=labeling_config)
        summary["forecast_horizon_hours"] = labeling_config["forecast_horizon_hours"]
        summary["label_primary_defined"] = int(labeled_df["label_primary"].notna().sum())
        summary["label_primary_positive"] = int(labeled_df["label_primary"].sum())
        labeled_df.to_csv(run_dir / "labeled_per_harp.csv", index=False)
        del aligned, attributed, labeled_df
        gc.collect()

    with _StepTimer("8. Dataset join (per-HARP features + labels)"):
        # Real ArrayMemoryError hit here twice at full scale before this
        # fix (see planning/FLARE_CATALOGUE_DOCUMENTATION.md) -- reloading
        # fresh from disk (rather than carrying features_df/labeled_df as
        # live objects since steps 6/7) guarantees this is the ONLY large
        # data alive at the pipeline's single most memory-intensive step.
        features_df = pd.read_csv(run_dir / "features_per_harp.csv")
        labeled_df = pd.read_csv(run_dir / "labeled_per_harp.csv")
        dataset, feature_manifest, target_cols = build_harp_flare_forecast_dataset(features_df, labeled_df)
        summary["n_dataset_rows"] = len(dataset)
        summary["n_features_in_manifest"] = len(feature_manifest)
        (run_dir / "feature_manifest.json").write_text(
            json.dumps({"feature_manifest": feature_manifest, "target_columns": target_cols}, indent=2),
            encoding="utf-8",
        )
        del features_df, labeled_df
        gc.collect()

    with _StepTimer("9. Chronological split + embargo (reused unchanged)"):
        dataset_with_split = assign_split_with_embargo(dataset)
        summary["split_sizes"] = dataset_with_split["split"].value_counts().to_dict()
        target_summary = summarize_targets(dataset_with_split, target_cols)
        summary["target_summary"] = target_summary.to_dict(orient="records")
        dataset_with_split.to_csv(run_dir / "dataset_with_split.csv", index=False)
        del dataset
        gc.collect()

    with _StepTimer("10. Baseline + classical model training"):
        task_config = load_task_config("flare_forecast")
        task_config["paths"] = {
            "dataset": f"results/{run_name}/train_dataset.csv",
            "feature_manifest": f"results/{run_name}/train_feature_manifest.json",
            "results": f"results/{run_name}/metrics.json",
            "checkpoint_dir": f"checkpoints/{run_name}",
        }
        # Scale-driven substitution (see modeling/classical_models.py's
        # make_hist_gradient_boosting_classifier docstring): swap
        # gradient_boosting_classifier -> hist_gradient_boosting_classifier
        # for THIS run only. configs/task_flare_forecast.yaml (the
        # disk-integrated task config this dict was just loaded from) is
        # untouched on disk -- only this in-memory copy is modified.
        for spec in task_config["classical_models"]:
            if spec["name"] == "gradient_boosting_classifier":
                spec["name"] = "hist_gradient_boosting_classifier"
                spec["hyperparameters"] = {
                    "max_iter": spec["hyperparameters"].get("n_estimators", 200),
                    "max_depth": spec["hyperparameters"].get("max_depth", 3),
                    "learning_rate": spec["hyperparameters"].get("learning_rate", 0.05),
                }
        # Real ArrayMemoryError on this exact machine (15.6GB RAM, 6.6GB
        # free at idle) trying to median-fit SimpleImputer on the full
        # ~1.97M-row train split -- see modeling/train.py's run_task
        # docstring and planning/FLARE_CATALOGUE_DOCUMENTATION.md.
        task_results = run_task(
            dataset_with_split, feature_manifest, task_config, project_root=SOLAR_ROOT,
            imputer_fit_sample_size=150_000,
        )
        summary["task_results"] = task_results

    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nPipeline complete. Summary: {run_dir / 'run_summary.json'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the real, per-HARP end-to-end solar pipeline (Phase 2).")
    parser.add_argument("--goes-csv", required=True, help="Path to a normalized GOES XRS CSV.")
    parser.add_argument("--sharp-csv", required=True, help="Normalized SHARP CSV with the 18 physical keywords.")
    parser.add_argument("--sharp-noaa-ars-csv", required=True, help="NOAA_ARS-only SHARP patch CSV (HARPNUM, T_REC, NOAA_ARS).")
    parser.add_argument("--flare-catalogue-csv", required=True, help="Normalized flare catalogue CSV.")
    parser.add_argument("--run-name", required=True, help="Subdirectory name under results/ and checkpoints/.")
    args = parser.parse_args(argv)

    try:
        run_pipeline_per_harp(
            Path(args.goes_csv), Path(args.sharp_csv), Path(args.sharp_noaa_ars_csv),
            Path(args.flare_catalogue_csv), args.run_name,
        )
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"RUN FAILED: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
