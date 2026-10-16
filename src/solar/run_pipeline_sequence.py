"""The reproducible entry point for the GOES sequence LSTM (Chapter 3.6,
Table 3.3): GOES Algorithm 3.1 preprocessing -> sequence-tensor
construction -> LSTM training.

    cd src/solar
    python -m run_pipeline_sequence --base-run-name <existing classical run> --run-name <name>

Reuses an EXISTING classical run's already-QC'd GOES rows
(results/<base_run_name>/goes_qc_accepted.csv) and already-built,
already-embargoed labeled dataset (results/<base_run_name>/dataset_with_split.csv)
-- QC and chronological splitting/embargo are not duplicated here, only
their real output is reused, so both the classical and sequence tasks are
evaluated on identical, real train/val/test partitions. Does not read or
modify anything in results/<base_run_name>/ -- writes only to
results/<run_name>/, the same purely-additive principle every other new
module this session has followed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from modeling.sequence_train import load_sequence_task_config, run_sequence_task
from preprocessing.alignment import aggregate_goes_disk, load_alignment_config
from preprocessing.goes_preprocessing import load_goes_preprocessing_config, preprocess_goes_sequence
from preprocessing.goes_sequences import load_goes_sequences_config
from preprocessing.sharp_sequences import SHARP_SEQ_MISSING_COL, build_combined_sequence_table
from splitting.chronological_split import load_splits_config

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


def run_pipeline_sequence(base_run_name: str, run_name: str, *, include_sharp: bool = False) -> dict:
    base_dir = SOLAR_ROOT / "results" / base_run_name
    run_dir = SOLAR_ROOT / "results" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"run_name": run_name, "base_run_name": base_run_name, "include_sharp": include_sharp}

    with _StepTimer("1. Load base run's QC'd GOES rows + labeled dataset"):
        goes_qc_accepted = pd.read_csv(base_dir / "goes_qc_accepted.csv")
        dataset_with_split = pd.read_csv(base_dir / "dataset_with_split.csv")
        summary["n_goes_qc_accepted"] = len(goes_qc_accepted)
        summary["n_dataset_rows"] = len(dataset_with_split)
        if include_sharp:
            sharp_qc_accepted = pd.read_csv(base_dir / "sharp_qc_accepted.csv")
            summary["n_sharp_qc_accepted"] = len(sharp_qc_accepted)

    with _StepTimer("2. Disk-level GOES aggregation (reused, unmodified)"):
        alignment_config = load_alignment_config()
        goes_disk = aggregate_goes_disk(goes_qc_accepted, alignment_config)
        summary["n_goes_disk_rows"] = len(goes_disk)

    with _StepTimer("3. GOES Algorithm 3.1 preprocessing"):
        preprocessing_config = load_goes_preprocessing_config()
        processed_goes = preprocess_goes_sequence(goes_disk, config=preprocessing_config)
        summary["n_processed_goes_rows"] = len(processed_goes)

    extra_gap_flag_columns: tuple[str, ...] = ()
    if include_sharp:
        with _StepTimer("3b. Real SHARP disk aggregation + exact-match merge onto GOES grid"):
            splits_config = load_splits_config()
            processed_goes, sharp_scaling_params = build_combined_sequence_table(
                processed_goes, sharp_qc_accepted,
                train_start=splits_config["train_start"], train_end=splits_config["train_end"],
                alignment_config=alignment_config,
            )
            summary["sharp_scaling_params"] = sharp_scaling_params
            summary["n_sharp_seq_missing_any"] = int(processed_goes[SHARP_SEQ_MISSING_COL].sum())
            extra_gap_flag_columns = (SHARP_SEQ_MISSING_COL,)

    processed_goes.to_csv(run_dir / "processed_goes.csv", index=False)

    with _StepTimer("4. Sequence-tensor construction + LSTM training"):
        sequences_config_name = "goes_sequences_with_sharp.yaml" if include_sharp else "goes_sequences.yaml"
        sequences_config = load_goes_sequences_config(SOLAR_ROOT / "configs" / sequences_config_name)
        task_config = load_sequence_task_config("flare_forecast_sequence")
        task_config["paths"] = {
            "checkpoint_dir": f"checkpoints/{run_name}",
            "results": f"results/{run_name}/metrics.json",
            "feasibility_report": f"results/{run_name}/feasibility_report.json",
        }
        task_results = run_sequence_task(
            processed_goes, dataset_with_split, task_config,
            project_root=SOLAR_ROOT, sequences_config=sequences_config, preprocessing_config=preprocessing_config,
            extra_gap_flag_columns=extra_gap_flag_columns,
        )
        summary["task_results"] = task_results

    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nPipeline complete. Summary: {run_dir / 'run_summary.json'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the GOES sequence LSTM pipeline (Chapter 3.6, Table 3.3).")
    parser.add_argument("--base-run-name", required=True, help="Existing classical run's results/ subdirectory to reuse QC'd GOES rows and the labeled dataset from.")
    parser.add_argument("--run-name", required=True, help="Subdirectory name under results/ and checkpoints/ for this run's outputs.")
    parser.add_argument(
        "--include-sharp", action="store_true",
        help="Extend the GOES-only sequence input with the 18 real SHARP magnetic descriptors "
             "(preprocessing/sharp_sequences.py), matching Chapter 5's described GOES+SHARP solar "
             "input. Default off: pure GOES-only input_size=2, unchanged from every prior run.",
    )
    args = parser.parse_args(argv)

    try:
        run_pipeline_sequence(args.base_run_name, args.run_name, include_sharp=args.include_sharp)
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"RUN FAILED: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
