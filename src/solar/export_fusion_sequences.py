"""Exports real GOES sequence tensors (X, y per split) for Chapter 5's
cross-domain fusion training (src/cross-domain/run_fusion.py).

Reuses an existing sequence run's real, already-processed data -- never
recomputes QC/preprocessing/labeling, only regenerates the sequence
tensors themselves, which modeling/sequence_train.py::run_sequence_task
builds in-memory and never persists to disk.

    cd src/solar
    python -m export_fusion_sequences --base-run-name <classical run> --sequence-run-name <sequence run>

Writes results/<sequence-run-name>/fusion_export/{X,y}_{train,val,test}.npy
and regime_test.npy (real calendar year per test-split row -- a simple,
non-fabricated grouping for cross-domain's post-hoc negative-transfer
reporting only; never fed into any training step).

`--include-sharp`: for a sequence run produced by
`run_pipeline_sequence.py --include-sharp`, the persisted
`processed_goes.csv` already IS the real GOES+SHARP combined table (see
that script's step 3b / preprocessing/sharp_sequences.py) -- this flag
just tells this exporter to read `configs/goes_sequences_with_sharp.yaml`
(20 feature columns) and propagate `sharp_seq_missing_any` as an extra
gap-flag column, matching how that sequence run was trained. It does NOT
redo the SHARP merge/scaling itself.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.sequence_train import load_sequence_task_config
from preprocessing.goes_preprocessing import load_goes_preprocessing_config
from preprocessing.goes_sequences import build_goes_sequences, load_goes_sequences_config
from preprocessing.sharp_sequences import SHARP_SEQ_MISSING_COL

SOLAR_ROOT = Path(__file__).resolve().parent


def export_fusion_sequences(base_run_name: str, sequence_run_name: str, *, include_sharp: bool = False) -> Path:
    base_dir = SOLAR_ROOT / "results" / base_run_name
    sequence_dir = SOLAR_ROOT / "results" / sequence_run_name
    out_dir = sequence_dir / "fusion_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_with_split = pd.read_csv(base_dir / "dataset_with_split.csv")
    processed_goes = pd.read_csv(sequence_dir / "processed_goes.csv")
    processed_goes["time_utc"] = pd.to_datetime(processed_goes["time_utc"], utc=True)

    task_config = load_sequence_task_config("flare_forecast_sequence")
    target_col = task_config["target_column"]
    sequences_config_name = "goes_sequences_with_sharp.yaml" if include_sharp else "goes_sequences.yaml"
    sequences_config = load_goes_sequences_config(SOLAR_ROOT / "configs" / sequences_config_name)
    preprocessing_config = load_goes_preprocessing_config()
    extra_gap_flag_columns = (SHARP_SEQ_MISSING_COL,) if include_sharp else ()

    # Mirrors modeling/sequence_train.py::run_sequence_task exactly (real,
    # already-tested logic) -- only the persistence step is new here.
    labeled = dataset_with_split.dropna(subset=[target_col]).reset_index(drop=True)
    X, meta = build_goes_sequences(
        processed_goes, labeled["time_utc"],
        sequences_config=sequences_config, preprocessing_config=preprocessing_config,
        extra_gap_flag_columns=extra_gap_flag_columns,
    )
    feasible_mask = meta["feasible"].to_numpy()
    joined = labeled.loc[feasible_mask].reset_index(drop=True)
    joined["time_utc"] = pd.to_datetime(joined["time_utc"], utc=True)

    counts = {}
    for split_name in ("train", "val", "test"):
        split_mask = (joined["split"] == split_name).to_numpy()
        X_split = X[split_mask].astype(np.float32)
        y_split = joined.loc[split_mask, target_col].to_numpy(dtype=np.float32)
        np.save(out_dir / f"X_{split_name}.npy", X_split)
        np.save(out_dir / f"y_{split_name}.npy", y_split)
        counts[split_name] = len(y_split)
        if split_name == "test":
            regime_test = joined.loc[split_mask, "time_utc"].dt.year.to_numpy()
            np.save(out_dir / "regime_test.npy", regime_test)

    print(f"Exported real solar fusion sequences to {out_dir}: {counts}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export real GOES sequence tensors for cross-domain fusion training.")
    parser.add_argument("--base-run-name", default="real_full_20100501_20250724")
    parser.add_argument("--sequence-run-name", default="real_full_sequence_20100501_20250724")
    parser.add_argument(
        "--include-sharp", action="store_true",
        help="Export the GOES+SHARP combined representation (requires the sequence run to have "
             "been produced by run_pipeline_sequence.py --include-sharp).",
    )
    args = parser.parse_args(argv)
    try:
        export_fusion_sequences(args.base_run_name, args.sequence_run_name, include_sharp=args.include_sharp)
    except Exception as exc:  # noqa: BLE001 -- top-level CLI boundary
        print(f"EXPORT FAILED: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
