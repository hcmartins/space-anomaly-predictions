"""One-off inference export for Chapter 3 examiner round-12 follow-up:
reload the two real, already-trained LSTM checkpoints (GOES-only and
GOES+SHARP) and write a per-test-instance CSV of predicted probabilities
for each, for the common-subset comparison + clustered-bootstrap analysis
requested against Section 3.8.

Reconstructs the exact same sequence-feasible test tensors
run_pipeline_sequence.py/modeling/sequence_train.py::run_sequence_task
built originally (same base run's QC'd GOES rows + labeled dataset, same
Algorithm 3.1 preprocessing, same disk-aggregated SHARP merge for the
combined variant) -- never retrains, only re-derives the deterministic
preprocessing and runs the frozen checkpoint forward. Cross-validates the
reconstructed (tp, fp, tn, fn) at the model's own recorded threshold
against run_summary.json before writing anything.

Run as two SEPARATE process invocations (--variant goes / --variant
goes_sharp), not both in one process: the combined (T=30, d=20) variant
needs the full ~1GB sharp_qc_accepted.csv plus the ~1.7GB goes_qc_accepted.csv
and ~1.9GB dataset_with_split.csv in memory at once on a machine with a real,
documented ~15.6GB ceiling (see preprocessing/scaling.py's own ArrayMemoryError
note) -- running the GOES-only variant first in the same process left enough
fragmentation/residual memory to make the SHARP variant fail even after both
variants' own local frames had gone out of scope. A fresh process for each
variant sidesteps this entirely.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_sources.schema import SHARP_PHYSICAL_KEYWORDS
from modeling.sequence_models import load_sequence_checkpoint
from preprocessing.alignment import aggregate_goes_disk, load_alignment_config
from preprocessing.goes_preprocessing import load_goes_preprocessing_config, preprocess_goes_sequence
from preprocessing.goes_sequences import build_goes_sequences, load_goes_sequences_config
from preprocessing.sharp_sequences import SHARP_SEQ_MISSING_COL, build_combined_sequence_table
from splitting.chronological_split import load_splits_config

SOLAR_ROOT = Path(__file__).resolve().parent
BASE_DIR = SOLAR_ROOT / "results" / "real_full_20100501_20250724"
OUT_DIR = SOLAR_ROOT / "results" / "prediction_exports"
TARGET_COL = "label_primary"

GOES_QC_USECOLS = ["time_utc", "xrsa_flux_w_m2", "xrsb_flux_w_m2", "satellite"]
SHARP_QC_USECOLS = ["HARPNUM", "T_REC"] + SHARP_PHYSICAL_KEYWORDS
DATASET_USECOLS = ["time_utc", "split", TARGET_COL]


def _prep(include_sharp: bool):
    goes_qc_accepted = pd.read_csv(BASE_DIR / "goes_qc_accepted.csv", usecols=GOES_QC_USECOLS)
    alignment_config = load_alignment_config()
    goes_disk = aggregate_goes_disk(goes_qc_accepted, alignment_config)
    del goes_qc_accepted
    gc.collect()

    preprocessing_config = load_goes_preprocessing_config()
    processed_goes = preprocess_goes_sequence(goes_disk, config=preprocessing_config)
    del goes_disk
    gc.collect()

    extra_gap_flag_columns: tuple[str, ...] = ()
    if include_sharp:
        sharp_qc_accepted = pd.read_csv(BASE_DIR / "sharp_qc_accepted.csv", usecols=SHARP_QC_USECOLS)
        splits_config = load_splits_config()
        processed_goes, _ = build_combined_sequence_table(
            processed_goes, sharp_qc_accepted,
            train_start=splits_config["train_start"], train_end=splits_config["train_end"],
            alignment_config=alignment_config,
        )
        del sharp_qc_accepted
        gc.collect()
        extra_gap_flag_columns = (SHARP_SEQ_MISSING_COL,)

    sequences_config_name = "goes_sequences_with_sharp.yaml" if include_sharp else "goes_sequences.yaml"
    sequences_config = load_goes_sequences_config(SOLAR_ROOT / "configs" / sequences_config_name)

    dataset_with_split = pd.read_csv(BASE_DIR / "dataset_with_split.csv", usecols=DATASET_USECOLS)
    labeled = dataset_with_split.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    del dataset_with_split
    gc.collect()
    # Restrict to the test split BEFORE building sequence tensors: this
    # export only needs test-split predictions. Windowing only depends on
    # each row's own trailing 6h GOES/SHARP history via processed_goes
    # (untouched, full series), so restricting the issuance-time list to
    # test rows changes nothing about which windows are feasible or what
    # values they contain -- only how many are built at once.
    labeled = labeled.loc[labeled["split"] == "test"].reset_index(drop=True)
    X, meta = build_goes_sequences(
        processed_goes, labeled["time_utc"],
        sequences_config=sequences_config, preprocessing_config=preprocessing_config,
        extra_gap_flag_columns=extra_gap_flag_columns,
    )
    del processed_goes
    gc.collect()
    feasible_mask = meta["feasible"].to_numpy()
    joined = labeled.loc[feasible_mask].reset_index(drop=True)
    return X, joined


def _run_one(*, include_sharp: bool, run_name: str, out_name: str) -> None:
    print(f"\n=== {out_name} (include_sharp={include_sharp}) ===", flush=True)
    X, joined = _prep(include_sharp)

    # joined is already restricted to the test split by _prep, so this
    # mask is all-True -- kept explicit for clarity/defensiveness.
    test_mask = (joined["split"] == "test").to_numpy()
    X_test = X[test_mask]
    joined_test = joined.loc[test_mask].reset_index(drop=True)
    y_test = joined_test[TARGET_COL].to_numpy(dtype=float)
    print(f"n_test sequences: {len(y_test)}", flush=True)

    checkpoint_path = SOLAR_ROOT / "checkpoints" / run_name / f"lstm_{TARGET_COL}.pt"
    model = load_sequence_checkpoint(checkpoint_path)
    model.eval()
    with torch.no_grad():
        prob = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    run_summary = json.loads((SOLAR_ROOT / "results" / run_name / "run_summary.json").read_text(encoding="utf-8"))
    lstm_result = run_summary["task_results"]["sequence_models"]["lstm"]
    threshold = lstm_result["decision_threshold"]
    rec = lstm_result["metrics"]["test"]

    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    match = (tp == rec["tp"]) and (fp == rec["fp"]) and (tn == rec["tn"]) and (fn == rec["fn"])
    status = "OK" if match else "MISMATCH"
    print(
        f"threshold={threshold:.3f} tp={tp} fp={fp} tn={tn} fn={fn}  "
        f"(recorded tp={rec['tp']} fp={rec['fp']} tn={rec['tn']} fn={rec['fn']})  [{status}]",
        flush=True,
    )
    if not match:
        raise SystemExit(f"Reconstructed confusion matrix for {out_name!r} does not match run_summary.json -- refusing to export.")

    out = pd.DataFrame({
        "time_utc": joined_test["time_utc"],
        "true_label": y_test.astype(int),
        "predicted_probability": prob,
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{out_name}_predictions_test.csv"
    out.to_csv(out_path, index=False)
    (OUT_DIR / f"{out_name}_threshold.json").write_text(json.dumps({"threshold": threshold}, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(out)} rows)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["goes", "goes_sharp"], required=True)
    args = parser.parse_args()

    if args.variant == "goes":
        _run_one(include_sharp=False, run_name="real_full_sequence_20100501_20250724", out_name="goes_lstm")
    else:
        _run_one(include_sharp=True, run_name="real_full_sequence_goes_sharp_20100501_20250724", out_name="goes_sharp_lstm")


if __name__ == "__main__":
    main()
