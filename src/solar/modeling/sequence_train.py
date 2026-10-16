"""Task orchestration for the GOES sequence LSTM (Chapter 3.6, Table 3.3).
Kept SEPARATE from modeling/train.py (the classical/baseline path) so that
already-tested, already-reported module stays completely untouched..
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data_sources.schema import CONFIG_DIR
from preprocessing.goes_preprocessing import load_goes_preprocessing_config
from preprocessing.goes_sequences import (
    assess_goes_sequence_feasibility,
    build_goes_sequences,
    load_goes_sequences_config,
    window_length_steps,
)

from .metrics import classification_metrics, select_threshold
from .sequence_models import save_sequence_checkpoint, train_sequence_model

SOLAR_ROOT = Path(__file__).resolve().parent.parent


def load_sequence_task_config(task_name: str = "flare_forecast_sequence", config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / f"task_{task_name}.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def run_sequence_task(
    processed_goes_df: pd.DataFrame,
    labeled_dataset: pd.DataFrame,
    task_config: dict,
    *,
    project_root: Path | None = None,
    sequences_config: dict | None = None,
    preprocessing_config: dict | None = None,
    extra_gap_flag_columns: tuple[str, ...] = (),
) -> dict:
    """`labeled_dataset` must carry `time_utc`, `split`, and
    `task_config['target_column']` columns -- reused directly from the
    classical pipeline's already-built, already-embargoed dataset
    (results/<run>/dataset_with_split.csv), never recomputed here: the
    real chronological split/embargo logic (splitting/chronological_split.py)
    is not duplicated for the sequence task, only its output is reused, so
    both tasks are evaluated on identical train/val/test partitions.

    Returns a JSON-shaped result under a `sequence_models` key (not
    `models`), so it never collides when merged into the same
    run_summary.json as modeling/train.py::run_task's classical results.

    `extra_gap_flag_columns`: forwarded unchanged to
    `assess_goes_sequence_feasibility`/`build_goes_sequences` (e.g.
    preprocessing/sharp_sequences.py's `SHARP_SEQ_MISSING_COL`) -- default
    empty means zero behavior change for every existing GOES-only caller."""
    project_root = project_root or SOLAR_ROOT
    sequences_config = sequences_config or load_goes_sequences_config()
    preprocessing_config = preprocessing_config or load_goes_preprocessing_config()
    target_col = task_config["target_column"]
    seed = task_config["seed"]

    paths = {k: project_root / v for k, v in task_config.get("paths", {}).items()}
    if "checkpoint_dir" in paths:
        paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    if "results" in paths:
        paths["results"].parent.mkdir(parents=True, exist_ok=True)

    window_steps = window_length_steps(sequences_config, preprocessing_config)
    feasibility_report = assess_goes_sequence_feasibility(
        processed_goes_df, window_steps, sequences_config=sequences_config,
        extra_gap_flag_columns=extra_gap_flag_columns,
    )
    if "feasibility_report" in paths:
        paths["feasibility_report"].parent.mkdir(parents=True, exist_ok=True)
        paths["feasibility_report"].write_text(json.dumps(feasibility_report, indent=2), encoding="utf-8")

    result: dict = {
        "task_name": task_config["task_name"],
        "task_type": task_config["task_type"],
        "seed": seed,
        "feasibility_report": feasibility_report,
    }

    # Negative results are first-class, reported outputs, never a silent
    # skip -- see preprocessing/goes_sequences.py's module docstring.
    if not feasibility_report["feasible"]:
        result["status"] = "infeasible"
        return result

    labeled = labeled_dataset.dropna(subset=[target_col]).reset_index(drop=True)
    X, meta = build_goes_sequences(
        processed_goes_df, labeled["time_utc"],
        sequences_config=sequences_config, preprocessing_config=preprocessing_config,
        extra_gap_flag_columns=extra_gap_flag_columns,
    )
    # build_goes_sequences preserves issuance_times' input order in both
    # its return values, so X (already filtered to feasible windows only)
    # and labeled.loc[meta['feasible']] are row-aligned by construction.
    feasible_mask = meta["feasible"].to_numpy()
    joined = labeled.loc[feasible_mask].reset_index(drop=True)

    result["n_issuance_times"] = int(len(labeled))
    result["n_feasible_sequences"] = int(len(joined))

    splits: dict[str, dict] = {}
    for split_name in ("train", "val", "test"):
        split_mask = (joined["split"] == split_name).to_numpy()
        splits[split_name] = {"X": X[split_mask], "y": joined.loc[split_mask, target_col].to_numpy(dtype=float)}
    result["n_train"] = len(splits["train"]["y"])
    result["n_val"] = len(splits["val"]["y"])
    result["n_test"] = len(splits["test"]["y"])

    if splits["train"]["X"].shape[0] < 2 or len(np.unique(splits["train"]["y"])) < 2:
        result["status"] = "insufficient_data"
        return result

    result["status"] = "trained"
    result["sequence_models"] = {}

    for spec in task_config["sequence"]["models"]:
        rnn_type = spec["rnn_type"]
        model, history, convergence_status, runtime_seconds = train_sequence_model(
            splits["train"]["X"], splits["train"]["y"],
            splits["val"]["X"], splits["val"]["y"],
            hyperparameters=spec["hyperparameters"],
            training_config=task_config["sequence"]["training"],
            seed=seed,
        )

        model.eval()
        y_prob_by_split = {}
        for split_name in ("train", "val", "test"):
            if len(splits[split_name]["X"]):
                with torch.no_grad():
                    y_prob_by_split[split_name] = model(
                        torch.tensor(splits[split_name]["X"], dtype=torch.float32)
                    ).numpy()

        threshold_cfg = task_config.get("threshold_selection", {})
        threshold = None
        if "val" in y_prob_by_split:
            threshold, _ = select_threshold(
                splits["val"]["y"], y_prob_by_split["val"],
                metric=threshold_cfg.get("metric", "tss"),
                grid_min=threshold_cfg.get("grid_min", 0.01),
                grid_max=threshold_cfg.get("grid_max", 0.99),
                grid_steps=threshold_cfg.get("grid_steps", 99),
            )
        if threshold is None:
            threshold = task_config.get("decision_threshold", 0.5)
            threshold_source = "fixed_fallback (val split too small or single-class to select a threshold from)"
        else:
            threshold_source = f"validation_selected (maximizing {threshold_cfg.get('metric', 'tss')})"

        metrics_by_split = {
            split_name: classification_metrics(splits[split_name]["y"], y_prob, threshold=threshold)
            for split_name, y_prob in y_prob_by_split.items()
        }

        checkpoint_path = None
        if "checkpoint_dir" in paths:
            checkpoint_path = paths["checkpoint_dir"] / f"{rnn_type}_{target_col}.pt"
            save_sequence_checkpoint(model, spec["hyperparameters"], checkpoint_path)

        result["sequence_models"][rnn_type] = {
            "kind": "sequence",
            "hyperparameters": spec["hyperparameters"],
            "convergence_status": convergence_status,
            "runtime_seconds": runtime_seconds,
            "training_history": history,
            "decision_threshold": threshold,
            "decision_threshold_source": threshold_source,
            "metrics": metrics_by_split,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        }

    if "results" in paths:
        paths["results"].write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    return result
