"""Generic per-task train/evaluate/checkpoint/figure driver, covering the
full model hierarchy for one task: baselines -> classical models ->
sequence models (LSTM/GRU, gated by a feasibility report). Every trained
model -- regardless of kind -- gets its own checkpoint and its own
provenance record (modeling/provenance.py): exact parameters, random seed,
software versions, training history, feature order, train/val sample
counts, convergence status, runtime, and hardware info.

Every baseline and classical model is additionally evaluated on ALL THREE
splits (train/val/test) with confidence intervals, the full required figure
suite, failure analysis, and a comparison against the task's "appropriate
baseline" -- see evaluation/evaluate_task.py, called once per model
immediately after it is fit (so the already-fitted model/imputer never need
reconstructing from a checkpoint). Sequence models get metrics + CIs on
train/val/test (val used ONLY for early stopping inside
modeling/sequence_models.py::train_sequence_model, never for the reported
metrics selection) but skip the full figure/failure-analysis battery (their
sample sizes are typically far too small for either to be meaningful -- see
planning/MODELING_DOCUMENTATION.md).

Each call to `run_task` operates on exactly ONE task's own dataset, own
target column(s), own configuration, own feature manifest, own checkpoints,
own results, own figures, and own seed -- see
planning/MODELING_DOCUMENTATION.md for the cross-task-separation and
no-leakage argument.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

from data_sources.schema import CONFIG_DIR
from splitting.chronological_split import assign_split, load_splits_config

from . import baselines
from .provenance import Timer, TrainingRecord, write_training_record
from .metrics import select_threshold
from .registry import get_model
from .sequence_feasibility import run_feasibility_report
from .sequence_models import build_sequence_dataset, train_sequence_model

MARTIAN_ROOT = Path(__file__).resolve().parent.parent


def load_task_config(task_name: str, config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / f"task_{task_name}.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _horizon_frame(dataset_df: pd.DataFrame, feature_manifest: list[str], target_col: str) -> pd.DataFrame:
    metadata_cols = [c for c in dataset_df.columns if c not in feature_manifest and not c.startswith("target_")]
    cols = metadata_cols + feature_manifest + [target_col]
    frame = dataset_df[cols].dropna(subset=[target_col]).reset_index(drop=True)
    return frame


def _impute_train_fit(train_X: pd.DataFrame, apply_X: pd.DataFrame) -> tuple[np.ndarray, SimpleImputer]:
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_X)
    return imputer.transform(apply_X), imputer


def _impute_sequence_train_only(X_train_seq: np.ndarray, *other_seqs: np.ndarray) -> tuple[np.ndarray, ...]:
    """Median-impute NaN feature values in (n_samples, window, n_features)
    sequence arrays, statistics fit on X_train_seq only, applied to
    X_train_seq and every array in `other_seqs` (val, test, ...). Falls back
    to 0.0 for a feature that is entirely NaN across all of train (never
    drops a column -- see the caller's comment for why that matters)."""
    n_features = X_train_seq.shape[2]
    flat_train = X_train_seq.reshape(-1, n_features)
    with warnings.catch_warnings():
        # An entirely-NaN column legitimately triggers numpy's "All-NaN
        # slice encountered" warning -- expected and handled by the
        # fallback below, not silently swallowing a real problem.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        col_medians = np.nanmedian(flat_train, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)

    def _apply(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(-1, n_features).copy()
        nan_rows, nan_cols = np.where(np.isnan(flat))
        flat[nan_rows, nan_cols] = col_medians[nan_cols]
        return flat.reshape(arr.shape)

    return (_apply(X_train_seq),) + tuple(_apply(a) for a in other_seqs)


def _model_convergence_and_history(model) -> tuple[str, list[dict]]:
    """Best-effort extraction of an iterative model's training history and
    convergence status -- never fabricated: a non-iterative model (e.g.
    RandomForest, KNN) is honestly reported as such."""
    if hasattr(model, "train_score_"):  # GradientBoosting*
        history = [{"iteration": i, "train_score": float(s)} for i, s in enumerate(model.train_score_)]
        return "converged (fixed n_estimators reached)", history
    if hasattr(model, "n_iter_"):  # LogisticRegression
        n_iter = model.n_iter_
        n_iter_value = int(np.max(n_iter)) if hasattr(n_iter, "__len__") else int(n_iter)
        max_iter = getattr(model, "max_iter", None)
        status = "converged" if (max_iter is None or n_iter_value < max_iter) else "max_iter_reached"
        return status, [{"iteration": n_iter_value}]
    return "n/a (non-iterative model)", []


def _predict(model, df_or_X, *, task_type: str):
    if task_type == "regression":
        return model.predict(df_or_X)
    return model.predict_proba(df_or_X)[:, 1]


def _build_split_frames_baseline(model, task_type: str, splits: dict) -> dict:
    frames = {}
    for split_name, split_df in splits.items():
        if len(split_df) == 0:
            continue
        pred = _predict(model, split_df, task_type=task_type)
        frames[split_name] = split_df.assign(_prediction=pred)
    return frames


def _build_split_frames_classical(model, task_type: str, splits: dict, feature_manifest: list[str], imputer: SimpleImputer) -> dict:
    frames = {}
    for split_name, split_df in splits.items():
        if len(split_df) == 0:
            continue
        X = imputer.transform(split_df[feature_manifest])
        pred = _predict(model, X, task_type=task_type)
        frames[split_name] = split_df.assign(_prediction=pred)
    return frames


def _save_and_record(
    *, task_name, model_kind, model_name, horizon, model, exact_parameters, seed,
    feature_order, n_train, n_val, convergence_status, runtime_seconds, training_history,
    checkpoint_dir: Path, results_dir: Path, checkpoint_suffix=".joblib", saver=joblib.dump,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_h{horizon}{checkpoint_suffix}"
    saver(model, checkpoint_path)

    record = TrainingRecord(
        task_name=task_name, model_kind=model_kind, model_name=model_name, horizon_sols=horizon,
        exact_parameters=exact_parameters, random_seed=seed, feature_order=feature_order,
        n_train=n_train, n_val=n_val, convergence_status=convergence_status,
        runtime_seconds=runtime_seconds, training_history=training_history,
        checkpoint_path=str(checkpoint_path),
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    write_training_record(results_dir / f"{model_name}_h{horizon}_provenance.json", record)
    return checkpoint_path


def run_task(
    task_name: str,
    dataset_df: pd.DataFrame,
    feature_manifest: list[str],
    task_config: dict,
    *,
    features_df: pd.DataFrame | None = None,
    episode_catalogue: pd.DataFrame | None = None,
    splits_config: dict | None = None,
    eval_config: dict | None = None,
    project_root: Path | None = None,
    tables_dir: Path | None = None,
    data_source_label: str = "REAL (Martian Chapter 4 modeling)",
) -> dict:
    # Imported here (not at module top) to avoid a hard import-time
    # dependency of modeling/ on evaluation/ for callers that only need
    # fitting, not full reporting.
    from evaluation.confidence_intervals import load_evaluation_config
    from evaluation.evaluate_task import evaluate_and_report

    project_root = project_root or MARTIAN_ROOT
    splits_config = splits_config or load_splits_config()
    eval_config = eval_config or load_evaluation_config()
    seed = task_config["seed"]
    task_type = task_config["task_type"]
    # Eq. 6.1 (2026-10-04): per-model tau* selected on VALIDATION TSS; the
    # configured decision_threshold is only the documented fallback when
    # validation cannot define TSS. Previously every classifier used the fixed
    # 0.5 -- see chapter6.md Section 12 / martian_harmonisation_target_study.md.
    fallback_threshold = task_config.get("decision_threshold", 0.5)
    threshold_selection = task_config.get("threshold_selection", {"enabled": False})

    def _choose_threshold(split_frames: dict, target_col: str) -> tuple[float, str]:
        if task_type == "regression":
            return fallback_threshold, "n/a (regression)"
        if not threshold_selection.get("enabled", False):
            return fallback_threshold, "fixed (threshold_selection disabled)"
        val = split_frames.get("val")
        if val is None or len(val) == 0:
            return fallback_threshold, "fixed_fallback (empty validation split)"
        t, reason = select_threshold(
            val[target_col].to_numpy(), val["_prediction"].to_numpy(),
            grid_min=threshold_selection.get("grid_min", 0.01), grid_max=threshold_selection.get("grid_max", 0.99),
            grid_steps=threshold_selection.get("grid_steps", 99),
        )
        return (fallback_threshold, f"fixed_fallback ({reason})") if t is None else (t, reason)
    primary_baseline_name = eval_config.get("primary_baseline_by_task", {}).get(task_name)
    config_paths = [f"configs/task_{task_name}.yaml", "configs/evaluation.yaml"]

    paths = {k: project_root / v for k, v in task_config["paths"].items()}
    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    paths["figures_dir"].mkdir(parents=True, exist_ok=True)
    paths["results"].parent.mkdir(parents=True, exist_ok=True)

    dataset_df.to_csv(paths["dataset"], index=False)
    paths["feature_manifest"].write_text(
        json.dumps({"task_name": task_name, "feature_manifest": feature_manifest}, indent=2), encoding="utf-8"
    )

    sequence_config = task_config.get("sequence", {})
    if sequence_config.get("enabled") and features_df is not None and "sequence_feasibility_report" in paths:
        window_lengths = sorted(set([sequence_config["window_length_primary"]] + sequence_config["window_length_sensitivity"]))
        feasibility_report = run_feasibility_report(features_df, window_lengths)
        feasibility_report.to_csv(paths["sequence_feasibility_report"], index=False)

    all_results: dict = {"task_name": task_name, "task_type": task_type, "seed": seed, "horizons": {}}

    for h in task_config["horizons_sols"]:
        target_col = f"{task_config['target_column_prefix']}{h}"
        frame = _horizon_frame(dataset_df, feature_manifest, target_col)
        frame["split"] = assign_split(frame, splits_config)

        train = frame.loc[frame["split"] == "train"]
        val = frame.loc[frame["split"] == "val"]
        test = frame.loc[frame["split"] == "test"]
        splits = {"train": train, "val": val, "test": test}

        horizon_result: dict = {
            "target_column": target_col,
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "models": {},
        }

        if len(train) < 2 or len(test) == 0:
            horizon_result["status"] = "insufficient_data"
            all_results["horizons"][str(h)] = horizon_result
            continue
        horizon_result["status"] = "trained"

        y_train = train[target_col].to_numpy()
        primary_baseline_metrics = None

        # ---- Baselines ----
        for spec in task_config.get("baselines", []):
            with Timer() as timer:
                model = baselines.build_baseline(spec["name"], spec["hyperparameters"])
                model.fit(train, y_train)
            split_frames = _build_split_frames_baseline(model, task_type, splits)
            checkpoint_path = _save_and_record(
                task_name=task_name, model_kind="baseline", model_name=spec["name"], horizon=h,
                model=model, exact_parameters=model.get_params(), seed=seed,
                feature_order=[], n_train=len(train), n_val=len(val),
                convergence_status="n/a (non-iterative baseline)", runtime_seconds=timer.elapsed_seconds,
                training_history=[], checkpoint_dir=paths["checkpoint_dir"], results_dir=paths["results"].parent,
            )
            threshold, threshold_source = _choose_threshold(split_frames, target_col)
            eval_result = evaluate_and_report(
                task_name=task_name, task_type=task_type, model_kind="baseline", model_name=spec["name"], horizon=h,
                split_frames=split_frames, target_col=target_col, prediction_col="_prediction", threshold=threshold, threshold_source=threshold_source,
                baseline_metrics=None, episode_catalogue=episode_catalogue,
                figures_dir=paths["figures_dir"], results_dir=paths["results"].parent,
                config_paths=config_paths, full_report=True, eval_config=eval_config,
                tables_dir=tables_dir, data_source_label=data_source_label,
            )
            test_metrics = eval_result["splits"].get("test", {}).get("metrics")
            if spec["name"] == primary_baseline_name:
                primary_baseline_metrics = test_metrics
            horizon_result["models"][spec["name"]] = {
                "kind": "baseline", "metrics": test_metrics, "checkpoint": str(checkpoint_path),
                "evaluation": eval_result,
            }

        # ---- Classical models ----
        X_train_raw = train[feature_manifest]
        _, imputer = _impute_train_fit(X_train_raw, X_train_raw)

        for spec in task_config.get("classical_models", []):
            with Timer() as timer:
                model = get_model(task_type, spec["name"], spec["hyperparameters"], seed)
                model.fit(imputer.transform(X_train_raw), y_train)
            split_frames = _build_split_frames_classical(model, task_type, splits, feature_manifest, imputer)
            convergence_status, history = _model_convergence_and_history(model)
            checkpoint_path = _save_and_record(
                task_name=task_name, model_kind="classical", model_name=spec["name"], horizon=h,
                model=model, exact_parameters=model.get_params(), seed=seed,
                feature_order=feature_manifest, n_train=len(train), n_val=len(val),
                convergence_status=convergence_status, runtime_seconds=timer.elapsed_seconds,
                training_history=history, checkpoint_dir=paths["checkpoint_dir"], results_dir=paths["results"].parent,
            )
            threshold, threshold_source = _choose_threshold(split_frames, target_col)
            eval_result = evaluate_and_report(
                task_name=task_name, task_type=task_type, model_kind="classical", model_name=spec["name"], horizon=h,
                split_frames=split_frames, target_col=target_col, prediction_col="_prediction", threshold=threshold, threshold_source=threshold_source,
                baseline_metrics=primary_baseline_metrics, episode_catalogue=episode_catalogue,
                figures_dir=paths["figures_dir"], results_dir=paths["results"].parent,
                config_paths=config_paths, full_report=True, eval_config=eval_config,
                tables_dir=tables_dir, data_source_label=data_source_label,
            )
            horizon_result["models"][spec["name"]] = {
                "kind": "classical", "metrics": eval_result["splits"].get("test", {}).get("metrics"),
                "checkpoint": str(checkpoint_path), "evaluation": eval_result,
            }

        # ---- Sequence models (LSTM/GRU), gated by feasibility ----
        # Early stopping uses VAL, never TEST (fixed 2026-08-13 -- the
        # original version of this block passed the test split into
        # train_sequence_model's early-stopping argument, a real form of
        # test-set leakage documented in CHAPTER_4_PARAMETER_VALUES.md
        # caveat 9 / Section 11 item 5. `configs/splits.yaml`'s val window
        # was empty in that run; it is not empty now (see that file's own
        # 2026-08-13 comment), so this is enforced structurally below by
        # requiring a non-empty val window before any training happens, not
        # merely by using the right variable name.
        if sequence_config.get("enabled") and features_df is not None:
            window_lengths = sorted(set([sequence_config["window_length_primary"]] + sequence_config["window_length_sensitivity"]))
            seq_summary = {}
            for w in window_lengths:
                X_train_seq, y_train_seq, meta_train = build_sequence_dataset(
                    train, features_df, feature_manifest, window_length_sols=w, target_col=target_col
                )
                X_val_seq, y_val_seq, meta_val = build_sequence_dataset(
                    val, features_df, feature_manifest, window_length_sols=w, target_col=target_col
                )
                X_test_seq, y_test_seq, meta_test = build_sequence_dataset(
                    test, features_df, feature_manifest, window_length_sols=w, target_col=target_col
                )
                if len(X_train_seq) == 0 or len(X_val_seq) == 0 or len(X_test_seq) == 0:
                    seq_summary[str(w)] = {
                        "status": "infeasible",
                        "n_train_windows": len(X_train_seq),
                        "n_val_windows": len(X_val_seq),
                        "n_test_windows": len(X_test_seq),
                    }
                    continue

                # Impute NaN feature values (e.g. sparse spatial-neighbour
                # stats) fit on TRAIN windows only, same discipline as the
                # tabular classical-model path -- never on sol-adjacency,
                # which build_sequence_dataset already enforces structurally.
                # NOT sklearn's SimpleImputer here: it silently DROPS a
                # column that is entirely NaN in train rather than filling
                # it, which changes the array's column count and breaks the
                # (n_samples, window, n_features) reshape -- caught when a
                # real run crashed with a reshape ValueError, see
                # planning/MODELING_DOCUMENTATION.md. A manual, shape-
                # preserving median-with-zero-fallback avoids that entirely.
                X_train_seq, X_val_seq, X_test_seq = _impute_sequence_train_only(X_train_seq, X_val_seq, X_test_seq)

                for model_spec in sequence_config["models"]:
                    with Timer() as timer:
                        model, history, convergence_status, _ = train_sequence_model(
                            X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                            rnn_type=model_spec["rnn_type"], task_type=task_type,
                            hyperparameters=model_spec["hyperparameters"],
                            training_config=sequence_config["training"], seed=seed,
                        )
                    import torch
                    with torch.no_grad():
                        train_output = model(torch.tensor(X_train_seq, dtype=torch.float32)).numpy()
                        val_output = model(torch.tensor(X_val_seq, dtype=torch.float32)).numpy()
                        test_output = model(torch.tensor(X_test_seq, dtype=torch.float32)).numpy()
                    if task_type != "regression":
                        train_output = 1 / (1 + np.exp(-train_output))
                        val_output = 1 / (1 + np.exp(-val_output))
                        test_output = 1 / (1 + np.exp(-test_output))

                    model_name = f"{model_spec['rnn_type']}_w{w}"
                    checkpoint_dir = paths["checkpoint_dir"]
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / f"{model_name}_h{h}.pt"
                    torch.save(model.state_dict(), checkpoint_path)
                    record = TrainingRecord(
                        task_name=task_name, model_kind="sequence", model_name=model_name, horizon_sols=h,
                        exact_parameters={**model_spec["hyperparameters"], "window_length_sols": w},
                        random_seed=seed, feature_order=feature_manifest,
                        n_train=len(X_train_seq), n_val=len(X_val_seq),
                        convergence_status=convergence_status, runtime_seconds=timer.elapsed_seconds,
                        training_history=history, checkpoint_path=str(checkpoint_path),
                    )
                    write_training_record(paths["results"].parent / f"{model_name}_h{h}_provenance.json", record)

                    seq_split_frames = {
                        "train": meta_train.assign(**{target_col: y_train_seq, "_prediction": train_output}),
                        "val": meta_val.assign(**{target_col: y_val_seq, "_prediction": val_output}),
                        "test": meta_test.assign(**{target_col: y_test_seq, "_prediction": test_output}),
                    }
                    threshold, threshold_source = _choose_threshold(seq_split_frames, target_col)
                    eval_result = evaluate_and_report(
                        task_name=task_name, task_type=task_type, model_kind="sequence", model_name=model_name, horizon=h,
                        split_frames=seq_split_frames, target_col=target_col, prediction_col="_prediction",
                        threshold=threshold, threshold_source=threshold_source,
                        baseline_metrics=primary_baseline_metrics, episode_catalogue=None,
                        figures_dir=paths["figures_dir"], results_dir=paths["results"].parent,
                        config_paths=config_paths, full_report=False, eval_config=eval_config,
                        tables_dir=tables_dir, data_source_label=data_source_label,
                    )
                    seq_summary.setdefault(str(w), {"status": "trained", "models": {}})
                    seq_summary[str(w)].setdefault("models", {})[model_spec["rnn_type"]] = {
                        "metrics": eval_result["splits"]["test"]["metrics"], "checkpoint": str(checkpoint_path),
                        "n_train_windows": len(X_train_seq),
                        "n_val_windows": len(X_val_seq),
                        "n_test_windows": len(X_test_seq),
                        "evaluation": eval_result,
                    }
            horizon_result["sequence"] = seq_summary

        all_results["horizons"][str(h)] = horizon_result

    paths["results"].write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    return all_results
