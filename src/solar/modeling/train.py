"""Fit/evaluate driver for the flare-occurrence forecasting task: baselines
-> classical models, each fit on TRAIN ONLY and evaluated on train/val/test.

Scoped down from `martian/modeling/train.py`'s equivalent: no sequence
(LSTM/GRU) tier (no torch dependency or windowing infrastructure exists for
solar yet), and no `evaluation/`-package integration (confidence intervals,
figures, failure analysis, baseline-comparison reporting) or
`modeling/provenance.py`-style full `TrainingRecord` schema -- this module
writes a plain metrics JSON and joblib checkpoints only. See
`planning/MODELING_DOCUMENTATION.md`'s "What is NOT implemented" section for
the precise, honest boundary; nothing here should be read as a
feature-complete substitute for martian's modeling stack..

Splitting: `run_task` calls `modeling.datasets.assign_split_with_embargo`
ONCE per dataset (not per target column) -- the embargo horizon is read
from the dataset's own `label_horizon_hours` column, so it always matches
whatever `labeling/flare_labels.py` run actually produced this dataset's
labels (see that function's docstring).
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

from data_sources.schema import CONFIG_DIR

from preprocessing.scaling import apply_robust_scaling, fit_robust_scaling, load_scaling_config, sharp_feature_columns

from . import baselines
from .datasets import TARGET_COLUMNS, assign_split_with_embargo
from .interpretability import extract_interpretability
from .metrics import classification_metrics, select_threshold
from .registry import get_model

SOLAR_ROOT = Path(__file__).resolve().parent.parent


def load_task_config(task_name: str = "flare_forecast", config_path: Path | None = None) -> dict:
    config_path = config_path or CONFIG_DIR / f"task_{task_name}.yaml"
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _prepare_frame(dataset_with_split: pd.DataFrame, feature_manifest: list[str], target_col: str) -> pd.DataFrame:
    """Restricts to metadata + feature_manifest + this ONE target_col
    (sibling target_* columns are dropped, mirroring
    martian/modeling/train.py's `_horizon_frame` -- never let another
    target column ride along as if it were a feature or metadata), then
    drops rows where target_col is UNDEFINED (NaN) -- never trained as a
    fabricated negative, per planning/LABELING_DOCUMENTATION.md."""
    other_targets = [c for c in TARGET_COLUMNS if c in dataset_with_split.columns and c != target_col]
    metadata_cols = [c for c in dataset_with_split.columns if c not in feature_manifest and c not in other_targets]
    cols = [c for c in metadata_cols if c != target_col] + feature_manifest + [target_col]
    frame = dataset_with_split[cols].dropna(subset=[target_col]).reset_index(drop=True)
    return frame


def _predict_proba_positive(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _predict_proba_by_split_baseline(model, splits: dict) -> dict:
    return {
        name: _predict_proba_positive(model, split_df)
        for name, split_df in splits.items()
        if len(split_df) > 0
    }


def _predict_proba_by_split_classical(
    model, splits: dict, feature_manifest: list[str], imputer: SimpleImputer, scaler_params: dict
) -> dict:
    return {
        name: _predict_proba_positive(
            model, imputer.transform(apply_robust_scaling(split_df[feature_manifest], scaler_params, copy=False))
        )
        for name, split_df in splits.items()
        if len(split_df) > 0
    }


def _select_threshold_and_evaluate(
    y_prob_by_split: dict, splits: dict, target_col: str, task_config: dict
) -> tuple[float, str, dict]:
    """Selects tau* on the VALIDATION split only (Chapter 3.6.2/3.8's
    documented protocol), then evaluates every split at that one threshold.
    Falls back to task_config's static decision_threshold -- documented as
    the reason why -- if val is too small or single-class to select from,
    never silently substituting test or an arbitrary value."""
    threshold_cfg = task_config.get("threshold_selection", {})
    metric = threshold_cfg.get("metric", "tss")
    fallback_threshold = task_config.get("decision_threshold", 0.5)

    threshold = None
    if "val" in y_prob_by_split:
        y_val_true = splits["val"][target_col].to_numpy()
        threshold, _ = select_threshold(
            y_val_true,
            y_prob_by_split["val"],
            metric=metric,
            grid_min=threshold_cfg.get("grid_min", 0.01),
            grid_max=threshold_cfg.get("grid_max", 0.99),
            grid_steps=threshold_cfg.get("grid_steps", 99),
        )

    if threshold is None:
        threshold = fallback_threshold
        threshold_source = "fixed_fallback (val split too small or single-class to select a threshold from)"
    else:
        threshold_source = f"validation_selected (maximizing {metric})"

    metrics_by_split = {
        split_name: classification_metrics(splits[split_name][target_col], y_prob, threshold=threshold)
        for split_name, y_prob in y_prob_by_split.items()
    }
    return threshold, threshold_source, metrics_by_split


def run_task(
    dataset: pd.DataFrame,
    feature_manifest: list[str],
    task_config: dict,
    *,
    splits_config: dict | None = None,
    project_root: Path | None = None,
    imputer_fit_sample_size: int | None = None,
) -> dict:
    """imputer_fit_sample_size: if set, SimpleImputer's median/mean
    statistics are fit on a random sample of this many TRAIN rows instead
    of the full train split -- statistically near-equivalent (imputation
    statistics stabilize with far fewer than millions of rows) but avoids
    sklearn's masked-array median implementation needing an internal
    sort/copy of the full train matrix, which hit a real
    ArrayMemoryError on the per-HARP pipeline's ~1.97M-row train split at
    this machine's 15.6GB RAM ceiling (see
    planning/FLARE_CATALOGUE_DOCUMENTATION.md). Default None preserves
    the exact original full-train-set behavior -- the disk-integrated
    pipeline's real, already-reported numbers never pass this and are
    unaffected.

    Classical models (not baselines) are always fit on SHARP-robust-scaled
    features -- see preprocessing/scaling.py -- fit on TRAIN only, applied
    to val/test before imputation. Unconditional (no opt-out flag): this is
    a Chapter 3.4.2-required preprocessing step, not a memory workaround,
    and a no-op for feature manifests with no SHARP-derived columns."""
    project_root = project_root or SOLAR_ROOT
    seed = task_config["seed"]

    dataset_with_split = assign_split_with_embargo(dataset, splits_config)
    # `dataset` (the input param) is never read again below -- dropping our
    # own reference here lets it be freed as soon as the caller drops
    # theirs too, real memory relief at per-HARP scale (~13GB dataset) --
    # see run_task's docstring and preprocessing/scaling.py's
    # ArrayMemoryError note for the real crash this class of fix targets.
    del dataset
    gc.collect()

    paths = {k: project_root / v for k, v in task_config.get("paths", {}).items()}
    if "checkpoint_dir" in paths:
        paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    if "dataset" in paths:
        paths["dataset"].parent.mkdir(parents=True, exist_ok=True)
        dataset_with_split.to_csv(paths["dataset"], index=False)
    if "feature_manifest" in paths:
        paths["feature_manifest"].parent.mkdir(parents=True, exist_ok=True)
        paths["feature_manifest"].write_text(
            json.dumps({"task_name": task_config["task_name"], "feature_manifest": feature_manifest}, indent=2),
            encoding="utf-8",
        )
    if "results" in paths:
        paths["results"].parent.mkdir(parents=True, exist_ok=True)

    all_results: dict = {
        "task_name": task_config["task_name"],
        "task_type": task_config["task_type"],
        "seed": seed,
        "targets": {},
    }

    target_columns = task_config["target_columns"]
    for i, target_col in enumerate(target_columns):
        frame = _prepare_frame(dataset_with_split, feature_manifest, target_col)
        if i == len(target_columns) - 1:
            # Last target_col that will ever need dataset_with_split (the
            # full, all-targets, embargo-assigned frame -- comparable in
            # size to the whole input dataset). run_task is its sole owner
            # (built internally, never returned to the caller), so freeing
            # it here is guaranteed to actually release the memory, not
            # just drop one of several references.
            del dataset_with_split
            gc.collect()
        train = frame.loc[frame["split"] == "train"]
        val = frame.loc[frame["split"] == "val"]
        test = frame.loc[frame["split"] == "test"]
        splits = {"train": train, "val": val, "test": test}
        del frame
        gc.collect()

        target_result: dict = {
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "models": {},
        }

        if len(train) < 2 or len(test) == 0:
            target_result["status"] = "insufficient_data"
            all_results["targets"][target_col] = target_result
            continue

        y_train = train[target_col].to_numpy()
        n_classes_train = len(np.unique(y_train))
        target_result["status"] = "trained"
        target_result["n_classes_in_train"] = n_classes_train

        # ---- Baselines ----
        for spec in task_config.get("baselines", []):
            start = time.perf_counter()
            model = baselines.build_baseline(spec["name"], spec["hyperparameters"])
            model.fit(train, y_train)
            runtime_seconds = time.perf_counter() - start

            y_prob_by_split = _predict_proba_by_split_baseline(model, splits)
            threshold_star, threshold_source, metrics_by_split = _select_threshold_and_evaluate(
                y_prob_by_split, splits, target_col, task_config
            )
            checkpoint_path = None
            if "checkpoint_dir" in paths:
                checkpoint_path = paths["checkpoint_dir"] / f"{spec['name']}_{target_col}.joblib"
                joblib.dump(model, checkpoint_path)

            target_result["models"][spec["name"]] = {
                "kind": "baseline",
                "parameters": model.get_params(),
                "runtime_seconds": runtime_seconds,
                "decision_threshold": threshold_star,
                "decision_threshold_source": threshold_source,
                "metrics": metrics_by_split,
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            }

        # ---- Classical models (skipped if train has only one class -- most
        # sklearn classifiers cannot fit on a single-class target; honestly
        # reported, never silently skipped without a trace) ----
        if n_classes_train < 2:
            target_result["classical_models_skipped_reason"] = (
                f"train split has only {n_classes_train} distinct class(es) for {target_col!r}; "
                f"classical classifiers require at least 2."
            )
        else:
            X_train_raw = train[feature_manifest]

            # SHARP robust scaling (Chapter 3.4.2, Algorithm 3.2): Winsorize
            # + median/IQR-scale the SHARP-derived feature columns, fit on
            # TRAIN only, applied to every split before imputation. See
            # preprocessing/scaling.py's module docstring for why (real,
            # repeated logistic-regression convergence failure). No-ops
            # (empty scaler_params) when feature_manifest has no
            # SHARP-derived columns, e.g. synthetic test datasets.
            scaler_params = fit_robust_scaling(
                X_train_raw, sharp_feature_columns(feature_manifest), load_scaling_config()
            )
            target_result["sharp_scaling"] = {"n_columns_scaled": len(scaler_params)}
            # copy=False: X_train_raw is a disposable local slice never read
            # again after this line -- mutating it in place avoids an extra
            # full-frame copy that hit a real ArrayMemoryError at the
            # per-HARP pipeline's ~1.97M-row train split (see
            # preprocessing/scaling.py::apply_robust_scaling's docstring).
            X_train_scaled = apply_robust_scaling(X_train_raw, scaler_params, copy=False)

            imputer = SimpleImputer(strategy="median")
            if imputer_fit_sample_size is not None and len(X_train_scaled) > imputer_fit_sample_size:
                imputer.fit(X_train_scaled.sample(n=imputer_fit_sample_size, random_state=seed))
            else:
                imputer.fit(X_train_scaled)

            for spec in task_config.get("classical_models", []):
                start = time.perf_counter()
                model = get_model(spec["name"], spec["hyperparameters"], seed)
                model.fit(imputer.transform(X_train_scaled), y_train)
                runtime_seconds = time.perf_counter() - start

                y_prob_by_split = _predict_proba_by_split_classical(model, splits, feature_manifest, imputer, scaler_params)
                threshold_star, threshold_source, metrics_by_split = _select_threshold_and_evaluate(
                    y_prob_by_split, splits, target_col, task_config
                )
                checkpoint_path = None
                if "checkpoint_dir" in paths:
                    checkpoint_path = paths["checkpoint_dir"] / f"{spec['name']}_{target_col}.joblib"
                    joblib.dump(model, checkpoint_path)

                target_result["models"][spec["name"]] = {
                    "kind": "classical",
                    "parameters": model.get_params(),
                    "runtime_seconds": runtime_seconds,
                    "decision_threshold": threshold_star,
                    "decision_threshold_source": threshold_source,
                    "metrics": metrics_by_split,
                    "interpretability": extract_interpretability(model, feature_manifest),
                    "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                }

        all_results["targets"][target_col] = target_result

    if "results" in paths:
        paths["results"].write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

    return all_results
