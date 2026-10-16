"""One-off inference export for Chapter 3 examiner round-12 follow-up:
reload each real, already-trained classical/baseline checkpoint
(checkpoints/real_full_20100501_20250724/*.joblib) and write a single
per-test-instance CSV of predicted probabilities, for the common-subset
comparison + clustered-bootstrap analysis requested against Section 3.8.

Does not retrain anything. Recomputes fit_robust_scaling/SimpleImputer
deterministically from the same train split modeling/train.py::run_task
used (both are pure functions of the data, no randomness), then applies
them to the test split before calling predict_proba -- exactly
modeling/train.py::_predict_proba_by_split_classical's own logic. Baselines
(climatology, persistence) are called directly on the unscaled split, per
_predict_proba_by_split_baseline. Cross-validates every model's
reconstructed (tp, fp, tn, fn) against metrics.json's already-recorded
values at that model's own reported threshold before writing anything, and
refuses to write output if any mismatch is found.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from preprocessing.scaling import apply_robust_scaling, fit_robust_scaling, load_scaling_config, sharp_feature_columns

SOLAR_ROOT = Path(__file__).resolve().parent
RUN_DIR = SOLAR_ROOT / "results" / "real_full_20100501_20250724"
CKPT_DIR = SOLAR_ROOT / "checkpoints" / "real_full_20100501_20250724"
OUT_DIR = SOLAR_ROOT / "results" / "prediction_exports"

TARGET_COL = "label_primary"

BASELINE_MODELS = ["climatology", "persistence"]
CLASSICAL_MODELS = [
    "logistic_regression",
    "random_forest_classifier",
    "gradient_boosting_classifier",
    "knn_classifier",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((RUN_DIR / "feature_manifest.json").read_text(encoding="utf-8"))
    feature_manifest = manifest["feature_manifest"]

    dataset = pd.read_csv(RUN_DIR / "dataset_with_split.csv")
    dataset = dataset.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train = dataset.loc[dataset["split"] == "train"]
    test = dataset.loc[dataset["split"] == "test"].reset_index(drop=True)
    print(f"train rows: {len(train)}, test rows: {len(test)}")

    # --- Recompute the exact same SHARP robust-scaling params + imputer
    # modeling/train.py::run_task fit on TRAIN only (deterministic; no seed
    # dependence in either fit_robust_scaling or SimpleImputer(median)).
    X_train_raw = train[feature_manifest]
    scaler_params = fit_robust_scaling(X_train_raw, sharp_feature_columns(feature_manifest), load_scaling_config())
    X_train_scaled = apply_robust_scaling(X_train_raw, scaler_params, copy=False)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train_scaled)

    X_test_scaled = apply_robust_scaling(test[feature_manifest], scaler_params, copy=False)
    X_test_imputed = imputer.transform(X_test_scaled)

    recorded_metrics = json.loads((RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
    recorded_models = recorded_metrics["targets"][TARGET_COL]["models"]

    y_test = test[TARGET_COL].to_numpy()
    out = pd.DataFrame({"time_utc": test["time_utc"], "true_label": y_test.astype(int)})
    thresholds: dict[str, float] = {}

    all_names = BASELINE_MODELS + CLASSICAL_MODELS
    for name in all_names:
        model = joblib.load(CKPT_DIR / f"{name}_{TARGET_COL}.joblib")
        if name in BASELINE_MODELS:
            prob = model.predict_proba(test)[:, 1]
        else:
            prob = model.predict_proba(X_test_imputed)[:, 1]

        threshold = recorded_models[name]["decision_threshold"]
        thresholds[name] = threshold
        pred = (prob >= threshold).astype(int)
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        tn = int(((pred == 0) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())

        rec = recorded_models[name]["metrics"]["test"]
        match = (tp == rec["tp"]) and (fp == rec["fp"]) and (tn == rec["tn"]) and (fn == rec["fn"])
        status = "OK" if match else "MISMATCH"
        print(
            f"{name:32s} thr={threshold:.3f} tp={tp} fp={fp} tn={tn} fn={fn}  "
            f"(recorded tp={rec['tp']} fp={rec['fp']} tn={rec['tn']} fn={rec['fn']})  [{status}]"
        )
        if not match:
            raise SystemExit(f"Reconstructed confusion matrix for {name!r} does not match metrics.json -- refusing to export.")

        out[f"{name}_prob"] = prob

    out_path = OUT_DIR / "baseline_predictions_test.csv"
    out.to_csv(out_path, index=False)
    (OUT_DIR / "baseline_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path} ({len(out)} rows, {len(all_names)} model columns) and baseline_thresholds.json")


if __name__ == "__main__":
    main()
