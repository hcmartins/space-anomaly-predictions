"""Tests for the baseline/classical-model fit-evaluate driver."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeling.train import load_task_config, run_task

# A self-contained synthetic config, deliberately NOT read from the real
# configs/splits.yaml -- that file's real date boundaries are a live
# research choice (see planning/SPLITTING_DOCUMENTATION.md) and have
# already changed once as real data coverage was validated; tests must not
# be coupled to whatever its current values happen to be. _synthetic_dataset
# below places its train/val/test blocks to match these boundaries exactly.
_TEST_SPLITS_CONFIG = {
    "train_start": "2010-05-01", "train_end": "2015-12-31",
    "val_start": "2016-01-01", "val_end": "2017-12-31",
    "test_start": "2018-01-01", "test_end": "2019-12-31",
    "embargo_hours": 24,
}


def _synthetic_dataset(n_per_split=10, *, single_class_train=False):
    rng = np.random.default_rng(0)

    def _rows(start_date, n):
        times = pd.date_range(start_date, periods=n, freq="D", tz="UTC")
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = (x1 > 0.5).astype(float)
        return pd.DataFrame(
            {
                "time_utc": times,
                "feat_x1": x1,
                "feat_x2": x2,
                "label_primary": np.zeros(n) if single_class_train else y,
                "label_horizon_hours": 1.0,
            }
        )

    train = _rows("2012-01-01", n_per_split)
    val = _rows("2016-06-01", n_per_split)
    test = _rows("2018-06-01", n_per_split)
    return pd.concat([train, val, test], ignore_index=True)


def _minimal_task_config(**overrides):
    config = {
        "task_name": "test_task",
        "task_type": "classification",
        "target_columns": ["label_primary"],
        "seed": 1,
        "decision_threshold": 0.5,
        "baselines": [
            {"name": "climatology", "hyperparameters": {"mode": "train_mean"}},
            {"name": "persistence", "hyperparameters": {"source_column": "feat_x1"}},
        ],
        "classical_models": [
            {"name": "logistic_regression", "hyperparameters": {"max_iter": 200}},
        ],
        "paths": {
            "dataset": "results/test_task/dataset.csv",
            "feature_manifest": "results/test_task/feature_manifest.json",
            "results": "results/test_task/metrics.json",
            "checkpoint_dir": "checkpoints/test_task",
        },
    }
    config.update(overrides)
    return config


def test_load_task_config_reads_the_real_flare_forecast_config():
    config = load_task_config("flare_forecast")
    assert config["task_name"] == "flare_forecast"
    assert "label_primary" in config["target_columns"]
    assert {"climatology", "persistence"} <= {b["name"] for b in config["baselines"]}


def test_run_task_trains_baselines_and_classical_models(tmp_path):
    dataset = _synthetic_dataset()
    task_config = _minimal_task_config()
    result = run_task(
        dataset, ["feat_x1", "feat_x2"], task_config, splits_config=_TEST_SPLITS_CONFIG, project_root=tmp_path
    )

    target_result = result["targets"]["label_primary"]
    assert target_result["status"] == "trained"
    assert target_result["n_train"] == 10 and target_result["n_test"] == 10

    for name in ("climatology", "persistence", "logistic_regression"):
        model_result = target_result["models"][name]
        for split in ("train", "val", "test"):
            assert "tss" in model_result["metrics"][split]
        assert model_result["checkpoint"] is not None
        assert Path(model_result["checkpoint"]).exists()
        # tau* selection (Chapter 3.6.2/3.8): every model reports the
        # threshold actually used and where it came from.
        assert 0.0 <= model_result["decision_threshold"] <= 1.0
        assert model_result["decision_threshold_source"]

    # Coefficient interpretability (Chapter 3.8) is reported for the linear
    # classical model; baselines aren't classical models so carry no such key.
    lr_result = target_result["models"]["logistic_regression"]
    assert lr_result["interpretability"]["type"] == "coefficients"
    assert {r["feature"] for r in lr_result["interpretability"]["ranked_features"]} == {"feat_x1", "feat_x2"}

    # Results/dataset/feature-manifest artifacts were actually written.
    assert (tmp_path / "results/test_task/metrics.json").exists()
    assert (tmp_path / "results/test_task/dataset.csv").exists()
    manifest = json.loads((tmp_path / "results/test_task/feature_manifest.json").read_text())
    assert manifest["feature_manifest"] == ["feat_x1", "feat_x2"]


def test_run_task_imputer_fit_sample_size_still_trains_successfully(tmp_path):
    # Real fix for a real ArrayMemoryError (SimpleImputer's median fit on
    # the per-HARP pipeline's ~1.97M-row train split) -- see
    # modeling/train.py's run_task docstring. Sample size larger than the
    # synthetic train split here, so this also covers the "sample size
    # exceeds available rows" no-op path.
    dataset = _synthetic_dataset()
    task_config = _minimal_task_config()
    result = run_task(
        dataset, ["feat_x1", "feat_x2"], task_config, splits_config=_TEST_SPLITS_CONFIG,
        project_root=tmp_path, imputer_fit_sample_size=5,
    )
    target_result = result["targets"]["label_primary"]
    assert target_result["status"] == "trained"
    assert "logistic_regression" in target_result["models"]


def test_run_task_scales_sharp_features_and_fixes_degenerate_logistic_regression(tmp_path):
    """Reproduces the real, repeated bug (LR: TSS=0, ROC-AUC=0.5 exactly,
    all-zero coefficients) with a synthetic SHARP-scale mismatch --
    USFLUX-like values ~1e21-1e23 alongside R_VALUE-like values 0-5, same
    magnitude gap seen in every real run so far -- and shows
    modeling/train.py's SHARP robust-scaling wiring (preprocessing/
    scaling.py) resolves it: the classical model actually separates the two
    classes on held-out data instead of failing to converge."""
    rng = np.random.default_rng(0)

    def _rows(start_date, n):
        times = pd.date_range(start_date, periods=n, freq="D", tz="UTC")
        usflux = rng.uniform(1e21, 1e23, n)
        r_value = rng.uniform(0, 5, n)
        y = (usflux > 5e22).astype(float)
        return pd.DataFrame(
            {
                "time_utc": times,
                "feat_USFLUX_lag1": usflux,
                "feat_R_VALUE_lag1": r_value,
                "label_primary": y,
                "label_horizon_hours": 1.0,
            }
        )

    train = _rows("2012-01-01", 200)
    val = _rows("2016-06-01", 60)
    test = _rows("2018-06-01", 60)
    dataset = pd.concat([train, val, test], ignore_index=True)

    task_config = _minimal_task_config(
        baselines=[
            {"name": "climatology", "hyperparameters": {"mode": "train_mean"}},
            {"name": "persistence", "hyperparameters": {"source_column": "feat_USFLUX_lag1"}},
        ]
    )
    result = run_task(
        dataset,
        ["feat_USFLUX_lag1", "feat_R_VALUE_lag1"],
        task_config,
        splits_config=_TEST_SPLITS_CONFIG,
        project_root=tmp_path,
    )
    target_result = result["targets"]["label_primary"]
    # Both feature columns are named after real SHARP physical keywords, so
    # both are recognized and scaled.
    assert target_result["sharp_scaling"]["n_columns_scaled"] == 2

    lr_result = target_result["models"]["logistic_regression"]
    assert lr_result["metrics"]["test"]["roc_auc"] > 0.9
    assert lr_result["metrics"]["test"]["tss"] > 0.5
    assert any(abs(r["coefficient"]) > 0 for r in lr_result["interpretability"]["ranked_features"])


def test_run_task_checkpoints_are_real_loadable_models(tmp_path):
    import joblib

    dataset = _synthetic_dataset()
    task_config = _minimal_task_config()
    result = run_task(
        dataset, ["feat_x1", "feat_x2"], task_config, splits_config=_TEST_SPLITS_CONFIG, project_root=tmp_path
    )
    checkpoint = result["targets"]["label_primary"]["models"]["logistic_regression"]["checkpoint"]
    model = joblib.load(checkpoint)
    assert hasattr(model, "predict_proba")


def test_run_task_single_class_train_skips_classical_models_but_keeps_baselines(tmp_path):
    dataset = _synthetic_dataset(single_class_train=True)
    task_config = _minimal_task_config()
    result = run_task(
        dataset, ["feat_x1", "feat_x2"], task_config, splits_config=_TEST_SPLITS_CONFIG, project_root=tmp_path
    )

    target_result = result["targets"]["label_primary"]
    assert target_result["n_classes_in_train"] == 1
    assert "classical_models_skipped_reason" in target_result
    assert "logistic_regression" not in target_result["models"]
    assert "climatology" in target_result["models"]  # baselines still ran


def test_run_task_insufficient_test_data_reports_status(tmp_path):
    dataset = _synthetic_dataset()
    dataset = dataset.loc[~((dataset["time_utc"] >= "2018-01-01") & (dataset["time_utc"] < "2020-01-01"))]
    task_config = _minimal_task_config()
    result = run_task(
        dataset, ["feat_x1", "feat_x2"], task_config, splits_config=_TEST_SPLITS_CONFIG, project_root=tmp_path
    )
    assert result["targets"]["label_primary"]["status"] == "insufficient_data"


def test_run_task_undefined_labels_are_dropped_not_trained_against(tmp_path):
    dataset = _synthetic_dataset()
    dataset.loc[dataset["time_utc"] < "2012-01-05", "label_primary"] = np.nan
    task_config = _minimal_task_config()
    result = run_task(
        dataset, ["feat_x1", "feat_x2"], task_config, splits_config=_TEST_SPLITS_CONFIG, project_root=tmp_path
    )
    # 4 of the 10 train-split rows (2012-01-01..04) have an undefined label.
    assert result["targets"]["label_primary"]["n_train"] == 6
