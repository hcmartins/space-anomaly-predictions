"""Smoke tests for the generic per-task train/evaluate/checkpoint/figure
driver: end-to-end regression and classification runs through the FULL
model hierarchy (baselines -> classical -> sequence) on small synthetic
datasets, per-task seed usage, provenance-record contents, and graceful
handling of insufficient data."""
import json

from split_fixtures import mechanics_splits

import joblib
import numpy as np
import pandas as pd
import pytest

from martian.modeling.train import _impute_sequence_train_only, load_task_config, run_task


def _synthetic_regression_dataset(seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    n_train, n_test = 200, 60
    rows = []
    for i in range(n_train):
        rows.append({"cell_id": i % 20, "mars_year": 36, "sol": (i % 500) + 1, "solar_longitude_ls": float(i % 360)})
    for i in range(n_test):
        rows.append({"cell_id": i % 20, "mars_year": 37, "sol": (i % 500) + 1, "solar_longitude_ls": float(i % 360)})
    df = pd.DataFrame(rows)
    n = len(df)
    df["feat_x1"] = rng.normal(size=n)
    df["feat_x2"] = rng.normal(size=n)
    df["dust_optical_depth"] = 0.2 + 0.05 * df["feat_x1"]
    df["dust_optical_depth_mu"] = 0.2
    df["target_dust_optical_depth_h1"] = df["feat_x1"] * 2.0 + rng.normal(scale=0.1, size=n)
    return df, ["feat_x1", "feat_x2"]


def _synthetic_classification_dataset(seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    n_train, n_test = 200, 60
    rows = []
    for i in range(n_train):
        rows.append({"cell_id": i % 20, "mars_year": 36, "sol": (i % 500) + 1, "solar_longitude_ls": float(i % 360)})
    for i in range(n_test):
        rows.append({"cell_id": i % 20, "mars_year": 37, "sol": (i % 500) + 1, "solar_longitude_ls": float(i % 360)})
    df = pd.DataFrame(rows)
    n = len(df)
    df["feat_x1"] = rng.normal(size=n)
    df["atmospheric_anomaly"] = (df["feat_x1"] > 0.5).astype(int)
    df["target_storm_horizon_h1"] = (df["feat_x1"] + rng.normal(scale=0.3, size=n) > 0.5).astype(int)
    return df, ["feat_x1"]


def _small_paths(tmp_path) -> dict:
    return {
        "dataset": str(tmp_path / "dataset.csv"),
        "feature_manifest": str(tmp_path / "feature_manifest.json"),
        "results": str(tmp_path / "metrics.json"),
        "predictions": str(tmp_path / "predictions.csv"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "figures_dir": str(tmp_path / "figures"),
        "sequence_feasibility_report": str(tmp_path / "sequence_feasibility_report.csv"),
    }


@pytest.fixture
def dust_task_config(tmp_path):
    config = load_task_config("dust_regression")
    config["horizons_sols"] = [1]
    config["classical_models"] = [{"name": "random_forest_regressor", "hyperparameters": {"n_estimators": 10, "max_depth": 3}}]
    config["sequence"]["enabled"] = False
    config["paths"] = _small_paths(tmp_path)
    return config


@pytest.fixture
def storm_task_config(tmp_path):
    config = load_task_config("storm_detection")
    config["horizons_sols"] = [1]
    config["classical_models"] = [
        {"name": "random_forest_classifier", "hyperparameters": {"n_estimators": 10, "max_depth": 3, "class_weight": "balanced"}},
        {"name": "logistic_regression", "hyperparameters": {"max_iter": 200}},
    ]
    config["sequence"]["enabled"] = False
    config["paths"] = _small_paths(tmp_path)
    return config


def test_run_task_regression_end_to_end(tmp_path, dust_task_config):
    dataset_df, feature_manifest = _synthetic_regression_dataset()
    result = run_task("dust_regression", dataset_df, feature_manifest, dust_task_config, splits_config=mechanics_splits(), project_root=tmp_path.parent)

    horizon_result = result["horizons"]["1"]
    assert horizon_result["status"] == "trained"
    assert horizon_result["n_test"] > 0

    models = horizon_result["models"]
    assert set(models.keys()) >= {"climatology", "persistence", "seasonal_mean", "random_forest_regressor"}
    assert models["climatology"]["kind"] == "baseline"
    assert models["random_forest_regressor"]["kind"] == "classical"
    assert models["random_forest_regressor"]["metrics"]["mae"] < 1.0

    assert (tmp_path / "checkpoints" / "random_forest_regressor_h1.joblib").exists()
    assert (tmp_path / "checkpoints" / "climatology_h1.joblib").exists()
    assert (tmp_path / "figures" / "random_forest_regressor_h1_obs_vs_pred.png").exists()
    assert (tmp_path / "figures" / "random_forest_regressor_h1_residuals.png").exists()
    assert (tmp_path / "feature_manifest.json").exists()
    manifest = json.loads((tmp_path / "feature_manifest.json").read_text(encoding="utf-8"))
    assert manifest["feature_manifest"] == feature_manifest

    # baselines are also fully evaluated with figures (regression: obs-vs-pred
    # + residuals), not just a single "primary model" figure.
    assert (tmp_path / "figures" / "persistence_h1_obs_vs_pred.png").exists()

    # every model's evaluation record carries per-split metrics + CIs
    eval_result = models["random_forest_regressor"]["evaluation"]
    assert set(eval_result["splits"].keys()) >= {"train", "test"}
    assert "mae" in eval_result["splits"]["test"]["ci"]
    assert "baseline_comparison" in eval_result  # classical model vs. its appropriate baseline


def test_run_task_classification_end_to_end(tmp_path, storm_task_config):
    dataset_df, feature_manifest = _synthetic_classification_dataset()
    result = run_task("storm_detection", dataset_df, feature_manifest, storm_task_config, splits_config=mechanics_splits(), project_root=tmp_path.parent)

    horizon_result = result["horizons"]["1"]
    assert horizon_result["status"] == "trained"
    models = horizon_result["models"]
    m = models["random_forest_classifier"]["metrics"]
    assert "tss" in m and "roc_auc" in m and "pr_auc" in m and "brier" in m
    assert models["persistence"]["kind"] == "baseline"
    assert (tmp_path / "checkpoints" / "random_forest_classifier_h1.joblib").exists()
    assert (tmp_path / "checkpoints" / "logistic_regression_h1.joblib").exists()
    assert (tmp_path / "figures" / "random_forest_classifier_h1_confusion_matrix.png").exists()
    assert (tmp_path / "figures" / "random_forest_classifier_h1_reliability.png").exists()
    assert (tmp_path / "figures" / "random_forest_classifier_h1_pr_curve.png").exists()
    assert (tmp_path / "figures" / "random_forest_classifier_h1_roc_curve.png").exists()

    eval_result = models["random_forest_classifier"]["evaluation"]
    assert "tss" in eval_result["splits"]["test"]["ci"]
    assert "baseline_comparison" in eval_result


def test_run_task_uses_task_specific_seed(tmp_path, dust_task_config):
    dataset_df, feature_manifest = _synthetic_regression_dataset()
    dust_task_config["seed"] = 12345
    run_task("dust_regression", dataset_df, feature_manifest, dust_task_config, splits_config=mechanics_splits(), project_root=tmp_path.parent)
    model = joblib.load(tmp_path / "checkpoints" / "random_forest_regressor_h1.joblib")
    assert model.random_state == 12345


def test_run_task_insufficient_data_status_no_crash(tmp_path, dust_task_config):
    # Only MY1 (train) rows -- no MY2 (test) rows at all.
    dataset_df, feature_manifest = _synthetic_regression_dataset()
    dataset_df = dataset_df.loc[dataset_df["mars_year"] == 1].reset_index(drop=True)
    result = run_task("dust_regression", dataset_df, feature_manifest, dust_task_config, splits_config=mechanics_splits(), project_root=tmp_path.parent)
    assert result["horizons"]["1"]["status"] == "insufficient_data"
    assert not (tmp_path / "checkpoints" / "random_forest_regressor_h1.joblib").exists()


def test_own_dataset_and_feature_manifest_files_written(tmp_path, dust_task_config):
    dataset_df, feature_manifest = _synthetic_regression_dataset()
    run_task("dust_regression", dataset_df, feature_manifest, dust_task_config, splits_config=mechanics_splits(), project_root=tmp_path.parent)
    assert (tmp_path / "dataset.csv").exists()
    saved = pd.read_csv(tmp_path / "dataset.csv")
    assert len(saved) == len(dataset_df)


def test_provenance_record_contains_required_fields(tmp_path, dust_task_config):
    dataset_df, feature_manifest = _synthetic_regression_dataset()
    run_task("dust_regression", dataset_df, feature_manifest, dust_task_config, splits_config=mechanics_splits(), project_root=tmp_path.parent)

    record_path = tmp_path / "random_forest_regressor_h1_provenance.json"
    assert record_path.exists()
    record = json.loads(record_path.read_text(encoding="utf-8"))

    for key in (
        "exact_parameters", "random_seed", "software_versions", "training_history",
        "checkpoint_path", "feature_order", "n_train", "n_val", "convergence_status",
        "runtime_seconds", "hardware_info",
    ):
        assert key in record, f"missing provenance field {key!r}"
    assert record["random_seed"] == dust_task_config["seed"]
    assert record["feature_order"] == feature_manifest
    assert record["software_versions"]["python"]
    assert record["hardware_info"]["cpu_count"] is not None


def test_baseline_provenance_has_empty_feature_order(tmp_path, dust_task_config):
    dataset_df, feature_manifest = _synthetic_regression_dataset()
    run_task("dust_regression", dataset_df, feature_manifest, dust_task_config, splits_config=mechanics_splits(), project_root=tmp_path.parent)
    record = json.loads((tmp_path / "persistence_h1_provenance.json").read_text(encoding="utf-8"))
    assert record["model_kind"] == "baseline"
    assert record["feature_order"] == []
    assert record["convergence_status"] == "n/a (non-iterative baseline)"


def test_impute_sequence_train_only_applies_train_stats_to_all_other_arrays():
    # 2026-08-13: _impute_sequence_train_only now takes an arbitrary number
    # of extra arrays (train, val, test) rather than exactly one -- confirm
    # imputation statistics are fit on train alone and applied identically
    # to every other array passed in.
    X_train = np.array([[[1.0], [3.0]]])  # median = 2.0
    X_val = np.array([[[np.nan], [5.0]]])
    X_test = np.array([[[np.nan], [7.0]]])

    train_out, val_out, test_out = _impute_sequence_train_only(X_train, X_val, X_test)

    assert train_out[0, 0, 0] == 1.0  # train itself has no NaN to fill
    assert val_out[0, 0, 0] == 2.0  # filled with TRAIN's median, not val's own
    assert test_out[0, 0, 0] == 2.0
    assert val_out[0, 1, 0] == 5.0  # non-NaN values pass through unchanged
    assert test_out[0, 1, 0] == 7.0


def test_run_task_sequence_models_feasible_window(tmp_path):
    # Dense, fully-consecutive per-cell sol coverage so small sequence
    # windows are genuinely feasible -- proves the sequence branch of
    # run_task works end-to-end (feasibility report -> windowed dataset ->
    # LSTM/GRU training -> checkpoint + provenance), not just the tabular path.
    # Train/val both come from mars_year 36 (train sols 1-5, val sols 6-8) so
    # early stopping genuinely has its own held-out data, distinct from the
    # mars_year-37 test split -- see test_run_task_sequence_models_uses_val_not_test_for_early_stopping
    # below for the dedicated regression test against the 2026-08-13 leakage fix.
    rng = np.random.default_rng(0)
    n_sols = 8  # long enough for window=3 to be feasible, too short for window=10
    rows = []
    for my in (36, 37):
        for s in range(1, n_sols + 1):
            rows.append({"cell_id": 0, "mars_year": my, "sol": s, "solar_longitude_ls": float(s % 360)})
    df = pd.DataFrame(rows)
    n = len(df)
    df["feat_x1"] = rng.normal(size=n)
    df["dust_optical_depth"] = 0.2 + 0.05 * df["feat_x1"]
    df["dust_optical_depth_mu"] = 0.2
    df["target_dust_optical_depth_h1"] = df["feat_x1"] * 2.0 + rng.normal(scale=0.05, size=n)
    feature_manifest = ["feat_x1"]
    features_df = df[["cell_id", "mars_year", "sol", "feat_x1"]].copy()

    config = load_task_config("dust_regression")
    config["horizons_sols"] = [1]
    config["baselines"] = []
    config["classical_models"] = []
    config["sequence"]["enabled"] = True
    config["sequence"]["window_length_primary"] = 3
    config["sequence"]["window_length_sensitivity"] = [3, 10]  # 10 must be infeasible (only 8 consecutive sols)
    config["sequence"]["training"] = {"epochs": 5, "learning_rate": 0.01, "early_stopping_patience": 10}
    config["paths"] = _small_paths(tmp_path)
    splits_config = {
        "train_mars_year": 36, "train_sol_range": [1, 5],
        "val_mars_year": 36, "val_sol_range": [6, 8],
        "test_mars_year": 37, "test_sol_range": [1, 8],
    }

    result = run_task(
        "dust_regression", df, feature_manifest, config, features_df=features_df,
        splits_config=splits_config, project_root=tmp_path.parent,
    )

    assert (tmp_path / "sequence_feasibility_report.csv").exists()
    report = pd.read_csv(tmp_path / "sequence_feasibility_report.csv")
    assert set(report["window_length_sols"]) == {3, 10}

    horizon_result = result["horizons"]["1"]
    assert "sequence" in horizon_result
    seq_w3 = horizon_result["sequence"]["3"]
    assert seq_w3["status"] == "trained"
    assert "lstm" in seq_w3["models"] and "gru" in seq_w3["models"]
    lstm_summary = seq_w3["models"]["lstm"]
    assert lstm_summary["n_train_windows"] > 0
    assert lstm_summary["n_val_windows"] > 0  # genuine val windows, not a test-split substitute
    assert lstm_summary["n_test_windows"] > 0
    assert horizon_result["sequence"]["10"]["status"] == "infeasible"
    assert (tmp_path / "checkpoints" / "lstm_w3_h1.pt").exists()
    assert (tmp_path / "lstm_w3_h1_provenance.json").exists()
    lstm_record = json.loads((tmp_path / "lstm_w3_h1_provenance.json").read_text(encoding="utf-8"))
    assert lstm_record["model_kind"] == "sequence"
    assert lstm_record["exact_parameters"]["window_length_sols"] == 3
    assert lstm_record["n_val"] == lstm_summary["n_val_windows"]  # provenance n_val is the VAL count, not test's
    assert len(lstm_record["training_history"]) > 0

    # The evaluation record must carry all three splits, val included, so a
    # reader can see the val metrics early stopping was actually driven by.
    eval_result = lstm_summary["evaluation"]
    assert set(eval_result["splits"].keys()) >= {"train", "val", "test"}


def test_run_task_sequence_models_infeasible_when_val_empty(tmp_path):
    """Regression test for the 2026-08-13 leakage fix: if the configured val
    window contains no data, sequence training must be reported INFEASIBLE,
    never silently trained by substituting the test split as if it were
    val (the original bug -- CHAPTER_4_PARAMETER_VALUES.md caveat 9)."""
    rng = np.random.default_rng(0)
    n_sols = 8
    rows = []
    for my in (36, 37):
        for s in range(1, n_sols + 1):
            rows.append({"cell_id": 0, "mars_year": my, "sol": s, "solar_longitude_ls": float(s % 360)})
    df = pd.DataFrame(rows)
    n = len(df)
    df["feat_x1"] = rng.normal(size=n)
    df["dust_optical_depth"] = 0.2 + 0.05 * df["feat_x1"]
    df["dust_optical_depth_mu"] = 0.2
    df["target_dust_optical_depth_h1"] = df["feat_x1"] * 2.0 + rng.normal(scale=0.05, size=n)
    feature_manifest = ["feat_x1"]
    features_df = df[["cell_id", "mars_year", "sol", "feat_x1"]].copy()

    config = load_task_config("dust_regression")
    config["horizons_sols"] = [1]
    config["baselines"] = []
    config["classical_models"] = []
    config["sequence"]["enabled"] = True
    config["sequence"]["window_length_primary"] = 3
    config["sequence"]["window_length_sensitivity"] = [3]
    config["sequence"]["training"] = {"epochs": 5, "learning_rate": 0.01, "early_stopping_patience": 10}
    config["paths"] = _small_paths(tmp_path)
    # val_sol_range [100, 200] overlaps NONE of the sol-1..8 data -- genuinely empty val.
    splits_config = {
        "train_mars_year": 36, "train_sol_range": [1, 8],
        "val_mars_year": 36, "val_sol_range": [100, 200],
        "test_mars_year": 37, "test_sol_range": [1, 8],
    }

    result = run_task(
        "dust_regression", df, feature_manifest, config, features_df=features_df,
        splits_config=splits_config, project_root=tmp_path.parent,
    )

    seq_w3 = result["horizons"]["1"]["sequence"]["3"]
    assert seq_w3["status"] == "infeasible"
    assert seq_w3["n_val_windows"] == 0
    assert not (tmp_path / "checkpoints" / "lstm_w3_h1.pt").exists()
