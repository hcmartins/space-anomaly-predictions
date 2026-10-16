"""Integration tests for the GOES sequence LSTM task orchestration
(modeling/sequence_train.py::run_sequence_task)."""
import numpy as np
import pandas as pd

from modeling.sequence_models import load_sequence_checkpoint
from modeling.sequence_train import run_sequence_task


def _sequences_config(**overrides):
    config = {
        "feature_columns": ["goes_seq_xrsb_flux_w_m2", "goes_seq_xrsa_flux_w_m2"],
        "look_back_hours": 1,   # window_steps = 5 at 12-min cadence -- keeps tests fast
    }
    config.update(overrides)
    return config


def _preprocessing_config(**overrides):
    config = {"resample_cadence_minutes": 12}
    config.update(overrides)
    return config


def _processed_goes(n_steps=100, all_flagged=False):
    times = pd.date_range("2018-01-01T00:00:00Z", periods=n_steps, freq="12min")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "time_utc": times,
            "goes_seq_xrsb_flux_w_m2": rng.normal(size=n_steps),
            "goes_seq_xrsa_flux_w_m2": rng.normal(size=n_steps),
            "goes_seq_interpolated_any": [all_flagged] * n_steps,
            "goes_seq_spike_removed_any": [False] * n_steps,
            "goes_seq_gap_minutes": [0.0] * n_steps,
        }
    )


def _labeled_dataset(times):
    n = len(times)
    split = np.where(np.arange(n) < n * 0.6, "train", np.where(np.arange(n) < n * 0.8, "val", "test"))
    label = (np.arange(n) % 2).astype(float)   # guarantees both classes in any contiguous slice >= 2 rows
    return pd.DataFrame({"time_utc": times, "split": split, "label_primary": label})


def _task_config(**overrides):
    config = {
        "task_name": "flare_forecast_sequence",
        "task_type": "classification",
        "target_column": "label_primary",
        "seed": 1,
        "decision_threshold": 0.5,
        "threshold_selection": {"metric": "tss", "grid_min": 0.01, "grid_max": 0.99, "grid_steps": 20},
        "sequence": {
            "models": [{"rnn_type": "lstm", "hyperparameters": {"hidden_size": 4, "num_layers": 1, "dropout": 0.0}}],
            "training": {"epochs": 2, "learning_rate": 0.01, "early_stopping_patience": 2, "batch_size": 16},
        },
        "paths": {
            "checkpoint_dir": "checkpoints/test_flare_forecast_sequence",
            "results": "results/test_flare_forecast_sequence/metrics.json",
            "feasibility_report": "results/test_flare_forecast_sequence/feasibility_report.json",
        },
    }
    config.update(overrides)
    return config


def test_run_sequence_task_trains_and_produces_expected_result_shape(tmp_path):
    processed_goes = _processed_goes(n_steps=100)
    issuance_times = processed_goes["time_utc"].iloc[10:90]   # plenty of history for a 5-step window
    labeled = _labeled_dataset(issuance_times.reset_index(drop=True))

    result = run_sequence_task(
        processed_goes, labeled, _task_config(),
        project_root=tmp_path, sequences_config=_sequences_config(), preprocessing_config=_preprocessing_config(),
    )

    assert result["status"] == "trained"
    assert result["feasibility_report"]["feasible"] is True
    assert result["n_train"] > 0 and result["n_val"] > 0 and result["n_test"] > 0

    lstm_result = result["sequence_models"]["lstm"]
    for split in ("train", "val", "test"):
        assert "tss" in lstm_result["metrics"][split]
    assert 0.0 <= lstm_result["decision_threshold"] <= 1.0
    assert lstm_result["decision_threshold_source"]

    checkpoint_path = tmp_path / "checkpoints/test_flare_forecast_sequence/lstm_label_primary.pt"
    assert checkpoint_path.exists()
    loaded = load_sequence_checkpoint(checkpoint_path)
    assert loaded is not None

    assert (tmp_path / "results/test_flare_forecast_sequence/metrics.json").exists()
    assert (tmp_path / "results/test_flare_forecast_sequence/feasibility_report.json").exists()


def test_run_sequence_task_reports_infeasible_without_crashing(tmp_path):
    processed_goes = _processed_goes(n_steps=100, all_flagged=True)   # every step flagged -> no clean windows
    issuance_times = processed_goes["time_utc"].iloc[10:90]
    labeled = _labeled_dataset(issuance_times.reset_index(drop=True))

    result = run_sequence_task(
        processed_goes, labeled, _task_config(),
        project_root=tmp_path, sequences_config=_sequences_config(), preprocessing_config=_preprocessing_config(),
    )
    assert result["status"] == "infeasible"
    assert result["feasibility_report"]["feasible"] is False
    assert "sequence_models" not in result


def test_run_sequence_task_reports_insufficient_data_for_a_single_class_train_split(tmp_path):
    processed_goes = _processed_goes(n_steps=100)
    issuance_times = processed_goes["time_utc"].iloc[10:90].reset_index(drop=True)
    labeled = _labeled_dataset(issuance_times)
    labeled["label_primary"] = 0.0   # single class everywhere -- train split cannot be fit

    result = run_sequence_task(
        processed_goes, labeled, _task_config(),
        project_root=tmp_path, sequences_config=_sequences_config(), preprocessing_config=_preprocessing_config(),
    )
    assert result["status"] == "insufficient_data"
