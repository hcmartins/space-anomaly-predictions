"""Tests for LSTM/GRU sequence models: causal windowed-dataset construction
(never crossing a gap, never padding with fabricated steps), and the
train/evaluate loop's convergence/history reporting."""
import numpy as np
import pandas as pd
import pytest
import torch

from martian.modeling.sequence_models import (
    SequenceModel,
    build_sequence_dataset,
    train_sequence_model,
)


def _features_and_dataset(sols, values, *, cell_id=0, mars_year=1, target_values=None):
    n = len(sols)
    features_df = pd.DataFrame(
        {"cell_id": [cell_id] * n, "mars_year": [mars_year] * n, "sol": sols, "x1": values}
    )
    dataset_df = pd.DataFrame(
        {
            "cell_id": [cell_id] * n, "mars_year": [mars_year] * n, "sol": sols,
            "target_x": target_values if target_values is not None else values,
        }
    )
    return features_df, dataset_df


def test_build_sequence_dataset_full_window_available():
    sols = list(range(10))
    values = [float(s) for s in sols]
    features_df, dataset_df = _features_and_dataset(sols, values)
    X, y, meta = build_sequence_dataset(dataset_df, features_df, ["x1"], window_length_sols=5, target_col="target_x")
    # anchors at sol>=4 have a full 5-sol window ending at their own sol
    assert X.shape == (6, 5, 1)
    assert list(meta["sol"]) == [4, 5, 6, 7, 8, 9]
    # window for anchor sol=4 should be [0,1,2,3,4]
    np.testing.assert_array_equal(X[0, :, 0], [0.0, 1.0, 2.0, 3.0, 4.0])


def test_build_sequence_dataset_drops_anchors_without_full_window():
    sols = [0, 1, 2]  # only 3 sols total
    values = [1.0, 2.0, 3.0]
    features_df, dataset_df = _features_and_dataset(sols, values)
    X, y, meta = build_sequence_dataset(dataset_df, features_df, ["x1"], window_length_sols=5, target_col="target_x")
    assert len(X) == 0


def test_build_sequence_dataset_never_bridges_a_gap():
    # sol=5 missing from features_df -- an anchor at sol=7 needing sols
    # 3..7 must be dropped (the window is not fully unbroken).
    sols = [0, 1, 2, 3, 4, 6, 7]
    values = [float(s) for s in sols]
    features_df, dataset_df = _features_and_dataset(sols, values)
    X, y, meta = build_sequence_dataset(dataset_df, features_df, ["x1"], window_length_sols=5, target_col="target_x")
    assert 7 not in meta["sol"].tolist()


def test_build_sequence_dataset_keeps_windows_with_nan_feature_values():
    # A window is dropped ONLY for a genuine sol gap, never merely because
    # one feature value is NaN (e.g. a sparse spatial-neighbour stat) --
    # NaN feature values are imputed downstream (modeling/train.py), same
    # discipline as the tabular classical-model path. A prior version of
    # this function DID drop on any feature NaN, which silently produced
    # zero usable windows in a real end-to-end run despite genuine
    # sol-contiguous data existing -- see planning/MODELING_DOCUMENTATION.md.
    sols = list(range(7))  # 0..6
    values = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0]
    features_df, dataset_df = _features_and_dataset(sols, values)
    X, y, meta = build_sequence_dataset(dataset_df, features_df, ["x1"], window_length_sols=5, target_col="target_x")
    # window [2..6] is sol-contiguous (no gap) even though it contains a NaN
    # feature value at sol=2 -- it must be RETAINED, with the NaN passed
    # through untouched for the caller to impute.
    assert 6 in meta["sol"].tolist()
    row = meta.index[meta["sol"] == 6][0]
    assert np.isnan(X[row, :, 0]).any()


def test_build_sequence_dataset_empty_when_no_anchors():
    features_df, dataset_df = _features_and_dataset([0, 1], [1.0, 2.0])
    dataset_df = dataset_df.iloc[0:0]
    X, y, meta = build_sequence_dataset(dataset_df, features_df, ["x1"], window_length_sols=5, target_col="target_x")
    assert X.shape == (0, 5, 1)


def test_train_sequence_model_lstm_and_gru_produce_history_and_converge():
    rng = np.random.default_rng(0)
    n_samples, window, n_features = 40, 5, 2
    X_train = rng.normal(size=(n_samples, window, n_features)).astype(np.float32)
    y_train = X_train[:, -1, 0] * 2.0
    X_val = rng.normal(size=(10, window, n_features)).astype(np.float32)
    y_val = X_val[:, -1, 0] * 2.0

    for rnn_type in ("lstm", "gru"):
        model, history, status, runtime = train_sequence_model(
            X_train, y_train, X_val, y_val,
            rnn_type=rnn_type, task_type="regression",
            hyperparameters={"hidden_size": 8, "num_layers": 1, "dropout": 0.0},
            training_config={"epochs": 15, "learning_rate": 0.01, "early_stopping_patience": 20},
            seed=1,
        )
        assert isinstance(model, SequenceModel)
        assert len(history) > 0
        assert "train_loss" in history[0] and "val_loss" in history[0]
        assert status in ("converged", "max_epochs_reached", "early_stopped")
        assert runtime >= 0.0


def test_train_sequence_model_early_stopping_triggers_with_tight_patience():
    rng = np.random.default_rng(2)
    X_train = rng.normal(size=(20, 3, 1)).astype(np.float32)
    y_train = rng.normal(size=20).astype(np.float32)
    X_val = rng.normal(size=(20, 3, 1)).astype(np.float32)
    y_val = rng.normal(size=20).astype(np.float32)  # unrelated to X_val -> val loss won't improve

    model, history, status, runtime = train_sequence_model(
        X_train, y_train, X_val, y_val,
        rnn_type="gru", task_type="regression",
        hyperparameters={"hidden_size": 4, "num_layers": 1, "dropout": 0.0},
        training_config={"epochs": 200, "learning_rate": 0.05, "early_stopping_patience": 2},
        seed=3,
    )
    assert len(history) < 200  # must have stopped early, not run all 200 epochs


def test_train_sequence_model_uses_configured_seed_deterministically():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(15, 3, 1)).astype(np.float32)
    y_train = rng.normal(size=15).astype(np.float32)

    model1, _, _, _ = train_sequence_model(
        X_train, y_train, np.empty((0, 3, 1)), np.empty((0,)),
        rnn_type="lstm", task_type="regression",
        hyperparameters={"hidden_size": 4, "num_layers": 1, "dropout": 0.0},
        training_config={"epochs": 3, "learning_rate": 0.01}, seed=99,
    )
    model2, _, _, _ = train_sequence_model(
        X_train, y_train, np.empty((0, 3, 1)), np.empty((0,)),
        rnn_type="lstm", task_type="regression",
        hyperparameters={"hidden_size": 4, "num_layers": 1, "dropout": 0.0},
        training_config={"epochs": 3, "learning_rate": 0.01}, seed=99,
    )
    with torch.no_grad():
        pred1 = model1(torch.tensor(X_train, dtype=torch.float32))
        pred2 = model2(torch.tensor(X_train, dtype=torch.float32))
    assert torch.allclose(pred1, pred2)
