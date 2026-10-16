"""Tests for the GOES sequence LSTM (Chapter 3.6, Table 3.3)."""
import numpy as np
import torch

from modeling.sequence_models import (
    SequenceModel,
    compute_class_weights,
    load_sequence_checkpoint,
    save_sequence_checkpoint,
    train_sequence_model,
    weighted_binary_cross_entropy,
)


def test_compute_class_weights_hand_computed():
    y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8 negatives, 2 positives
    w_pos, w_neg = compute_class_weights(y_train)
    assert w_pos == 10 / (2 * 2)
    assert w_neg == 10 / (2 * 8)


def test_compute_class_weights_handles_single_class():
    y_train = np.array([0, 0, 0])
    w_pos, w_neg = compute_class_weights(y_train)
    assert w_pos == 0.0
    assert w_neg == 3 / (2 * 3)


def test_weighted_binary_cross_entropy_hand_computed():
    y_prob = torch.tensor([0.8, 0.3])
    y_true = torch.tensor([1.0, 0.0])
    w_pos, w_neg = 2.0, 0.5
    loss = weighted_binary_cross_entropy(y_prob, y_true, w_pos, w_neg)
    expected = -(w_pos * np.log(0.8) + w_neg * np.log(1 - 0.3)) / 2
    assert abs(loss.item() - expected) < 1e-6


def test_sequence_model_encode_and_forward_shapes():
    model = SequenceModel(input_size=2, hidden_size=8, num_layers=1, dropout=0.0)
    model.eval()
    x = torch.randn(5, 6, 2)  # (batch, T, d)
    h = model.encode(x)
    assert h.shape == (5, 8)
    p = model(x)
    assert p.shape == (5,)
    assert torch.all((p >= 0) & (p <= 1))


def test_sequence_model_forward_is_sigmoid_of_head_of_encode():
    model = SequenceModel(input_size=2, hidden_size=4, num_layers=1, dropout=0.0)
    model.eval()
    x = torch.randn(3, 5, 2)
    with torch.no_grad():
        h = model.encode(x)
        expected = torch.sigmoid(model.head(h)).squeeze(-1)
        actual = model(x)
    assert torch.allclose(actual, expected)


def _synthetic_dataset(n=200, T=6, d=1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, T, d)).astype(np.float32)
    # Clear, learnable signal: label = 1 iff the sequence mean is positive.
    y = (X.mean(axis=(1, 2)) > 0).astype(np.float32)
    return X, y


def test_train_sequence_model_reduces_loss_on_an_easy_synthetic_task():
    X_train, y_train = _synthetic_dataset(n=200, seed=0)
    X_val, y_val = _synthetic_dataset(n=60, seed=1)
    hyperparameters = {"hidden_size": 8, "num_layers": 1, "dropout": 0.0}
    training_config = {"epochs": 15, "learning_rate": 0.01, "early_stopping_patience": 15, "batch_size": 32}

    model, history, status, runtime = train_sequence_model(
        X_train, y_train, X_val, y_val, hyperparameters=hyperparameters, training_config=training_config, seed=0
    )
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert runtime > 0
    assert "no_validation_selection_available" not in status

    model.eval()
    with torch.no_grad():
        val_prob = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
    from modeling.metrics import classification_metrics

    final_tss = classification_metrics(y_val, val_prob)["tss"]
    assert final_tss > 0.3   # real signal, should clearly beat random (TSS=0)


def test_train_sequence_model_handles_single_class_validation_without_crashing():
    X_train, y_train = _synthetic_dataset(n=50, seed=0)
    X_val = np.zeros((10, 6, 1), dtype=np.float32)
    y_val = np.zeros(10)   # single class -- no valid TSS to select on
    hyperparameters = {"hidden_size": 4, "num_layers": 1, "dropout": 0.0}
    training_config = {"epochs": 3, "learning_rate": 0.01, "early_stopping_patience": 3, "batch_size": 16}

    model, history, status, runtime = train_sequence_model(
        X_train, y_train, X_val, y_val, hyperparameters=hyperparameters, training_config=training_config, seed=0
    )
    assert "no_validation_selection_available" in status
    assert len(history) == 3


def test_save_and_load_sequence_checkpoint_roundtrip(tmp_path):
    model = SequenceModel(input_size=2, hidden_size=4, num_layers=1, dropout=0.0)
    model.eval()
    checkpoint_path = tmp_path / "lstm.pt"
    save_sequence_checkpoint(model, {"hidden_size": 4, "num_layers": 1, "dropout": 0.0}, checkpoint_path)

    loaded = load_sequence_checkpoint(checkpoint_path)
    x = torch.randn(3, 5, 2)
    with torch.no_grad():
        original_pred = model(x)
        loaded_pred = loaded(x)
    assert torch.allclose(original_pred, loaded_pred)
    assert loaded.encode(x).shape == (3, 4)
