"""Integration test for Algorithm 5.1's Stage 2 training loop, using the
REAL frozen encoders on small synthetic batches (training on real domain
data is out of scope for this delivery -- see README.md)."""
import numpy as np
import torch

from encoders.martian_encoder import load_martian_encoder
from encoders.solar_encoder import load_encoders_config, load_solar_encoder
from fusion.model import CrossDomainFusionModel
from training.train_fusion import train_fusion

D_S = load_encoders_config()["solar"]["input_size"]   # GOES-only (2) or GOES+SHARP (20), config-driven
D_M = load_encoders_config()["martian"]["input_size"]   # real storm-detection feature width, config-driven


def _synthetic_solar(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 30, D_S)).astype(np.float32)
    y = (rng.uniform(size=n) < 0.3).astype(np.float32)
    return X, y


def _synthetic_martian(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3, D_M)).astype(np.float32)
    y = (rng.uniform(size=n) < 0.3).astype(np.float32)
    return X, y


def _real_model():
    solar_encoder = load_solar_encoder()
    martian_encoder, _provenance = load_martian_encoder()
    return CrossDomainFusionModel(
        solar_encoder=solar_encoder, martian_encoder=martian_encoder,
        shared_dim=8, attention_hidden_size=4,
    )


def test_train_fusion_runs_end_to_end_and_produces_history():
    model = _real_model()
    X_s_train, y_s_train = _synthetic_solar(80, seed=0)
    X_m_train, y_m_train = _synthetic_martian(80, seed=1)
    X_s_val, y_s_val = _synthetic_solar(20, seed=2)
    X_m_val, y_m_val = _synthetic_martian(20, seed=3)

    training_config = {
        "batch_size_solar": 16, "batch_size_martian": 16,
        "learning_rate": 0.01, "epochs": 3, "early_stopping_patience": 3,
        "lambda_align": 0.1, "lambda_bal": 0.1, "lambda_wd": 1e-4,
    }
    model, history, status, runtime = train_fusion(
        model, X_s_train, y_s_train, X_m_train, y_m_train,
        X_s_val, y_s_val, X_m_val, y_m_val,
        training_config=training_config, seed=0,
    )
    assert len(history) == 3
    assert runtime > 0
    assert "no_validation_selection_available" not in status
    for h in history:
        assert np.isfinite(h["total"])


def test_train_fusion_updates_prototypes_not_left_at_zero():
    model = _real_model()
    X_s_train, y_s_train = _synthetic_solar(40, seed=0)
    X_m_train, y_m_train = _synthetic_martian(40, seed=1)

    assert torch.all(model.solar_prototype == 0)
    training_config = {
        "batch_size_solar": 8, "batch_size_martian": 8,
        "learning_rate": 0.01, "epochs": 2, "early_stopping_patience": 2,
        "lambda_align": 0.1, "lambda_bal": 0.1, "lambda_wd": 1e-4,
    }
    model, _history, _status, _runtime = train_fusion(
        model, X_s_train, y_s_train, X_m_train, y_m_train,
        np.empty((0, 30, D_S), dtype=np.float32), np.empty(0), np.empty((0, 3, D_M), dtype=np.float32), np.empty(0),
        training_config=training_config, seed=0,
    )
    assert not torch.all(model.solar_prototype == 0)
    assert not torch.all(model.martian_prototype == 0)


def test_train_fusion_handles_missing_validation_without_crashing():
    model = _real_model()
    X_s_train, y_s_train = _synthetic_solar(40, seed=0)
    X_m_train, y_m_train = _synthetic_martian(40, seed=1)
    training_config = {
        "batch_size_solar": 8, "batch_size_martian": 8,
        "learning_rate": 0.01, "epochs": 2, "early_stopping_patience": 2,
        "lambda_align": 0.1, "lambda_bal": 0.1, "lambda_wd": 1e-4,
    }
    _model, _history, status, _runtime = train_fusion(
        model, X_s_train, y_s_train, X_m_train, y_m_train,
        np.empty((0, 30, D_S), dtype=np.float32), np.empty(0), np.empty((0, 3, D_M), dtype=np.float32), np.empty(0),
        training_config=training_config, seed=0,
    )
    assert "no_validation_selection_available" in status
