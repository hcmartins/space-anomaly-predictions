"""Tests for A2's capacity-matched unimodal model + its training loop."""
import numpy as np
import torch

from encoders.frozen_encoder import FrozenSequenceEncoder
from fusion.unimodal import CapacityMatchedUnimodalModel, _solve_head_hidden_size, domain_share_of_trainable_params
from training.train_unimodal import train_unimodal


def _tiny_encoder():
    return FrozenSequenceEncoder(input_size=3, hidden_size=8, num_layers=1)


def test_head_hidden_size_solves_the_real_formula_exactly():
    shared_dim = 16
    for target in (0, 1, 10, 100, 1000, 5000):
        hidden = _solve_head_hidden_size(shared_dim=shared_dim, target_head_param_count=target)
        actual = hidden * (shared_dim + 2) + 1
        assert actual >= target
        if hidden > 1:
            assert (hidden - 1) * (shared_dim + 2) + 1 < target  # hidden is the SMALLEST that meets the target


def test_model_trainable_param_count_meets_or_exceeds_target():
    encoder = _tiny_encoder()
    for target in (50, 500, 5000):
        model = CapacityMatchedUnimodalModel(encoder, shared_dim=16, target_param_count=target)
        assert model.trainable_parameter_count() >= target
        # Encoder itself must stay frozen/excluded.
        assert all(not p.requires_grad for p in model.encoder.parameters())


def test_forward_produces_valid_probability():
    encoder = _tiny_encoder()
    model = CapacityMatchedUnimodalModel(encoder, shared_dim=16, target_param_count=500)
    x = torch.randn(5, 4, 3)
    y_prob = model(x)
    assert y_prob.shape == (5,)
    assert torch.all((y_prob >= 0) & (y_prob <= 1))


def test_domain_share_of_trainable_params_sums_three_modules():
    projection = torch.nn.Linear(8, 16)
    scorer = torch.nn.Linear(16, 8)
    head = torch.nn.Linear(16, 1)
    expected = sum(p.numel() for p in projection.parameters()) + sum(p.numel() for p in scorer.parameters()) + sum(
        p.numel() for p in head.parameters()
    )
    assert domain_share_of_trainable_params(projection=projection, scorer=scorer, head=head) == expected


def test_train_unimodal_reduces_loss_on_separable_data():
    encoder = _tiny_encoder()
    model = CapacityMatchedUnimodalModel(encoder, shared_dim=16, target_param_count=200)

    rng = np.random.default_rng(0)
    n = 40
    X = rng.normal(size=(n, 4, 3)).astype(np.float32)
    y = (rng.uniform(size=n) < 0.5).astype(np.float32)

    training_config = {
        "learning_rate": 0.01, "batch_size_solar": 16, "epochs": 20,
        "early_stopping_patience": 20, "lambda_wd": 1e-4,
    }
    _, history, convergence_status, runtime_seconds = train_unimodal(
        model, X[:30], y[:30], X[30:], y[30:], domain="solar", training_config=training_config, seed=1,
    )
    assert len(history) > 0
    assert runtime_seconds >= 0.0
    assert history[-1]["loss"] <= history[0]["loss"] + 1e-6


def test_train_unimodal_rejects_unknown_domain():
    encoder = _tiny_encoder()
    model = CapacityMatchedUnimodalModel(encoder, shared_dim=16, target_param_count=200)
    training_config = {"learning_rate": 0.01, "batch_size_bogus": 4, "epochs": 1, "early_stopping_patience": 1, "lambda_wd": 0.0}
    try:
        train_unimodal(model, np.zeros((2, 4, 3), dtype=np.float32), np.zeros(2, dtype=np.float32),
                        np.zeros((0, 4, 3), dtype=np.float32), np.zeros(0, dtype=np.float32),
                        domain="bogus", training_config=training_config, seed=1)
        assert False, "expected ValueError"
    except ValueError:
        pass
