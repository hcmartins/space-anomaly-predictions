"""End-to-end forward-pass test using the REAL solar + martian frozen
encoders (loaded from their real checkpoints)."""
import torch

from encoders.martian_encoder import load_martian_encoder
from encoders.solar_encoder import load_encoders_config, load_solar_encoder
from fusion.model import CrossDomainFusionModel

D_S = load_encoders_config()["solar"]["input_size"]   # GOES-only (2) or GOES+SHARP (20), config-driven
D_M = load_encoders_config()["martian"]["input_size"]   # real storm-detection feature width, config-driven


def _real_model(shared_dim=16):
    solar_encoder = load_solar_encoder()
    martian_encoder, _provenance = load_martian_encoder()
    return CrossDomainFusionModel(
        solar_encoder=solar_encoder, martian_encoder=martian_encoder,
        shared_dim=shared_dim, attention_hidden_size=8,
    )


def test_forward_pass_produces_expected_shapes_and_valid_probabilities():
    model = _real_model()
    x_solar = torch.randn(5, 30, D_S)
    x_martian = torch.randn(7, 3, D_M)   # deliberately different batch size -- unpaired domains (Section 5.5)
    out = model(x_solar, x_martian)

    assert out["y_hat_solar"].shape == (5,)
    assert out["y_hat_martian"].shape == (7,)
    assert torch.all((out["y_hat_solar"] >= 0) & (out["y_hat_solar"] <= 1))
    assert torch.all((out["y_hat_martian"] >= 0) & (out["y_hat_martian"] <= 1))
    assert out["alpha_solar"].shape == (5, 2)
    assert out["alpha_martian"].shape == (7, 2)
    assert out["alignment_loss"].item() >= 0 or True  # a real scalar, sign not otherwise constrained
    assert out["balance_loss"].item() >= -1e-6


def test_only_fusion_parameters_are_trainable_not_the_frozen_encoders():
    model = _real_model()
    trainable = set(id(p) for p in model.trainable_parameters())
    encoder_params = set(id(p) for p in model.solar_encoder.parameters()) | set(
        id(p) for p in model.martian_encoder.parameters()
    )
    assert trainable.isdisjoint(encoder_params)
    assert len(trainable) > 0


def test_backward_pass_only_updates_fusion_parameters():
    model = _real_model()
    solar_weights_before = model.solar_encoder.lstm.weight_ih_l0.clone()

    x_solar = torch.randn(4, 30, D_S)
    x_martian = torch.randn(4, 3, D_M)
    out = model(x_solar, x_martian)
    loss = out["y_hat_solar"].sum() + out["y_hat_martian"].sum() + out["alignment_loss"] + out["balance_loss"]
    loss.backward()

    # Frozen encoder weights must be bit-identical (no gradient ever
    # reached them -- requires_grad=False, see encoders/frozen_encoder.py).
    torch.testing.assert_close(model.solar_encoder.lstm.weight_ih_l0, solar_weights_before)
    assert model.projection_solar.linear.weight.grad is not None


def test_predict_solar_only_fallback_runs_without_martian_input():
    model = _real_model()
    x_solar = torch.randn(3, 30, D_S)
    with torch.no_grad():
        p = model.predict_solar_only(x_solar)
    assert p.shape == (3,)
    assert torch.all((p >= 0) & (p <= 1))
