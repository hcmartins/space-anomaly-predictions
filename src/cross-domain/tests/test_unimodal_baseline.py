"""Real integration test: loads the SAME real checkpoints as
tests/test_frozen_encoders.py, this time extracting the real head layer
(instead of discarding it) and confirming it produces a valid probability
on top of the frozen encoder's own real hidden state."""
import torch

from encoders.martian_encoder import load_martian_encoder
from encoders.solar_encoder import CROSS_DOMAIN_ROOT, load_encoders_config, load_solar_encoder
from evaluation.unimodal_baseline import load_unimodal_head


def _checkpoint_path(domain: str) -> str:
    config = load_encoders_config()[domain]
    return str((CROSS_DOMAIN_ROOT / config["checkpoint_path"]).resolve())


def test_solar_unimodal_head_produces_valid_probability():
    encoder = load_solar_encoder()
    config = load_encoders_config()["solar"]
    head = load_unimodal_head(_checkpoint_path("solar"), hidden_size=config["hidden_size"], checkpoint_format=config["checkpoint_format"])

    x = torch.randn(4, 30, config["input_size"])
    h = encoder(x)
    y_prob = head(h)
    assert y_prob.shape == (4,)
    assert torch.all((y_prob >= 0) & (y_prob <= 1))


def test_martian_unimodal_head_produces_valid_probability():
    encoder, _provenance = load_martian_encoder()
    config = load_encoders_config()["martian"]
    head = load_unimodal_head(_checkpoint_path("martian"), hidden_size=config["hidden_size"], checkpoint_format=config["checkpoint_format"])

    x = torch.randn(4, 3, config["input_size"])
    h = encoder(x)
    y_prob = head(h)
    assert y_prob.shape == (4,)
    assert torch.all((y_prob >= 0) & (y_prob <= 1))


def test_unimodal_head_has_no_trainable_parameters():
    config = load_encoders_config()["solar"]
    head = load_unimodal_head(_checkpoint_path("solar"), hidden_size=config["hidden_size"], checkpoint_format=config["checkpoint_format"])
    assert all(not p.requires_grad for p in head.parameters())
    assert not head.training


def test_unimodal_head_rejects_unknown_checkpoint_format():
    config = load_encoders_config()["solar"]
    try:
        load_unimodal_head(_checkpoint_path("solar"), hidden_size=config["hidden_size"], checkpoint_format="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
