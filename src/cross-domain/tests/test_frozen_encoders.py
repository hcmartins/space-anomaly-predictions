"""Real integration test: loads the REAL solar and martian checkpoints and
confirms the frozen encoders produce embeddings of the expected shape."""
import torch

from encoders.martian_encoder import load_martian_encoder
from encoders.solar_encoder import load_encoders_config, load_solar_encoder


def test_solar_encoder_loads_real_checkpoint_and_produces_expected_embedding_shape():
    encoder = load_solar_encoder()
    assert encoder.hidden_size == 32
    d_S = load_encoders_config()["solar"]["input_size"]   # GOES-only (2) or GOES+SHARP (20), config-driven
    x = torch.randn(4, 30, d_S)   # (batch, T_G=30, d_S) -- matches src/solar's real sequence shape
    h = encoder(x)
    assert h.shape == (4, 32)


def test_martian_encoder_loads_real_checkpoint_and_reports_real_provenance():
    encoder, provenance = load_martian_encoder()
    assert encoder.hidden_size == 16
    assert "REAL" in provenance
    d_M = load_encoders_config()["martian"]["input_size"]   # real storm-detection feature width (config-driven)
    x = torch.randn(4, 3, d_M)   # (batch, window=3, d_M) -- matches src/martian's real feature width
    h = encoder(x)
    assert h.shape == (4, 16)


def test_frozen_encoders_have_no_trainable_parameters():
    encoder = load_solar_encoder()
    assert all(not p.requires_grad for p in encoder.parameters())
    assert not encoder.training
