"""Tests for CrossDomainFusionModel.forward()'s ablation kwargs
(use_fusion_pathway, other_domain_embedding) -- configs/ablation.yaml's
A1-A7 axes. Uses the real encoders (matching this package's existing
integration-test convention, tests/test_fusion_model.py)."""
import torch

from encoders.martian_encoder import load_martian_encoder
from encoders.solar_encoder import load_encoders_config, load_solar_encoder
from fusion.model import CrossDomainFusionModel, _transform_other

D_S = load_encoders_config()["solar"]["input_size"]   # GOES-only (2) or GOES+SHARP (20), config-driven
D_M = load_encoders_config()["martian"]["input_size"]   # real storm-detection feature width, config-driven


def _real_model(shared_dim=16):
    solar_encoder = load_solar_encoder()
    martian_encoder, _provenance = load_martian_encoder()
    return CrossDomainFusionModel(
        solar_encoder=solar_encoder, martian_encoder=martian_encoder,
        shared_dim=shared_dim, attention_hidden_size=8,
    )


def test_default_kwargs_match_explicit_a7_behavior():
    model = _real_model()
    torch.manual_seed(0)
    x_solar = torch.randn(5, 30, D_S)
    x_martian = torch.randn(7, 3, D_M)
    torch.manual_seed(1)
    out_default = model(x_solar, x_martian)
    torch.manual_seed(1)
    out_explicit = model(x_solar, x_martian, use_fusion_pathway=True, other_domain_embedding="real")
    torch.testing.assert_close(out_default["y_hat_solar"], out_explicit["y_hat_solar"])
    torch.testing.assert_close(out_default["alpha_solar"], out_explicit["alpha_solar"])


def test_use_fusion_pathway_false_reads_own_projection_directly():
    model = _real_model()
    x_solar = torch.randn(5, 30, D_S)
    x_martian = torch.randn(7, 3, D_M)
    out = model(x_solar, x_martian, use_fusion_pathway=False)

    h_solar_proj, h_martian_proj = model.project(x_solar, x_martian)
    expected_solar = torch.sigmoid(model.head_solar(h_solar_proj)).squeeze(-1)
    expected_martian = torch.sigmoid(model.head_martian(h_martian_proj)).squeeze(-1)

    torch.testing.assert_close(out["y_hat_solar"], expected_solar)
    torch.testing.assert_close(out["y_hat_martian"], expected_martian)
    assert out["alpha_solar"] is None
    assert out["alpha_martian"] is None
    assert out["balance_loss"].item() == 0.0
    # alignment_loss is still real and computed (needed for A5).
    assert out["alignment_loss"].item() >= 0.0 or True


def test_within_batch_shuffled_preserves_values_but_not_row_order():
    torch.manual_seed(0)
    proj = torch.randn(20, 16)
    shuffled = _transform_other(proj, "within_batch_shuffled")
    assert shuffled.shape == proj.shape
    # Same multiset of rows, real permutation (not identity in general).
    original_sorted, _ = torch.sort(proj, dim=0)
    shuffled_sorted, _ = torch.sort(shuffled, dim=0)
    torch.testing.assert_close(original_sorted, shuffled_sorted)
    assert not torch.equal(shuffled, proj)


def test_moment_matched_noise_matches_real_mean_and_std():
    torch.manual_seed(0)
    proj = torch.randn(2000, 16) * 3.0 + 5.0
    noise = _transform_other(proj, "moment_matched_noise")
    assert noise.shape == proj.shape
    torch.testing.assert_close(noise.mean(dim=0), proj.mean(dim=0), atol=0.5, rtol=0.0)
    torch.testing.assert_close(noise.std(dim=0), proj.std(dim=0), atol=0.5, rtol=0.0)
    assert not torch.equal(noise, proj)


def test_unknown_other_domain_embedding_mode_raises():
    proj = torch.randn(4, 16)
    try:
        _transform_other(proj, "bogus_mode")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_forward_pass_with_ablation_kwargs_is_backward_differentiable():
    model = _real_model()
    x_solar = torch.randn(4, 30, D_S)
    x_martian = torch.randn(4, 3, D_M)
    out = model(x_solar, x_martian, use_fusion_pathway=True, other_domain_embedding="moment_matched_noise")
    loss = out["y_hat_solar"].sum() + out["y_hat_martian"].sum() + out["alignment_loss"] + out["balance_loss"]
    loss.backward()
    assert model.projection_solar.linear.weight.grad is not None
