"""Tests for the shared-space projection (Eq. 5.2)."""
import torch

from fusion.projections import DomainProjection


def test_projection_output_shape():
    proj = DomainProjection(input_dim=32, shared_dim=16)
    h = torch.randn(5, 32)
    out = proj(h)
    assert out.shape == (5, 16)


def test_projection_is_affine_no_nonlinearity():
    # [CONFIRM resolved]: Eq. (5.2) is a plain affine map -- confirm no
    # activation is applied by checking linearity: proj(a*h) == a*proj(h) + (1-a)*proj(0).
    proj = DomainProjection(input_dim=4, shared_dim=3)
    h = torch.randn(2, 4)
    with torch.no_grad():
        out_h = proj(h)
        out_2h = proj(2 * h)
        out_0 = proj(torch.zeros_like(h))
    # affine: f(2h) - f(0) == 2*(f(h) - f(0))
    torch.testing.assert_close(out_2h - out_0, 2 * (out_h - out_0))
