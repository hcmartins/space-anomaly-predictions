"""Hand-computed tests for the moment-matching alignment loss (Eq. 5.3-5.4)."""
import torch

from fusion.alignment import batch_moments, moment_alignment_loss


def test_batch_moments_hand_computed():
    h = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean, var = batch_moments(h)
    # column means: [3, 4]; biased variance (1/n): [(4+0+4)/3, (4+0+4)/3] = [8/3, 8/3]
    torch.testing.assert_close(mean, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(var, torch.tensor([8 / 3, 8 / 3]))


def test_moment_alignment_loss_zero_for_identical_batches():
    h = torch.randn(10, 4)
    loss = moment_alignment_loss(h, h.clone())
    assert loss.item() < 1e-6


def test_moment_alignment_loss_hand_computed():
    h_solar = torch.tensor([[0.0], [2.0]])     # mean=1, var=1
    h_martian = torch.tensor([[3.0], [3.0]])   # mean=3, var=0
    loss = moment_alignment_loss(h_solar, h_martian, variance_weight=2.0)
    # d=1: mean_term = (1-3)^2/1 = 4; variance_term = (1-0)^2/1 = 1, weighted by beta=2 -> 2
    expected = 4.0 + 2.0 * 1.0
    assert abs(loss.item() - expected) < 1e-6


def test_moment_alignment_loss_variance_weight_zero_ignores_variance_mismatch():
    h_solar = torch.tensor([[0.0], [2.0]])
    h_martian = torch.tensor([[1.0], [1.0]])   # same mean (1), different variance
    loss = moment_alignment_loss(h_solar, h_martian, variance_weight=0.0)
    assert abs(loss.item()) < 1e-6
