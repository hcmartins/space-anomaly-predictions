"""Tests for adaptive weighting (Eq. 5.5-5.6) and balance regularization (Eq. 5.7)."""
import torch

from fusion.attention import AttentionScorer, fuse
from fusion.balance import balance_regularizer


def test_fuse_weights_are_nonnegative_and_sum_to_one():
    scorer = AttentionScorer(embedding_dim=8, hidden_size=4)
    h_solar = torch.randn(6, 8)
    h_martian = torch.randn(6, 8)
    z, alpha = fuse(h_solar, h_martian, scorer)
    assert torch.all(alpha >= 0)
    torch.testing.assert_close(alpha.sum(dim=1), torch.ones(6))
    assert z.shape == (6, 8)


def test_fuse_is_a_convex_combination_hand_verified():
    scorer = AttentionScorer(embedding_dim=2, hidden_size=3)
    h_solar = torch.tensor([[1.0, 0.0]])
    h_martian = torch.tensor([[0.0, 1.0]])
    with torch.no_grad():
        z, alpha = fuse(h_solar, h_martian, scorer)
    expected_z = alpha[0, 0] * h_solar[0] + alpha[0, 1] * h_martian[0]
    torch.testing.assert_close(z[0], expected_z)


def test_fuse_handles_mismatched_batch_sizes_via_other_domain_mean():
    # Real fix (2026-07-29): domains are independently sampled and can have
    # different batch sizes (Section 5.5.1's n_S != n_M) -- the other
    # domain's contribution is its batch MEAN, broadcast across every row
    # of the own-domain batch.
    scorer = AttentionScorer(embedding_dim=2, hidden_size=3)
    own = torch.randn(5, 2)
    other = torch.tensor([[2.0, 4.0], [4.0, 8.0]])   # other domain batch, size 2, mean = [3, 6]
    with torch.no_grad():
        z, alpha = fuse(own, other, scorer)
    assert z.shape == (5, 2)
    assert alpha.shape == (5, 2)
    # Every row must combine `own`'s own row with the SAME other-domain
    # mean, [3, 6] -- verify hand-computed for row 0.
    expected_z0 = alpha[0, 0] * own[0] + alpha[0, 1] * torch.tensor([3.0, 6.0])
    torch.testing.assert_close(z[0], expected_z0)


def test_fuse_equal_scores_give_equal_weights():
    # A scorer whose weights are all zero gives identical (zero) scores
    # for any input -> softmax([0,0]) = [0.5, 0.5] exactly.
    scorer = AttentionScorer(embedding_dim=3, hidden_size=2)
    with torch.no_grad():
        for p in scorer.parameters():
            p.zero_()
    h_solar = torch.randn(4, 3)
    h_martian = torch.randn(4, 3)
    _z, alpha = fuse(h_solar, h_martian, scorer)
    torch.testing.assert_close(alpha, torch.full((4, 2), 0.5))


def test_balance_regularizer_zero_when_exactly_uniform():
    alpha = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    loss = balance_regularizer(alpha)
    assert abs(loss.item()) < 1e-6


def test_balance_regularizer_hand_computed():
    # A single batch entirely favoring domain S: alpha_bar = [1.0, 0.0] (clamped).
    alpha = torch.tensor([[1.0, 0.0]])
    loss = balance_regularizer(alpha)
    eps = 1e-12
    expected = 1.0 * torch.log(torch.tensor(1.0 / 0.5)) + eps * torch.log(torch.tensor(eps / 0.5))
    assert abs(loss.item() - expected.item()) < 1e-4


def test_balance_regularizer_positive_when_imbalanced():
    alpha = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    loss = balance_regularizer(alpha)
    assert loss.item() > 0
