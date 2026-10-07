import torch

from sculpture.loss.silhouette import image_loss, secant_utility, soft_union, view_losses


def _masks():
    g = torch.Generator().manual_seed(0)
    I = (torch.rand(2, 32, 32, generator=g) > 0.6).float()
    R = (torch.rand(2, 32, 32, generator=g) > 0.7).float()
    S = (torch.rand(2, 32, 32, generator=g) > 0.8).float()
    return I, R, S


def test_perfect_and_empty():
    I, _, _ = _masks()
    assert view_losses(I, I).abs().max() < 1e-6
    assert torch.allclose(view_losses(torch.zeros_like(I), I, 1.0, 4.0), torch.ones(2), atol=1e-5)
    assert torch.allclose(view_losses(torch.ones_like(I), I, 1.0, 4.0), torch.full((2,), 4.0), atol=1e-4)


def test_secant_utility_is_exact_for_binary_union():
    I, R, S = _masks()
    W = secant_utility(R, I, 1.0, 4.0)
    delta = image_loss(torch.maximum(R, S), I) - image_loss(R, I)
    assert torch.allclose(delta, -(W * S * (1 - R)).sum(), atol=1e-6) or torch.allclose(delta, -(W * S).sum(), atol=1e-6)


def test_soft_union_binary_equals_max():
    _, R, S = _masks()
    assert torch.equal(soft_union(R, S), torch.maximum(R, S))


def test_spill_weight():
    I = torch.zeros(1, 8, 8)
    I[0, :4] = 1
    R = torch.zeros(1, 8, 8)
    R[0, 4:] = 1  # all spill, no coverage
    assert torch.allclose(view_losses(R, I, 1.0, 5.0), torch.tensor([6.0]))
