"""Asymmetric silhouette loss shared by every method.

L_cov = sum(I (1-R)^2) / (sum(I) + eps)
L_neg = sum((1-I) R^2) / (sum(1-I) + eps)
L_view = w_cov L_cov + w_neg L_neg,       L_image = mean_v L_view
"""

from __future__ import annotations

import torch

EPS = 1e-6


def view_loss_terms(R: torch.Tensor, I: torch.Tensor, eps: float = EPS):
    """R [..., V, H, W], I [V, H, W] -> (L_cov, L_neg) each [..., V]."""
    nI = I.sum((-1, -2)) + eps
    nB = (1 - I).sum((-1, -2)) + eps
    cov = (I * (1 - R) ** 2).sum((-1, -2)) / nI
    neg = ((1 - I) * R ** 2).sum((-1, -2)) / nB
    return cov, neg


def view_losses(R, I, w_cov: float = 1.0, w_neg: float = 4.0) -> torch.Tensor:
    cov, neg = view_loss_terms(R, I)
    return w_cov * cov + w_neg * neg


def image_loss(R, I, w_cov: float = 1.0, w_neg: float = 4.0) -> torch.Tensor:
    """Mean over views: [...]."""
    return view_losses(R, I, w_cov, w_neg).mean(-1)


def gradient_utility(R, I, w_cov: float = 1.0, w_neg: float = 4.0) -> torch.Tensor:
    """W = -dL_image/dR, per pixel [..., V, H, W] (first-order utility)."""
    V = I.shape[0]
    nI = I.sum((-1, -2), keepdim=True) + EPS
    nB = (1 - I).sum((-1, -2), keepdim=True) + EPS
    return (2 * w_cov * I * (1 - R) / nI - 2 * w_neg * (1 - I) * R / nB) / V


def secant_utility(R, I, w_cov: float = 1.0, w_neg: float = 4.0) -> torch.Tensor:
    """Exact loss decrease if pixel p becomes fully covered: L_p(R) - L_p(1).

    For a binary candidate mask S composited by union (R' = max(R, S)), the
    change of L_image is exactly -sum_p W_p S_p. Unlike the gradient utility it
    correctly charges spill on currently empty background pixels (where the
    derivative of R^2 vanishes).
    """
    V = I.shape[0]
    nI = I.sum((-1, -2), keepdim=True) + EPS
    nB = (1 - I).sum((-1, -2), keepdim=True) + EPS
    return (w_cov * I * (1 - R) ** 2 / nI - w_neg * (1 - I) * (1 - R ** 2) / nB) / V


def soft_union(R: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Probabilistic union 1 - (1-R)(1-S); equals max(R, S) for binary inputs."""
    return 1 - (1 - R) * (1 - S)
