"""Similarity transforms ``x_world = s * R @ x_local + t`` (isotropic scale)."""

from __future__ import annotations

import torch


def local_to_world(points: torch.Tensor, R: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Transform canonical points by N instance transforms.

    points: [P, 3] (shared) or [N, P, 3]; R: [N, 3, 3]; s: [N]; t: [N, 3] -> [N, P, 3]
    """
    if points.dim() == 2:
        out = torch.einsum("pj,nij->npi", points, R)
    else:
        out = torch.einsum("npj,nij->npi", points, R)
    return out * s[:, None, None] + t[:, None, :]


def world_to_local(points: torch.Tensor, R: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Inverse transform. points: [N, P, 3] world -> [N, P, 3] local."""
    d = points - t[:, None, :]
    return torch.einsum("npi,nij->npj", d, R) / s[:, None, None]


def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def similarity_matrix(R, s, t):
    """4x4 matrices for N transforms (torch). Returns [N, 4, 4]."""
    N = R.shape[0]
    M = torch.zeros(N, 4, 4, dtype=R.dtype, device=R.device)
    M[:, :3, :3] = R * s[:, None, None]
    M[:, :3, 3] = t
    M[:, 3, 3] = 1.0
    return M
