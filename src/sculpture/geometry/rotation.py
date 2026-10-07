"""Rotation utilities. The optimization parameterization is the continuous 6D
representation of Zhou et al. (CVPR 2019): the first two columns of R,
re-orthonormalized with Gram-Schmidt. Euler angles are never used."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """[..., 6] -> [..., 3, 3]; columns of the result are (b1, b2, b3)."""
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] -> [..., 6] (first two columns, concatenated)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternions [..., 4] in (w, x, y, z) order -> [..., 3, 3]."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def axis_angle_to_matrix(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues: rotation vectors [..., 3] -> [..., 3, 3]."""
    angle = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = v / angle
    half = 0.5 * angle
    q = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
    return quaternion_to_matrix(q)


def super_fibonacci_rotations(n: int) -> torch.Tensor:
    """Approximately uniform deterministic SO(3) samples (Alexa, CVPR 2022)."""
    phi = math.sqrt(2.0)
    psi = 1.533751168755204288118041
    i = np.arange(n, dtype=np.float64)
    s = i + 0.5
    r = np.sqrt(s / n)
    R = np.sqrt(1.0 - s / n)
    alpha = 2.0 * np.pi * s / phi
    beta = 2.0 * np.pi * s / psi
    q = np.stack([r * np.sin(alpha), r * np.cos(alpha), R * np.sin(beta), R * np.cos(beta)], axis=-1)
    return quaternion_to_matrix(torch.from_numpy(q).float())


def random_rotations(n: int, generator: torch.Generator | None = None, device="cpu") -> torch.Tensor:
    """Uniform random rotations via normalized Gaussian quaternions."""
    q = torch.randn(n, 4, generator=generator, device=device if generator is None else generator.device)
    return quaternion_to_matrix(q.to(device))


def perturb_rotations(R: torch.Tensor, sigma_rad: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Left-multiply by random small rotations with Gaussian axis-angle std ``sigma_rad``."""
    gdev = generator.device if generator is not None else R.device
    v = torch.randn(R.shape[:-2] + (3,), generator=generator, device=gdev).to(R.device) * sigma_rad
    return axis_angle_to_matrix(v) @ R


def rotation_angle_between(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (radians) between rotation batches."""
    tr = (R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.arccos(((tr - 1) / 2).clamp(-1.0, 1.0))
