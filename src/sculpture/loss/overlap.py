"""Volumetric penetration: soft pairwise overlap volume of the pieces.

  L_overlap = mean_q  sum_{i<j}  o_i(q) o_j(q),   o_i = sigmoid(-phi_i / eps)

a Monte Carlo estimate (points q inside the visual hull) of how much volume pieces share.
Bounded and smooth, and it also penalizes a piece sitting completely inside another, unlike
surface-sample penetration terms.
"""

from __future__ import annotations

import torch

from ..geometry.union_sdf import instance_sdf
from ..scene.assembly import Assembly


def overlap_from_phi(phi_i: torch.Tensor, eps: float) -> torch.Tensor:
    """phi_i [N, Q] -> scalar mean pairwise soft overlap."""
    occ = torch.sigmoid(-phi_i / eps)
    s1 = occ.sum(0)
    return (0.5 * (s1 * s1 - (occ * occ).sum(0))).mean()


def overlap_loss(sdf_lib, a: Assembly, points: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    if len(a) < 2:
        return torch.zeros((), device=points.device)
    return overlap_from_phi(instance_sdf(sdf_lib, a, points, margin=4 * eps), eps)
