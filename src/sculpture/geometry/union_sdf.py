"""Soft union of transformed model SDFs:  phi_X(x) = softmin_i phi_i(x).

phi_i(x) = s_i * phi_mesh(R_i^T (x - t_i) / s_i)   (isotropic scale rescales distance)

Broad phase: for (instance, point) pairs farther than ``margin`` from the
instance's bounding sphere the exact SDF is replaced by the sphere distance
|x - t_i| - s_i (a lower bound that is still differentiable in t_i and s_i).
Those terms are exponentially small in the soft-min anyway.
"""

from __future__ import annotations

import torch

from ..scene.assembly import Assembly


def instance_sdf(sdf_lib, a: Assembly, pts: torch.Tensor, margin: float) -> torch.Tensor:
    """phi_i at shared points: [N, Q]."""
    R = a.rotation_matrices()
    s = torch.exp(a.log_scale)
    t = a.translation
    d_sph = torch.cdist(t, pts) - s[:, None]  # canonical bounding radius is 1
    with torch.no_grad():
        near = d_sph < margin
        ii, qq = near.nonzero(as_tuple=True)
    if ii.numel() == 0:
        return d_sph
    local = torch.einsum("pj,pji->pi", pts[qq] - t[ii], R[ii]) / s[ii, None]
    vals = torch.zeros(ii.shape[0], device=pts.device, dtype=pts.dtype)
    mids = a.mesh_ids[ii]
    for m in torch.unique(mids).tolist():
        sel = (mids == m).nonzero(as_tuple=True)[0]
        vals = vals.index_put((sel,), sdf_lib.query_local(m, local[sel]) * s[ii[sel]])
    return d_sph.index_put((ii, qq), vals)


def softmin(x: torch.Tensor, tau: float, dim: int) -> torch.Tensor:
    """Log-sum-exp soft minimum. Biased low by up to tau*log(n)."""
    return -tau * torch.logsumexp(-x / tau, dim=dim)


def softmin_st(x: torch.Tensor, tau: float, dim: int) -> torch.Tensor:
    """Straight-through soft minimum: forward value = exact min, gradient = soft-min
    gradient (softmax weights over all terms). Avoids the tau*log(n) bias that would
    otherwise report rays as 'hit' when no object actually intersects them."""
    soft = softmin(x, tau, dim)
    return x.amin(dim=dim) + (soft - soft.detach())


def union_sdf(sdf_lib, a: Assembly, pts: torch.Tensor, tau: float, margin: float) -> torch.Tensor:
    """phi_X at points [Q] (soft union with temperature tau)."""
    if len(a) == 0:
        return torch.full(pts.shape[:1], 10.0, device=pts.device)
    return softmin_st(instance_sdf(sdf_lib, a, pts, margin), tau, dim=0)
