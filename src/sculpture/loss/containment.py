"""Visual-hull containment  L_contain = mean_q ReLU(phi_Omega(T(q)))^2."""

from __future__ import annotations

import torch

from ..scene.assembly import Assembly


def surface_points(assembly: Assembly, library, n: int | None = None) -> torch.Tensor:
    """World-space surface samples of every instance [N, n, 3]."""
    S = library.samples
    if n is not None and n < S.shape[1]:
        S = S[:, :n]
    pts = S[assembly.mesh_ids]  # [N, n, 3]
    R = assembly.rotation_matrices()
    s = torch.exp(assembly.log_scale)
    return torch.einsum("npj,nij->npi", pts, R) * s[:, None, None] + assembly.translation[:, None, :]


def containment_per_object(assembly: Assembly, library, hull, n: int | None = None):
    """(mean ReLU(phi)^2 per object [N], max phi per object [N])."""
    if len(assembly) == 0:
        z = torch.zeros(0, device=library.device)
        return z, z
    pts = surface_points(assembly, library, n)
    phi = hull.query_sdf(pts)
    return (torch.relu(phi) ** 2).mean(-1), phi.amax(-1)


def containment_loss(assembly: Assembly, library, hull, n: int | None = None) -> torch.Tensor:
    per, _ = containment_per_object(assembly, library, hull, n)
    return per.mean() if per.numel() else torch.zeros((), device=library.device)
