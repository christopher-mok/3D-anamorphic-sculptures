"""SDF-based collision penalty and hard test with a clearance margin.

Objects i, j are considered intersecting when any collision point x of one
lies closer than ``margin`` to (or inside) the other:  phi_j(x) < margin.
A positive margin is deliberately conservative: it absorbs the SDF grid's
interpolation error and the finite point sampling, so "no collision" under
this test implies no intersection in the dense reference check
(geometry/collisions.py::reference_intersections).

Collision points = every proxy vertex + edge midpoint (corners and edges are
where boxes poke into neighbours) + surface samples; see MeshLibrary.
"""

from __future__ import annotations

import torch

from ..geometry.collisions import broad_phase_cross, broad_phase_pairs
from ..scene.assembly import Assembly


def collision_points(a: Assembly, library, n: int | None = None) -> torch.Tensor:
    """World-space collision points of every instance [N, n, 3]."""
    P = library.collision_points
    if n is not None and n < P.shape[1]:
        P = P[:, :n]
    pts = P[a.mesh_ids]
    R = a.rotation_matrices()
    s = torch.exp(a.log_scale)
    return torch.einsum("npj,nij->npi", pts, R) * s[:, None, None] + a.translation[:, None, :]


def _pair_terms(A: Assembly, B: Assembly, pairs: torch.Tensor, library, sdf_lib, n: int | None, margin: float):
    """(penalty [P], violation [P]) for broad-phase pairs.

    violation = max over both directions of (margin - phi_other(x)); > 0 means "colliding".
    penalty   = mean + max of ReLU(margin - phi)^2 (the max term keeps single corner pokes
                from being diluted by the many non-penetrating samples).
    """
    if pairs.shape[0] == 0:
        z = torch.zeros(0, device=library.device)
        return z, z
    ia, ib = pairs[:, 0], pairs[:, 1]
    Ra, Rb = A.rotation_matrices(), B.rotation_matrices()
    sa, sb = torch.exp(A.log_scale), torch.exp(B.log_scale)
    pa = collision_points(A[ia], library, n)
    pb = collision_points(B[ib], library, n)
    phi_b_at_a = sdf_lib.query_instances(B.mesh_ids[ib], Rb[ib], sb[ib], B.translation[ib], pa)
    phi_a_at_b = sdf_lib.query_instances(A.mesh_ids[ia], Ra[ia], sa[ia], A.translation[ia], pb)
    va, vb = torch.relu(margin - phi_b_at_a), torch.relu(margin - phi_a_at_b)
    penalty = (va ** 2).mean(-1) + (vb ** 2).mean(-1) + (va ** 2).amax(-1) + (vb ** 2).amax(-1)
    violation = torch.maximum((margin - phi_b_at_a).amax(-1), (margin - phi_a_at_b).amax(-1))
    return penalty, violation


def _radii(a: Assembly, library) -> torch.Tensor:
    return torch.exp(a.log_scale) * library.radii[a.mesh_ids]


def collision_loss(assembly: Assembly, library, sdf_lib, n: int = 128, margin: float = 0.0) -> torch.Tensor:
    """Sum over broad-phase pairs of the clearance penalty."""
    if len(assembly) < 2:
        return torch.zeros((), device=library.device)
    with torch.no_grad():
        pairs = broad_phase_pairs(assembly.translation, _radii(assembly, library), margin=2 * margin)
    pen, _ = _pair_terms(assembly, assembly, pairs, library, sdf_lib, n, margin)
    return pen.sum()


def cross_collision_loss(A: Assembly, B: Assembly, library, sdf_lib, n: int = 96, margin: float = 0.0) -> torch.Tensor:
    """Per-object penalty of each A[i] against all of B: [len(A)]."""
    out = torch.zeros(len(A), device=library.device)
    if len(A) == 0 or len(B) == 0:
        return out
    with torch.no_grad():
        pairs = broad_phase_cross(A.translation, _radii(A, library), B.translation, _radii(B, library), margin=2 * margin)
    pen, _ = _pair_terms(A, B, pairs, library, sdf_lib, n, margin)
    return out.index_add(0, pairs[:, 0], pen) if pairs.shape[0] else out


@torch.no_grad()
def colliding_pairs(A: Assembly, library, sdf_lib, margin: float, n: int | None = None, B: Assembly | None = None):
    """Hard (conservative) test with ALL collision points by default.

    Returns (pairs [P,2], violation [P]) for pairs closer than ``margin``.
    If B is None, tests pairs within A (i<j); else pairs (i in A, j in B).
    """
    if B is None:
        pairs = broad_phase_pairs(A.translation, _radii(A, library), margin=2 * margin)
        B = A
    else:
        pairs = broad_phase_cross(A.translation, _radii(A, library), B.translation, _radii(B, library), margin=2 * margin)
    if pairs.shape[0] == 0:
        return pairs, torch.zeros(0, device=library.device)
    out = []
    for s in range(0, pairs.shape[0], 512):
        _, v = _pair_terms(A, B, pairs[s : s + 512], library, sdf_lib, n, margin)
        out.append(v)
    viol = torch.cat(out)
    keep = viol > 0
    return pairs[keep], viol[keep]
