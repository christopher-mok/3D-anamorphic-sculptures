"""Broad-phase (bounding spheres) and SDF-based fine-phase collision tests."""

from __future__ import annotations

import torch


def broad_phase_pairs(t: torch.Tensor, r: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Pairs (i<j) whose bounding spheres overlap. t [N,3], r [N] -> [P,2]."""
    n = t.shape[0]
    if n < 2:
        return torch.zeros(0, 2, dtype=torch.long, device=t.device)
    d = torch.cdist(t, t)
    close = d < (r[:, None] + r[None, :] + margin)
    close = torch.triu(close, diagonal=1)
    return torch.nonzero(close)


def broad_phase_cross(ta, ra, tb, rb, margin: float = 0.0) -> torch.Tensor:
    """Pairs (i in A, j in B) whose bounding spheres overlap -> [P,2]."""
    if ta.shape[0] == 0 or tb.shape[0] == 0:
        return torch.zeros(0, 2, dtype=torch.long, device=ta.device)
    d = torch.cdist(ta, tb)
    return torch.nonzero(d < (ra[:, None] + rb[None, :] + margin))


def aabb_overlap(min_a, max_a, min_b, max_b) -> torch.Tensor:
    return ((min_a <= max_b) & (min_b <= max_a)).all(-1)


def pair_penetration(points_a: torch.Tensor, sdf_b_at_points_a: torch.Tensor) -> torch.Tensor:
    """Max penetration depth of A's surface samples inside B: [P]."""
    return torch.relu(-sdf_b_at_points_a).amax(-1)


@torch.no_grad()
def reference_intersections(a, library, max_faces: int = 20000) -> list[tuple[int, int]]:
    """Accurate (slow) intersection check used for evaluation and verification.

    For every broad-phase pair, dense points of each object's ORIGINAL mesh (surface
    samples + all vertices + edge midpoints) are tested against the other mesh with the
    exact generalized winding number (> 0.5 = inside). Meshes with more than
    ``max_faces`` faces are tested against their rendering proxy instead.
    Returns intersecting pairs (i < j).
    """
    from .sdf import winding_numbers

    if len(a) < 2:
        return []
    R, s, t, ids = a.rotation_matrices(), torch.exp(a.log_scale), a.translation, a.mesh_ids.tolist()
    pairs = broad_phase_pairs(t, s * library.radii[a.mesh_ids]).tolist()
    out = []
    for i, j in pairs:
        hit = False
        for p, q in ((i, j), (j, i)):
            world = library.reference_points(ids[p]) @ R[p].T * s[p] + t[p]
            local = (world - t[q]) @ R[q] / s[q]
            near = local.norm(dim=-1) <= 1.0 + 1e-4  # canonical bounding sphere of q
            if not near.any():
                continue
            lod = "original" if library.entries[ids[q]].triangles <= max_faces else "proxy"
            v, f = library.proxy(ids[q], lod)
            if (winding_numbers(local[near], v, f.long(), chunk=2048).abs() > 0.5).any():
                hit = True
                break
        if hit:
            out.append((min(i, j), max(i, j)))
    return out
