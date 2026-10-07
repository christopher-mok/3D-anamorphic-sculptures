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
def reference_intersections(a, library, band_voxels: float = 1.0) -> list[tuple[int, int]]:
    """Accurate intersection check against the ORIGINAL meshes ("substituted back").

    Dense points of each piece's original mesh (surface samples + vertices + edge midpoints)
    are classified against the other piece's original mesh in three tiers:
      1. model SDF grid (built from the original): clearly inside (phi < -band) -> hit,
         clearly outside (phi > band) -> skip;  band = 1 voxel + coarse-LOD error;
      2. band points: generalized winding number against the PROXY, trusted when the point is
         farther from the proxy surface than the proxy's measured deviation (then proxy and
         original agree on inside/outside);
      3. the remaining points (within the proxy error of its surface): exact winding number
         against the full-resolution ORIGINAL mesh.
    Returns intersecting pairs (i < j).
    """
    from .mesh_preprocess import point_triangle_distance
    from .sdf import winding_numbers

    if len(a) < 2:
        return []
    sdf = library._sdf if library._sdf is not None else library.sdf()  # reuse the grids already in use
    voxel = (sdf.hi - sdf.lo) / (sdf.resolution - 1)
    R, s, t, ids = a.rotation_matrices(), torch.exp(a.log_scale), a.translation, a.mesh_ids.tolist()
    pairs = broad_phase_pairs(t, s * library.radii[a.mesh_ids]).tolist()
    out = []
    for i, j in pairs:
        hit = False
        for p, q in ((i, j), (j, i)):
            e = library.entries[ids[q]]
            band = band_voxels * voxel + float(e.data.get("coarse_error", 0.0))
            world = library.reference_points(ids[p]) @ R[p].T * s[p] + t[p]
            local = (world - t[q]) @ R[q] / s[q]
            local = local[local.norm(dim=-1) <= 1.0 + 1e-4]  # canonical bounding sphere of q
            if len(local) == 0:
                continue
            phi = sdf.query_local(ids[q], local)
            if bool((phi < -band).any()):
                hit = True
                break
            amb = local[phi.abs() <= band]
            if len(amb) == 0:
                continue
            pv, pf = library.proxy(ids[q], "proxy")
            perr = float(e.data.get("proxy_error", 0.0))
            if perr > 0 and len(pf) < e.triangles:
                w = winding_numbers(amb, pv, pf.long(), chunk=max(16, 4_000_000 // max(1, len(pf)))).abs()
                d = torch.as_tensor(point_triangle_distance(amb, pv[pf.long()]), device=amb.device)
                sure = d > 2.5 * perr + 1e-4  # margin over the 99.9th-percentile deviation
                if bool(((w > 0.5) & sure).any()):
                    hit = True
                    break
                amb = amb[~sure]
                if len(amb) == 0:
                    continue
            v, f = library.proxy(ids[q], "original")
            if bool((winding_numbers(amb, v, f.long(), chunk=max(16, 4_000_000 // max(1, len(f)))).abs() > 0.5).any()):
                hit = True
                break
        if hit:
            out.append((min(i, j), max(i, j)))
    return out
