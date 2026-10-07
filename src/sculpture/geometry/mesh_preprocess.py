"""Load and canonicalize a single mesh file."""

from __future__ import annotations

import logging

import numpy as np
import trimesh

log = logging.getLogger(__name__)


def load_mesh(path) -> trimesh.Trimesh:
    """Load any trimesh-supported file, merging scenes into one triangle mesh."""
    loaded = trimesh.load(str(path), force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):  # older trimesh versions
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"{path}: could not interpret file as a triangle mesh")
    mesh = loaded  # trimesh triangulates polygonal faces on load
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise ValueError(f"{path}: mesh has no faces")
    return mesh


def decimate(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadric decimation down to roughly ``max_faces`` faces (no-op if already smaller)."""
    if len(faces) <= max_faces:
        return vertices.astype(np.float32), faces.astype(np.int32)
    try:
        import fast_simplification

        reduction = 1.0 - max_faces / len(faces)
        v, f = fast_simplification.simplify(vertices.astype(np.float32), faces.astype(np.int32), target_reduction=reduction)
        if len(f) > 0:
            return v.astype(np.float32), f.astype(np.int32)
    except Exception as exc:  # pragma: no cover - depends on optional package
        log.warning("fast_simplification failed (%s); falling back to trimesh", exc)
    m = trimesh.Trimesh(vertices, faces, process=False)
    try:
        m = m.simplify_quadric_decimation(face_count=max_faces)
    except Exception:
        # last resort: keep a random subset of faces (still a valid silhouette proxy)
        idx = np.random.default_rng(0).choice(len(faces), max_faces, replace=False)
        m = trimesh.Trimesh(vertices, faces[idx], process=True)
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)


def point_triangle_distance(points, tris, chunk: int | None = None):
    """Exact unsigned distance from points [P,3] to a triangle soup [F,3,3] (min over faces).
    Torch implementation of the closest-point-on-triangle region test (Ericson, RTCD 5.1.5);
    runs on the GPU when available, chunked over points."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = torch.as_tensor(points, dtype=torch.float32, device=dev)
    T = torch.as_tensor(tris, dtype=torch.float32, device=dev)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, ac = b - a, c - a
    chunk = chunk or max(4, 3_000_000 // max(1, len(T)))
    out = torch.empty(len(P), device=dev)

    def dot(x, y):
        return (x * y).sum(-1)

    for s0 in range(0, len(P), chunk):
        p = P[s0 : s0 + chunk, None, :]                       # [k,1,3]
        ap, bp, cp = p - a, p - b, p - c                      # [k,F,3]
        d1, d2 = dot(ab, ap), dot(ac, ap)
        d3, d4 = dot(ab, bp), dot(ac, bp)
        d5, d6 = dot(ab, cp), dot(ac, cp)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        denom = (va + vb + vc).clamp_min(1e-20)
        v = vb / denom
        w = vc / denom
        q = a + ab * v[..., None] + ac * w[..., None]          # interior
        # edge / vertex regions (applied in reverse priority so vertices win)
        e_bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        t_bc = ((d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(1e-20))[..., None]
        q = torch.where(e_bc[..., None], b + (c - b) * t_bc, q)
        e_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        q = torch.where(e_ac[..., None], a + ac * (d2 / (d2 - d6).clamp_min(1e-20))[..., None], q)
        e_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        q = torch.where(e_ab[..., None], a + ab * (d1 / (d1 - d3).clamp_min(1e-20))[..., None], q)
        q = torch.where(((d6 >= 0) & (d5 <= d6))[..., None], c.expand_as(q), q)
        q = torch.where(((d3 >= 0) & (d4 <= d3))[..., None], b.expand_as(q), q)
        q = torch.where(((d1 <= 0) & (d2 <= 0))[..., None], a.expand_as(q), q)
        out[s0 : s0 + chunk] = (p - q).norm(dim=-1).amin(-1)
    return out.cpu().numpy()


def surface_deviation(v_a, f_a, v_b, f_b, n: int = 3000, seed: int = 0) -> float:
    """Two-sided surface deviation between meshes A and B (canonical units): the 99.9th
    percentile of EXACT point-to-surface distances from n surface samples (+ a vertex subset)
    of each mesh to the other mesh's triangles."""
    ma = trimesh.Trimesh(v_a, f_a, process=False)
    mb = trimesh.Trimesh(v_b, f_b, process=False)
    sa, _ = trimesh.sample.sample_surface(ma, n, seed=seed)
    sb, _ = trimesh.sample.sample_surface(mb, n, seed=seed + 1)
    pa = np.concatenate([sa, v_a[:: max(1, len(v_a) // n)]])
    pb = np.concatenate([sb, v_b[:: max(1, len(v_b) // n)]])
    d_ab = point_triangle_distance(pa, np.asarray(v_b)[np.asarray(f_b)])
    d_ba = point_triangle_distance(pb, np.asarray(v_a)[np.asarray(f_a)])
    return float(max(np.quantile(d_ab, 0.999), np.quantile(d_ba, 0.999)))


def adaptive_lod(vn: np.ndarray, f: np.ndarray, tolerance: float, max_faces: int, min_faces: int = 500):
    """Smallest quadric decimation (face-count ladder doubling from min_faces) whose surface
    deviation from the original is <= tolerance (canonical units), capped at max_faces.
    Returns (vertices, faces, deviation)."""
    if len(f) <= min_faces:
        return vn.astype(np.float32), f.astype(np.int32), 0.0
    target = min_faces
    while True:
        target = min(target, max_faces)
        v, ff = decimate(vn, f, target)
        if len(ff) >= len(f):  # no decimation happened: exact
            return v, ff, 0.0
        err = surface_deviation(v, ff, vn, f)
        if err <= tolerance or target >= max_faces:
            if err > tolerance:
                log.warning("LOD with %d faces deviates %.4f > tolerance %.4f (raise meshes.proxy_max_faces)", len(ff), err, tolerance)
            return v, ff, err
        target *= 2


def canonicalize(mesh: trimesh.Trimesh, cfg) -> dict:
    """Center at the bounding-box center, scale to unit bounding-sphere radius,
    build proxies and samples. Returns plain numpy arrays (cacheable)."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    bmin, bmax = v.min(0), v.max(0)
    center = 0.5 * (bmin + bmax)
    radius = float(np.linalg.norm(v - center, axis=1).max())
    if radius <= 0:
        raise ValueError("degenerate mesh (zero extent)")
    scale = 1.0 / radius
    vn = (v - center) * scale
    canon = trimesh.Trimesh(vn, f, process=False)

    # precise level-of-detail approximations: the optimizers only ever touch these; the original
    # mesh is substituted back for export, final evaluation and the final intersection check
    proxy_v, proxy_f, proxy_err = adaptive_lod(vn, f, float(cfg.get("proxy_tolerance", 0.003)), int(cfg.get("proxy_max_faces", 20000)))
    coarse_v, coarse_f, coarse_err = adaptive_lod(vn, f, float(cfg.get("coarse_tolerance", 0.015)), int(cfg.get("coarse_max_faces", 3000)), 250)

    n_samples = int(cfg.get("surface_samples", 1024))
    samples, face_idx = trimesh.sample.sample_surface_even(canon, n_samples, seed=0)
    if len(samples) < n_samples:  # sample_surface_even may return fewer points
        extra, extra_idx = trimesh.sample.sample_surface(canon, n_samples - len(samples), seed=1)
        samples = np.concatenate([samples, extra])
        face_idx = np.concatenate([face_idx, extra_idx])
    normals = canon.face_normals[face_idx]

    if canon.is_watertight and canon.is_volume:
        com = np.asarray(canon.center_mass)
    else:  # area-weighted surface centroid
        com = np.asarray(canon.triangles_center.T @ canon.area_faces / max(canon.area, 1e-12))

    return {
        "vertices": vn.astype(np.float32),
        "faces": f.astype(np.int32),
        "proxy_vertices": proxy_v,
        "proxy_faces": proxy_f,
        "coarse_vertices": coarse_v,
        "coarse_faces": coarse_f,
        "samples": samples.astype(np.float32),
        "sample_normals": normals.astype(np.float32),
        "bbox_min": vn.min(0).astype(np.float32),
        "bbox_max": vn.max(0).astype(np.float32),
        "radius": np.float32(1.0),
        "center_of_mass": com.astype(np.float32),
        "norm_center": center.astype(np.float64),
        "norm_scale": np.float64(scale),
        "original_extent": (bmax - bmin).astype(np.float64),
        "watertight": np.bool_(canon.is_watertight),
        "proxy_error": np.float32(proxy_err),    # surface deviation proxy vs original (canonical units)
        "coarse_error": np.float32(coarse_err),
    }
