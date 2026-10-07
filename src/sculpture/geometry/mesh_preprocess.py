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

    proxy_v, proxy_f = decimate(vn, f, int(cfg.get("proxy_max_faces", 8000)))
    coarse_v, coarse_f = decimate(vn, f, int(cfg.get("coarse_max_faces", 1500)))

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
    }
