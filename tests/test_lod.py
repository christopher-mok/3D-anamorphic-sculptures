"""Precise level-of-detail approximations and original-mesh substitution."""
import math

import numpy as np
import pytest
import torch
import trimesh

from sculpture.geometry.mesh_preprocess import adaptive_lod, point_triangle_distance, surface_deviation
from sculpture.scene.assembly import Assembly

from conftest import known_assembly, requires_cuda


def test_point_triangle_distance_exact():
    box = trimesh.creation.box((2, 2, 2))
    pts = np.array([[0, 0, 0], [3, 0, 0], [2, 2, 0], [0.5, 0.2, 1.0], [2, 2, 2]])
    d = point_triangle_distance(pts, box.vertices[box.faces])
    assert np.allclose(d, [1, 2, math.sqrt(2), 0, math.sqrt(3)], atol=1e-5)


def test_adaptive_lod_meets_tolerance():
    m = trimesh.creation.icosphere(subdivisions=5, radius=1.0)  # 20480 faces
    v, f, err = adaptive_lod(np.asarray(m.vertices), np.asarray(m.faces), tolerance=0.004, max_faces=20000)
    assert len(f) < len(m.faces) // 2           # actually simplified
    assert err <= 0.004
    assert abs(surface_deviation(v, f, m.vertices, m.faces) - err) < 2e-3


@requires_cuda
def test_chunked_union_render_matches(renderer, cameras, library):
    a = known_assembly(library)
    full = renderer.render_silhouettes(a, cameras, (128, 128))
    old = renderer.max_triangles
    try:
        renderer.max_triangles = 50  # forces one rasterize call per instance
        chunked = renderer.render_silhouettes(a, cameras, (128, 128))
        inst = renderer.render_instance_silhouettes(a, cameras, (64, 64))
    finally:
        renderer.max_triangles = old
    assert ((full > 0.5) != (chunked > 0.5)).float().mean() < 1e-3
    assert inst.shape == (3, 2, 64, 64)


@requires_cuda
def test_pick_lod_by_resolution(ctx, library):
    a = known_assembly(library)
    lods = [ctx.pick_lod(a, r) for r in (32, 4096)]
    assert lods[-1] == "proxy" and lods[0] in ("coarse", "proxy")


@requires_cuda
def test_reference_check_uses_original_geometry(tmp_path, cfg):
    """Two dense spheres: a 0.005-deep overlap is found, a 0.005 gap is not."""
    from sculpture.geometry.collisions import reference_intersections
    from sculpture.geometry.mesh_library import MeshLibrary

    trimesh.creation.icosphere(subdivisions=6, radius=1.0).export(tmp_path / "dense_sphere.ply")  # 81920 faces
    lib = MeshLibrary.from_folder(tmp_path, cfg.meshes, device="cuda", cache_dir=tmp_path / "cache")
    e = lib.entries[0]
    assert e.proxy_triangles < e.triangles and float(e.data["proxy_error"]) <= float(cfg.meshes.get("proxy_tolerance", 0.003)) + 1e-6
    lib.sdf(48)

    def pair(gap):
        s = 0.25
        t = torch.tensor([[-(s + gap / 2), 0.0, 0.0], [s + gap / 2, 0.0, 0.0]], device="cuda")
        return Assembly.from_matrices(torch.tensor([0, 0], device="cuda"), t, torch.eye(3, device="cuda").expand(2, 3, 3),
                                      torch.tensor([s, s], device="cuda"))

    assert reference_intersections(pair(-0.005), lib) == [(0, 1)]
    assert reference_intersections(pair(+0.005), lib) == []
