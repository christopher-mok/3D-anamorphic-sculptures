"""Generate a small demo model pool and target silhouettes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def make_demo_models(folder) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meshes = {
        "cube.obj": trimesh.creation.box((1.0, 1.0, 1.0)),
        "sphere.ply": trimesh.creation.icosphere(subdivisions=3, radius=0.5),
        "cylinder.stl": trimesh.creation.cylinder(radius=0.25, height=1.2, sections=32),
        "cone.glb": trimesh.creation.cone(radius=0.45, height=1.0, sections=32),
        "torus.obj": trimesh.creation.torus(major_radius=0.5, minor_radius=0.16, major_sections=48, minor_sections=16),
        "rod.ply": trimesh.creation.box((1.6, 0.25, 0.25)),
        "plate.stl": trimesh.creation.box((1.0, 0.8, 0.12)),
    }
    out = []
    for name, m in meshes.items():
        p = folder / name
        m.export(p)
        out.append(p)
    return out


def _polygon_image(points: np.ndarray, size: int, path: Path) -> Path:
    img = Image.new("L", (size, size), 255)
    ImageDraw.Draw(img).polygon([tuple(p) for p in points], fill=0)
    img.save(path)
    return path


def heart_points(size: int, n: int = 200, extent=(0.12, 0.88)) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return _fit(np.stack([x, -y], 1), size, extent)


def star_points(size: int, k: int = 5, inner: float = 0.45, extent=(0.12, 0.88)) -> np.ndarray:
    pts = []
    for i in range(2 * k):
        r = 1.0 if i % 2 == 0 else inner
        a = math.pi / k * i - math.pi / 2
        pts.append((r * math.cos(a), r * math.sin(a)))
    return _fit(np.array(pts), size, extent)


def _fit(p: np.ndarray, size: int, extent) -> np.ndarray:
    """Scale/translate so the polygon spans rows extent[0]..extent[1] and is centered."""
    p = p - (p.min(0) + p.max(0)) / 2
    h = p[:, 1].max() - p[:, 1].min()
    scale = (extent[1] - extent[0]) * size / h
    p = p * scale
    p[:, 0] += size / 2
    p[:, 1] += (extent[0] + extent[1]) / 2 * size
    return p


def _displace(m: trimesh.Trimesh, amp: float, freq: float, seed: int) -> trimesh.Trimesh:
    """Smooth pseudo-random bumps along vertex normals (sum of random sinusoids)."""
    rng = np.random.default_rng(seed)
    v = m.vertices
    h = np.zeros(len(v))
    for _ in range(6):
        k = rng.normal(size=3) * freq
        h += np.sin(v @ k + rng.uniform(0, 2 * np.pi)) / 6
    return trimesh.Trimesh(v + m.vertex_normals * (amp * h)[:, None], m.faces, process=False)


def _tube(curve: np.ndarray, radius: float, sides: int) -> trimesh.Trimesh:
    """Closed tube along a closed polyline (parallel-transport-free Frenet frames)."""
    n = len(curve)
    t = np.roll(curve, -1, 0) - np.roll(curve, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    ref = np.array([0.0, 0.0, 1.0])
    nrm = np.cross(t, ref)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    bin_ = np.cross(t, nrm)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(ang)[None, :, None] * nrm[:, None] + np.sin(ang)[None, :, None] * bin_[:, None]
    verts = (curve[:, None] + radius * ring).reshape(-1, 3)
    i = np.arange(n)[:, None]
    j = np.arange(sides)[None, :]
    a = i * sides + j
    b = ((i + 1) % n) * sides + j
    c = ((i + 1) % n) * sides + (j + 1) % sides
    d = i * sides + (j + 1) % sides
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return trimesh.Trimesh(verts, faces, process=False)


def make_complex_models(folder) -> list[Path]:
    """High-resolution stress-test pool (~1M triangles total) for LOD / performance testing."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    s = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    knot = np.stack([(2 + np.cos(3 * s)) * np.cos(2 * s), (2 + np.cos(3 * s)) * np.sin(2 * s), np.sin(3 * s)], 1) * 0.3
    box = trimesh.creation.box((1.0, 0.6, 0.6))
    for _ in range(7):
        box = box.subdivide()
    meshes = {
        "bumpy_sphere.ply": _displace(trimesh.creation.icosphere(subdivisions=7, radius=0.5), 0.06, 9.0, 0),
        "dense_torus.ply": trimesh.creation.torus(major_radius=0.5, minor_radius=0.15, major_sections=512, minor_sections=256),
        "torus_knot.ply": _tube(knot, 0.08, 32),
        "rock.ply": _displace(trimesh.creation.icosphere(subdivisions=6, radius=0.5), 0.12, 4.0, 3),
        "dense_box.ply": _displace(box, 0.01, 20.0, 5),
    }
    out = []
    for name, m in meshes.items():
        p = folder / name
        m.export(p)
        out.append(p)
    return out


def make_demo_targets(folder, size: int = 512) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = [
        _polygon_image(heart_points(size), size, folder / "view_0.png"),
        _polygon_image(star_points(size), size, folder / "view_1.png"),
    ]
    # extra simple shapes, useful for single-view experiments
    img = Image.new("L", (size, size), 255)
    ImageDraw.Draw(img).ellipse([size * 0.2, size * 0.2, size * 0.8, size * 0.8], fill=0)
    img.save(folder / "circle.png")
    img = Image.new("L", (size, size), 255)
    ImageDraw.Draw(img).rectangle([size * 0.25, size * 0.25, size * 0.75, size * 0.75], fill=0)
    img.save(folder / "square.png")
    out += [folder / "circle.png", folder / "square.png"]
    return out


if __name__ == "__main__":
    make_demo_models("assets/models")
    make_demo_targets("assets/targets")
