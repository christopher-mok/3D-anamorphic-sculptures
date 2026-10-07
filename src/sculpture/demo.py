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
