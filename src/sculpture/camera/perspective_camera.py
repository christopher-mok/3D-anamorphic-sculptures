"""Pinhole perspective camera (OpenGL conventions: camera looks down -Z, +Y up).

Image convention used everywhere in the code base: row 0 is the TOP of the
image, column 0 the LEFT, pixel (row i, col j) has its center at
NDC ( (j + 0.5) / W * 2 - 1 ,  1 - (i + 0.5) / H * 2 ).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np
import torch


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("degenerate vector in camera definition")
    return v / n


@dataclass
class PerspectiveCamera:
    position: Sequence[float] = (0.0, 0.0, 5.0)
    look_at: Sequence[float] = (0.0, 0.0, 0.0)
    up: Sequence[float] = (0.0, 1.0, 0.0)
    fov_y_deg: float = 25.0
    near: float = 0.1
    far: float = 20.0
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, d: Mapping) -> "PerspectiveCamera":
        return cls(
            position=tuple(float(x) for x in d.get("position", (0, 0, 5))),
            look_at=tuple(float(x) for x in d.get("look_at", (0, 0, 0))),
            up=tuple(float(x) for x in d.get("up", (0, 1, 0))),
            fov_y_deg=float(d.get("fov_y_deg", 25.0)),
            near=float(d.get("near", 0.1)),
            far=float(d.get("far", 20.0)),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_cache", None)
        d["position"] = [float(x) for x in self.position]
        d["look_at"] = [float(x) for x in self.look_at]
        d["up"] = [float(x) for x in self.up]
        return d

    # ------------------------------------------------------------ matrices
    @property
    def eye(self) -> np.ndarray:
        return np.asarray(self.position, dtype=np.float64)

    def rotation(self) -> np.ndarray:
        """World-to-camera rotation (rows are camera right, up, back axes)."""
        eye = self.eye
        fwd = _normalize(np.asarray(self.look_at, dtype=np.float64) - eye)
        up = np.asarray(self.up, dtype=np.float64)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-8:  # up parallel to view direction
            right = np.cross(fwd, np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.9 else np.array([1.0, 0.0, 0.0]))
        right = _normalize(right)
        true_up = np.cross(right, fwd)
        return np.stack([right, true_up, -fwd], axis=0)

    def view_matrix(self) -> np.ndarray:
        R = self.rotation()
        V = np.eye(4)
        V[:3, :3] = R
        V[:3, 3] = -R @ self.eye
        return V

    def projection_matrix(self, aspect: float = 1.0) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(self.fov_y_deg) / 2.0)
        n, fa = self.near, self.far
        P = np.zeros((4, 4))
        P[0, 0] = f / aspect
        P[1, 1] = f
        P[2, 2] = (fa + n) / (n - fa)
        P[2, 3] = 2 * fa * n / (n - fa)
        P[3, 2] = -1.0
        return P

    def mvp(self, aspect: float = 1.0) -> np.ndarray:
        return self.projection_matrix(aspect) @ self.view_matrix()

    def focal_px(self, height: int) -> float:
        """Focal length in pixels for an image of the given height."""
        return 0.5 * height / math.tan(math.radians(self.fov_y_deg) / 2.0)

    def forward(self) -> np.ndarray:
        return -self.rotation()[2]

    # ------------------------------------------------------------ torch helpers
    def torch_mvp(self, device, aspect: float = 1.0) -> torch.Tensor:
        key = ("mvp", str(device), aspect, self._signature())
        if key not in self._cache:
            self._cache[key] = torch.tensor(self.mvp(aspect), dtype=torch.float32, device=device)
        return self._cache[key]

    def torch_rotation(self, device) -> torch.Tensor:
        key = ("rot", str(device), self._signature())
        if key not in self._cache:
            self._cache[key] = torch.tensor(self.rotation(), dtype=torch.float32, device=device)
        return self._cache[key]

    def torch_eye(self, device) -> torch.Tensor:
        return torch.tensor(self.eye, dtype=torch.float32, device=device)

    def _signature(self):
        return (tuple(self.position), tuple(self.look_at), tuple(self.up), self.fov_y_deg, self.near, self.far)


def cameras_from_config(cam_cfgs, n: int | None = None) -> list[PerspectiveCamera]:
    cams = [PerspectiveCamera.from_dict(c) for c in cam_cfgs]
    if n is not None:
        if len(cams) < n:
            raise ValueError(f"{n} targets but only {len(cams)} cameras configured")
        cams = cams[:n]
    return cams
