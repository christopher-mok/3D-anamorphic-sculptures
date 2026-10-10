"""REVEAL: reward the sculpture for NOT resembling the targets from other angles.

Off-axis cameras orbit each user camera's look-at point (rotations about the camera's up axis
and an elevated ring), skipping directions within ``min_separation_deg`` of any user camera.
For each off-axis render the similarity is the soft Dice coefficient with the most similar
target; the loss is its mean, weighted by ``reveal.weight`` in the shared refinement. Low
similarity = the image only "snaps together" from the intended viewpoint(s).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..camera.perspective_camera import PerspectiveCamera


def _rotate(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * (axis @ v) * (1 - math.cos(ang))


class RevealTerm:
    def __init__(self, ctx, cfg):
        self.ctx = ctx
        V0 = ctx.num_primary_views or ctx.num_views
        prim = ctx.cameras[:V0]
        dirs = [(c.eye - np.asarray(c.look_at)) / np.linalg.norm(c.eye - np.asarray(c.look_at)) for c in prim]
        min_sep = math.radians(float(cfg.get("min_separation_deg", 25.0)))
        cams = []
        for c in prim:
            la = np.asarray(c.look_at, dtype=np.float64)
            d0 = c.eye - la
            up = np.asarray(c.up, dtype=np.float64)
            right = np.cross(up, d0)
            for elev in cfg.get("elevations_deg", [0.0, 35.0]):
                d1 = _rotate(d0, right, math.radians(float(elev))) if abs(float(elev)) > 1e-6 else d0
                for ang in cfg.get("angles_deg", [45, 135, 180, 225, 315]):
                    d = _rotate(d1, up, math.radians(float(ang)))
                    du = d / np.linalg.norm(d)
                    if any(math.acos(float(np.clip(du @ p, -1, 1))) < min_sep for p in dirs):
                        continue
                    cams.append(PerspectiveCamera(position=tuple(la + d), look_at=tuple(la), up=tuple(c.up),
                                                  fov_y_deg=c.fov_y_deg, near=c.near, far=c.far))
        self.cameras = cams
        self.V0 = V0

    def similarity(self, a, res: int = 64) -> torch.Tensor:
        """Soft Dice of each off-axis silhouette with the most similar target: [O]."""
        if not self.cameras or len(a) == 0:
            return torch.zeros(max(len(self.cameras), 1), device=self.ctx.device)
        R = self.ctx.renderer.render_silhouettes(a, self.cameras, (res, res))       # [O,h,w]
        T = self.ctx.targets.soft(res)[: self.V0]                                    # [V0,h,w]
        inter = torch.einsum("ohw,vhw->ov", R, T)
        dice = 2 * inter / (R.sum((-1, -2))[:, None] + T.sum((-1, -2))[None] + 1e-6)
        return dice.amax(1)

    def loss(self, a, res: int = 64) -> torch.Tensor:
        return self.similarity(a, res).mean()
