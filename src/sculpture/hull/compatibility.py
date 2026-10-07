"""Target-pair compatibility: projection of Omega back into each view."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from ..camera.perspective_camera import PerspectiveCamera
from ..camera.projection import pixel_rays, ray_box_intersection
from ..targets.target import TargetSet
from .visual_hull import VisualHull


@dataclass
class RayIntervals:
    """Per-pixel intersection of camera rays with Omega for one view."""

    hit: torch.Tensor      # [H, W] bool: ray meets Omega
    t_enter: torch.Tensor  # [H, W] first sample inside Omega (or box entry if no hit)
    t_exit: torch.Tensor   # [H, W] last sample inside Omega
    t_box_near: torch.Tensor
    t_box_far: torch.Tensor
    resolution: int


@torch.no_grad()
def hull_ray_intervals(hull: VisualHull, camera: PerspectiveCamera, resolution: int, strict: bool = True, chunk: int = 8192) -> RayIntervals:
    H = W = resolution
    dev = hull.device
    origins, dirs = pixel_rays(camera, (H, W), dev)
    o, d = origins.reshape(-1, 3), dirs.reshape(-1, 3)
    tn, tf, box_hit = ray_box_intersection(o, d, hull.bmin, hull.bmax)
    step = 0.5 * float(np.min(hull.voxel_size))
    n_steps = int(np.ceil(np.linalg.norm(hull.bmax - hull.bmin) / step)) + 1
    hit = torch.zeros(len(o), dtype=torch.bool, device=dev)
    t_in = tn.clone()
    t_out = tn.clone()
    for s in range(0, len(o), chunk):
        sl = slice(s, s + chunk)
        frac = torch.linspace(0, 1, n_steps, device=dev)
        ts = tn[sl, None] + (tf[sl] - tn[sl]).clamp_min(0)[:, None] * frac[None]
        pts = o[sl, None, :] + ts[..., None] * d[sl, None, :]
        inside = (hull.query_exact(pts) if strict else hull.query_occupancy(pts)) & box_hit[sl, None]
        any_in = inside.any(-1)
        first = torch.argmax(inside.int(), dim=-1)
        last = n_steps - 1 - torch.argmax(inside.flip(-1).int(), dim=-1)
        rows = torch.arange(len(ts), device=dev)
        hit[sl] = any_in
        t_in[sl] = torch.where(any_in, ts[rows, first], tn[sl])
        t_out[sl] = torch.where(any_in, ts[rows, last], tn[sl])
    return RayIntervals(hit.view(H, W), t_in.view(H, W), t_out.view(H, W), tn.view(H, W), tf.view(H, W), resolution)


@dataclass
class CompatibilityReport:
    strict_coverage_upper_bound: list[float]
    projected_outside_target: list[float]   # |Q_v \ I_v| / |Q_v| (should be ~0)
    hull_voxels: int
    warnings: list[str] = field(default_factory=list)
    resolution: int = 0

    def to_dict(self) -> dict:
        return {
            "strict_coverage_upper_bound": self.strict_coverage_upper_bound,
            "projected_outside_target": self.projected_outside_target,
            "hull_voxels": self.hull_voxels,
            "warnings": self.warnings,
            "resolution": self.resolution,
        }


def compatibility_report(
    hull: VisualHull,
    targets: TargetSet,
    cameras: Sequence[PerspectiveCamera],
    resolution: int,
    warn_threshold: float = 0.9,
) -> tuple[CompatibilityReport, list[RayIntervals]]:
    """strict_coverage_upper_bound_v = |Q_v ∩ I_v| / |I_v| with Q_v the projection of Omega."""
    bounds, outside, warnings, intervals = [], [], [], []
    masks = targets.mask(resolution).bool()
    for v, cam in enumerate(cameras):
        ri = hull_ray_intervals(hull, cam, resolution, strict=True)
        intervals.append(ri)
        Q, I = ri.hit, masks[v]
        b = (Q & I).sum().item() / max(I.sum().item(), 1)
        out = (Q & ~I).sum().item() / max(Q.sum().item(), 1)
        bounds.append(float(b))
        outside.append(float(out))
        if b < warn_threshold:
            warnings.append(
                f"view {v}: strict coverage upper bound is {b:.3f} < {warn_threshold:.2f}. The target images are "
                "geometrically inconsistent under strict containment for these cameras (some target pixels have no "
                "3D point that also projects inside the other target). Consider RELAXED mode, different cameras, "
                "or adjusting the targets."
            )
    return CompatibilityReport(bounds, outside, hull.num_voxels, warnings, resolution), intervals
