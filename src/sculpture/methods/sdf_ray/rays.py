"""Camera ray sets for SDF ray packing (Method 3).

Foreground rays: one per target-foreground pixel, restricted to the segment
where the ray passes through Omega (rays that never meet Omega are
unsatisfiable under strict containment and are excluded and counted).
Background rays: target-background pixels, sampled across the bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...camera.projection import pixel_rays
from ...hull.compatibility import hull_ray_intervals


@dataclass
class RaySet:
    origins: torch.Tensor   # [R, 3]
    dirs: torch.Tensor      # [R, 3]
    t0: torch.Tensor        # [R]
    t1: torch.Tensor        # [R]
    view: torch.Tensor      # [R] view index
    pixel: torch.Tensor     # [R] flat pixel index (row * res + col)

    def __len__(self) -> int:
        return int(self.origins.shape[0])

    def subset(self, idx: torch.Tensor) -> "RaySet":
        return RaySet(self.origins[idx], self.dirs[idx], self.t0[idx], self.t1[idx], self.view[idx], self.pixel[idx])

    def sample_points(self, k: int, gen: torch.Generator | None = None, stratified: bool = True) -> torch.Tensor:
        """Stratified samples along each segment: [R, k, 3]."""
        n = len(self)
        base = (torch.arange(k, device=self.origins.device).float() + 0.5) / k
        if stratified:
            jitter = (torch.rand(n, k, device=self.origins.device, generator=gen) - 0.5) / k
            u = (base[None] + jitter).clamp(0, 1)
        else:
            u = base[None].expand(n, k)
        t = self.t0[:, None] + (self.t1 - self.t0)[:, None] * u
        return self.origins[:, None, :] + t[..., None] * self.dirs[:, None, :]


def build_ray_sets(ctx, resolution: int) -> tuple[RaySet, RaySet, dict]:
    """(foreground rays, background rays, statistics)."""
    dev = ctx.device
    pad = ctx.hull.mean_voxel_size
    masks = ctx.targets.mask(resolution) > 0.5
    fg_parts, bg_parts = [], []
    stats = {"fg_rays": 0, "fg_unsatisfiable": 0, "bg_rays": 0}
    for v, cam in enumerate(ctx.cameras):
        ri = hull_ray_intervals(ctx.hull, cam, resolution, strict=True)
        o, d = pixel_rays(cam, (resolution, resolution), dev)
        o, d = o.reshape(-1, 3), d.reshape(-1, 3)
        m = masks[v].reshape(-1)
        hit = ri.hit.reshape(-1)
        fg = m & hit
        stats["fg_rays"] += int(fg.sum())
        stats["fg_unsatisfiable"] += int((m & ~hit).sum())
        idx = fg.nonzero(as_tuple=True)[0]
        fg_parts.append(RaySet(o[idx], d[idx], ri.t_enter.reshape(-1)[idx] - pad, ri.t_exit.reshape(-1)[idx] + pad,
                               torch.full_like(idx, v), idx))
        box = ri.t_box_far.reshape(-1) > ri.t_box_near.reshape(-1)
        bidx = (~m & box).nonzero(as_tuple=True)[0]
        stats["bg_rays"] += len(bidx)
        bg_parts.append(RaySet(o[bidx], d[bidx], ri.t_box_near.reshape(-1)[bidx], ri.t_box_far.reshape(-1)[bidx],
                               torch.full_like(bidx, v), bidx))

    def cat(parts):
        return RaySet(*[torch.cat([getattr(p, f) for p in parts]) for f in ("origins", "dirs", "t0", "t1", "view", "pixel")])

    return cat(fg_parts), cat(bg_parts), stats
