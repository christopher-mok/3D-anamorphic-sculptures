"""Exact perspective projection and camera rays (consistent with the renderer)."""

from __future__ import annotations

import math

import torch

from .perspective_camera import PerspectiveCamera


def project_points(points: torch.Tensor, camera: PerspectiveCamera, resolution: tuple[int, int]):
    """Project world points [..., 3].

    Returns (uv, depth, in_front):
      uv[..., 0] = column, uv[..., 1] = row, continuous, pixel centers at integers;
      depth = distance along the viewing axis (positive in front of the camera).
    """
    H, W = resolution
    mvp = camera.torch_mvp(points.device, aspect=W / H)
    ph = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    clip = ph @ mvp.T
    w = clip[..., 3]
    in_front = w > camera.near * 0.999
    w_safe = torch.where(in_front, w, torch.ones_like(w))
    x_ndc = clip[..., 0] / w_safe
    y_ndc = clip[..., 1] / w_safe
    u = (x_ndc + 1.0) * 0.5 * W - 0.5
    v = (1.0 - y_ndc) * 0.5 * H - 0.5
    return torch.stack([u, v], dim=-1), w, in_front


def project_to_pixel_index(points: torch.Tensor, camera: PerspectiveCamera, resolution: tuple[int, int]):
    """Nearest pixel (row, col) indices plus validity (inside image and in front)."""
    H, W = resolution
    uv, depth, front = project_points(points, camera, resolution)
    col = torch.round(uv[..., 0]).long()
    row = torch.round(uv[..., 1]).long()
    valid = front & (col >= 0) & (col < W) & (row >= 0) & (row < H) & (depth < camera.far)
    return row.clamp(0, H - 1), col.clamp(0, W - 1), valid


def pixel_rays(camera: PerspectiveCamera, resolution: tuple[int, int], device, rows=None, cols=None):
    """World-space rays through pixel centers.

    If rows/cols (LongTensors) are given, returns rays for those pixels only.
    Returns (origins [..., 3], unit directions [..., 3]).
    """
    H, W = resolution
    if rows is None:
        rows, cols = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    rows = rows.to(device).float()
    cols = cols.to(device).float()
    t = math.tan(math.radians(camera.fov_y_deg) / 2.0)
    aspect = W / H
    x_ndc = (cols + 0.5) / W * 2.0 - 1.0
    y_ndc = 1.0 - (rows + 0.5) / H * 2.0
    d_cam = torch.stack([x_ndc * t * aspect, y_ndc * t, -torch.ones_like(x_ndc)], dim=-1)
    Rwc = camera.torch_rotation(device)  # world -> camera
    d_world = d_cam @ Rwc  # == (Rwc^T d_cam^T)^T
    d_world = d_world / d_world.norm(dim=-1, keepdim=True)
    origins = camera.torch_eye(device).expand_as(d_world)
    return origins, d_world


def ray_box_intersection(origins: torch.Tensor, dirs: torch.Tensor, bmin, bmax):
    """Slab test. Returns (t_near, t_far, hit) with t >= 0."""
    bmin = torch.as_tensor(bmin, dtype=origins.dtype, device=origins.device)
    bmax = torch.as_tensor(bmax, dtype=origins.dtype, device=origins.device)
    inv = 1.0 / torch.where(dirs.abs() < 1e-12, torch.full_like(dirs, 1e-12), dirs)
    t0 = (bmin - origins) * inv
    t1 = (bmax - origins) * inv
    tmin = torch.minimum(t0, t1).amax(dim=-1).clamp_min(0.0)
    tmax = torch.maximum(t0, t1).amin(dim=-1)
    return tmin, tmax, tmax > tmin
