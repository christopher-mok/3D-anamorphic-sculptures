"""Canonical-space signed distance fields for model meshes.

Sign: generalized winding number (robust to small holes / non-manifold input).
If the winding number is ambiguous for a large fraction of the grid (open
surfaces, sheets), fall back to flood-fill occupancy with an unsigned distance
shell. Distance magnitude: nearest-neighbour distance to a dense surface sample
set (KD-tree), which approximates point-to-triangle distance to within half the
sample spacing.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


def winding_numbers(points: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Generalized winding number of a triangle soup at query points (GPU, chunked)."""
    tri = verts[faces.long()]  # [F, 3, 3]
    out = torch.empty(points.shape[0], device=points.device)
    for s in range(0, points.shape[0], chunk):
        p = points[s : s + chunk]
        a = tri[None, :, 0, :] - p[:, None, :]
        b = tri[None, :, 1, :] - p[:, None, :]
        c = tri[None, :, 2, :] - p[:, None, :]
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        det = (a * torch.cross(b, c, dim=-1)).sum(-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        out[s : s + chunk] = torch.atan2(det, den).sum(-1) / (2.0 * math.pi)
    return out


def grid_points(res: int, lo: float, hi: float) -> np.ndarray:
    """Grid nodes in (z, y, x) memory order; returns [res^3, 3] xyz coordinates."""
    ax = np.linspace(lo, hi, res)
    zz, yy, xx = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def compute_mesh_sdf(
    vertices: np.ndarray,
    faces: np.ndarray,
    sign_vertices: np.ndarray,
    sign_faces: np.ndarray,
    resolution: int = 64,
    padding: float = 0.1,
    device: str = "cuda",
) -> dict:
    """SDF of a canonical (unit-radius) mesh on a cubic grid over [-1-pad, 1+pad]^3."""
    lo, hi = -1.0 - padding, 1.0 + padding
    pts = grid_points(resolution, lo, hi)
    voxel = (hi - lo) / (resolution - 1)

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    n_samples = int(min(400_000, max(50_000, 30 * resolution * resolution)))
    samples, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=0)
    tree = cKDTree(np.concatenate([samples, vertices]))
    udf, _ = tree.query(pts, workers=-1)
    udf = udf.astype(np.float32)

    with torch.no_grad():
        w = winding_numbers(
            torch.tensor(pts, dtype=torch.float32, device=device),
            torch.tensor(sign_vertices, dtype=torch.float32, device=device),
            torch.tensor(sign_faces, dtype=torch.int64, device=device),
        ).abs().cpu().numpy()

    far = udf > 2 * voxel
    ambiguous = np.mean((w[far] > 0.2) & (w[far] < 0.8)) if far.any() else 0.0
    method = "winding"
    if ambiguous > 0.05:
        # robust occupancy fallback: walls = voxels touching the surface; flood exterior
        method = "floodfill"
        wall = (udf < 0.75 * voxel).reshape(resolution, resolution, resolution)
        labels, _ = ndimage.label(~wall)
        border = np.unique(
            np.concatenate(
                [labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(), labels[:, -1].ravel(), labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]
            )
        )
        exterior = np.isin(labels, border[border > 0])
        inside = (~exterior).reshape(-1)
        sdf = np.where(inside, -udf, udf) - 0.5 * voxel  # thin parts become a half-voxel shell
    else:
        inside = w > 0.5
        sdf = np.where(inside, -udf, udf)

    return {
        "sdf": sdf.reshape(resolution, resolution, resolution).astype(np.float32),
        "lo": np.float32(lo),
        "hi": np.float32(hi),
        "method": method,
        "ambiguous_fraction": np.float32(ambiguous),
    }


class MeshSDFLibrary:
    """Stacked canonical SDF grids with batched trilinear queries."""

    def __init__(self, grids: list[np.ndarray], lo: float, hi: float, device):
        self.device = torch.device(device)
        self.grids = torch.stack([torch.from_numpy(g) for g in grids]).to(self.device)[:, None]  # [M,1,R,R,R]
        self.lo, self.hi = float(lo), float(hi)
        self.resolution = self.grids.shape[-1]

    def query_local(self, mesh_id: int, points: torch.Tensor) -> torch.Tensor:
        """Canonical-space SDF of mesh ``mesh_id`` at points [..., 3]."""
        shape = points.shape[:-1]
        p = points.reshape(-1, 3)
        pc = p.clamp(self.lo, self.hi)
        g = (pc - self.lo) / (self.hi - self.lo) * 2.0 - 1.0
        val = F.grid_sample(
            self.grids[mesh_id : mesh_id + 1], g.view(1, 1, 1, -1, 3), mode="bilinear", padding_mode="border", align_corners=True
        ).view(-1)
        val = val + (p - pc).norm(dim=-1)
        return val.view(shape)

    def query_instances(self, mesh_ids: torch.Tensor, R: torch.Tensor, s: torch.Tensor, t: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """World-space SDF of each instance i at its own query points.

        mesh_ids [N]; R [N,3,3]; s [N]; t [N,3]; points [N, P, 3] (world) -> [N, P]
        """
        local = torch.einsum("npi,nij->npj", points - t[:, None, :], R) / s[:, None, None]
        out = torch.empty(points.shape[:2], device=points.device, dtype=points.dtype)
        for m in torch.unique(mesh_ids).tolist():
            idx = (mesh_ids == m).nonzero(as_tuple=True)[0]
            out[idx] = self.query_local(m, local[idx]) * s[idx, None]
        return out
