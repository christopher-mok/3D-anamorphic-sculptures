"""Perspective visual hull  Omega = B  ∩  ⋂_v  π_v^{-1}(I_v)  on a voxel grid."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from ..camera.perspective_camera import PerspectiveCamera
from ..camera.projection import project_to_pixel_index
from ..targets.target import TargetSet
from .hull_sdf import occupancy_sdf

log = logging.getLogger(__name__)


class VisualHull:
    """Voxelized hull with SDF queries.

    occupancy: bool tensor [Z, Y, X] (voxel centers inside Omega)
    sdf:       float tensor [Z, Y, X]; phi <= 0 inside
    In RELAXED mode ``occupancy``/``sdf`` describe the dilated region used for
    containment while ``strict_occupancy`` keeps the exact intersection.
    """

    def __init__(self, strict_occupancy: np.ndarray, bmin, bmax, mode: str = "strict", dilation_voxels: int = 0, device="cuda"):
        self.device = torch.device(device)
        self.bmin = np.asarray(bmin, dtype=np.float64)
        self.bmax = np.asarray(bmax, dtype=np.float64)
        self.mode = mode
        self.resolution = strict_occupancy.shape[::-1]  # (X, Y, Z)
        self.voxel_size = (self.bmax - self.bmin) / np.asarray(self.resolution)
        occ = strict_occupancy.astype(bool)
        if mode == "relaxed" and dilation_voxels > 0:
            occ = ndimage.binary_dilation(occ, iterations=int(dilation_voxels))
        elif mode not in ("strict", "relaxed"):
            raise ValueError(f"unknown hull mode {mode!r}")
        self.strict_occupancy = torch.tensor(strict_occupancy.astype(bool), device=self.device)
        self.occupancy = torch.tensor(occ, device=self.device)
        self.sdf = torch.tensor(occupancy_sdf(occ, self.voxel_size), device=self.device)
        self._centers = None

    # ------------------------------------------------------------ construction
    @classmethod
    def carve(
        cls,
        targets: TargetSet,
        cameras: Sequence[PerspectiveCamera],
        bmin,
        bmax,
        resolution: int = 128,
        mode: str = "strict",
        dilation_voxels: int = 0,
        carve_resolution: int | None = None,
        device="cuda",
        isotropic: bool = True,
    ) -> "VisualHull":
        """Keep voxels whose centers project into the target foreground of every view."""
        bmin = np.asarray(bmin, np.float64)
        bmax = np.asarray(bmax, np.float64)
        ext = bmax - bmin
        if isotropic:  # cubic voxels: `resolution` cells along the longest axis
            vs = ext.max() / resolution
            res_xyz = np.maximum(1, np.round(ext / vs)).astype(int)
        else:  # `resolution` cells along every axis (elongated volumes, e.g. an unbounded view axis)
            res_xyz = np.full(3, int(resolution))
        carve_res = carve_resolution or targets.base_resolution
        masks = targets.mask(carve_res).bool()
        axes = [torch.linspace(bmin[i] + 0.5 * ext[i] / res_xyz[i], bmax[i] - 0.5 * ext[i] / res_xyz[i], int(res_xyz[i]), device=device) for i in range(3)]
        zz, yy = torch.meshgrid(axes[2], axes[1], indexing="ij")
        occ = torch.ones(int(res_xyz[2]), int(res_xyz[1]), int(res_xyz[0]), dtype=torch.bool, device=device)
        for zi in range(occ.shape[0]):  # slice-wise to bound memory at 256^3+
            pts = torch.stack(
                [axes[0][None, :].expand(occ.shape[1], -1), yy[zi][:, None].expand(-1, occ.shape[2]), zz[zi][:, None].expand(-1, occ.shape[2])],
                dim=-1,
            )
            inside = torch.ones(pts.shape[:2], dtype=torch.bool, device=device)
            for v, cam in enumerate(cameras):
                r, c, valid = project_to_pixel_index(pts, cam, (carve_res, carve_res))
                inside &= valid & masks[v][r, c]
            occ[zi] = inside
        occ_np = occ.cpu().numpy()
        if not occ_np.any():
            raise ValueError("visual hull is empty: the target silhouettes have no common back-projection inside the bounding volume")
        hull = cls(occ_np, bmin, bmax, mode, dilation_voxels, device)
        hull._views = (masks, list(cameras), carve_res)
        return hull

    # ------------------------------------------------------------ queries
    def _grid_coords(self, points: torch.Tensor):
        bmin = torch.tensor(self.bmin, dtype=points.dtype, device=points.device)
        bmax = torch.tensor(self.bmax, dtype=points.dtype, device=points.device)
        vs = torch.tensor(self.voxel_size, dtype=points.dtype, device=points.device)
        c0, c1 = bmin + 0.5 * vs, bmax - 0.5 * vs
        pc = torch.maximum(torch.minimum(points, c1), c0)
        g = (pc - c0) / (c1 - c0).clamp_min(1e-9) * 2.0 - 1.0
        return pc, g

    def query_sdf(self, points: torch.Tensor) -> torch.Tensor:
        """phi_Omega at world points [..., 3] (trilinear; linear extrapolation outside B)."""
        shape = points.shape[:-1]
        p = points.reshape(-1, 3)
        pc, g = self._grid_coords(p)
        val = F.grid_sample(self.sdf[None, None], g.view(1, 1, 1, -1, 3), mode="bilinear", padding_mode="border", align_corners=True).view(-1)
        return (val + (p - pc).norm(dim=-1)).view(shape)

    def query_occupancy(self, points: torch.Tensor, strict: bool = False) -> torch.Tensor:
        """Nearest-voxel occupancy at world points [..., 3]."""
        occ = self.strict_occupancy if strict else self.occupancy
        bmin = torch.tensor(self.bmin, dtype=points.dtype, device=points.device)
        vs = torch.tensor(self.voxel_size, dtype=points.dtype, device=points.device)
        idx = torch.floor((points - bmin) / vs).long()
        X, Y, Z = self.resolution
        valid = (idx[..., 0] >= 0) & (idx[..., 0] < X) & (idx[..., 1] >= 0) & (idx[..., 1] < Y) & (idx[..., 2] >= 0) & (idx[..., 2] < Z)
        ix, iy, iz = idx[..., 0].clamp(0, X - 1), idx[..., 1].clamp(0, Y - 1), idx[..., 2].clamp(0, Z - 1)
        return valid & occ[iz, iy, ix]

    def query_exact(self, points: torch.Tensor) -> torch.Tensor:
        """Continuous membership in the strict Omega: inside B and projecting into
        every target foreground (no voxelization). Falls back to voxels if the
        hull was loaded from disk."""
        views = getattr(self, "_views", None)
        if views is None:
            return self.query_occupancy(points, strict=True)
        masks, cams, res = views
        bmin = torch.tensor(self.bmin, dtype=points.dtype, device=points.device)
        bmax = torch.tensor(self.bmax, dtype=points.dtype, device=points.device)
        inside = ((points >= bmin) & (points <= bmax)).all(-1)
        for v, cam in enumerate(cams):
            r, c, valid = project_to_pixel_index(points, cam, (res, res))
            inside &= valid & masks[v][r, c]
        return inside

    def inside_centers(self, strict: bool = False, max_points: int | None = None, generator=None) -> torch.Tensor:
        """World coordinates of voxel centers inside the hull [K, 3]."""
        occ = self.strict_occupancy if strict else self.occupancy
        iz, iy, ix = torch.nonzero(occ, as_tuple=True)
        vs = torch.tensor(self.voxel_size, dtype=torch.float32, device=self.device)
        bmin = torch.tensor(self.bmin, dtype=torch.float32, device=self.device)
        pts = bmin + (torch.stack([ix, iy, iz], -1).float() + 0.5) * vs
        if max_points is not None and len(pts) > max_points:
            sel = torch.randperm(len(pts), generator=generator, device="cpu" if generator is None else generator.device)[:max_points]
            pts = pts[sel.to(self.device)]
        return pts

    @property
    def num_voxels(self) -> int:
        return int(self.strict_occupancy.sum().item())

    @property
    def mean_voxel_size(self) -> float:
        return float(np.mean(self.voxel_size))

    # ------------------------------------------------------------ IO
    def save(self, path) -> None:
        np.savez_compressed(
            path,
            strict_occupancy=self.strict_occupancy.cpu().numpy(),
            occupancy=self.occupancy.cpu().numpy(),
            sdf=self.sdf.cpu().numpy(),
            bmin=self.bmin,
            bmax=self.bmax,
            mode=self.mode,
        )

    @classmethod
    def load(cls, path, device="cuda") -> "VisualHull":
        with np.load(path) as z:
            h = cls.__new__(cls)
            h.device = torch.device(device)
            h.bmin, h.bmax, h.mode = z["bmin"], z["bmax"], str(z["mode"])
            so = z["strict_occupancy"]
            h.resolution = so.shape[::-1]
            h.voxel_size = (h.bmax - h.bmin) / np.asarray(h.resolution)
            h.strict_occupancy = torch.tensor(so, device=h.device)
            h.occupancy = torch.tensor(z["occupancy"], device=h.device)
            h.sdf = torch.tensor(z["sdf"], device=h.device)
            h._centers = None
            h._views = None
        return h

    def export_preview_obj(self, path, max_resolution: int = 64) -> int:
        """Write the voxel boundary as an OBJ mesh (downsampled). Returns face count."""
        occ = self.strict_occupancy.float()[None, None]
        f = int(np.ceil(max(occ.shape[2:]) / max_resolution))
        if f > 1:
            occ = F.max_pool3d(occ, kernel_size=f, stride=f, ceil_mode=True)
        occ = occ[0, 0].bool().cpu().numpy()
        Z, Y, X = occ.shape
        vs = (self.bmax - self.bmin) / np.array([X, Y, Z])
        verts, faces = voxel_boundary_mesh(occ)
        verts = self.bmin + verts * vs
        with open(path, "w") as fh:
            fh.write("# visual hull preview\n")
            for v in verts:
                fh.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            for q in faces:
                fh.write(f"f {q[0]+1} {q[1]+1} {q[2]+1}\nf {q[0]+1} {q[2]+1} {q[3]+1}\n")
        return 2 * len(faces)


def voxel_boundary_mesh(occ: np.ndarray):
    """Quads between occupied and empty voxels. occ is [Z, Y, X]; vertices in (x, y, z) voxel units."""
    p = np.pad(occ, 1)
    quads = []
    # unit cube corner offsets (x, y, z) for each face direction, CCW seen from outside
    face_defs = {
        (1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
        (-1, 0, 0): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        (0, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
        (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    }
    for (dx, dy, dz), corners in face_defs.items():
        nb = p[1 + dz : p.shape[0] - 1 + dz, 1 + dy : p.shape[1] - 1 + dy, 1 + dx : p.shape[2] - 1 + dx]
        z, y, x = np.nonzero(occ & ~nb)
        base = np.stack([x, y, z], -1)
        quads.append(base[:, None, :] + np.array(corners)[None])
    q = np.concatenate(quads).reshape(-1, 3)
    verts, inv = np.unique(q, axis=0, return_inverse=True)
    return verts.astype(np.float64), inv.reshape(-1, 4)
