"""nvdiffrast production backend. nvdiffrast objects never leave this module."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..camera.perspective_camera import PerspectiveCamera
from ..geometry.mesh_library import MeshLibrary
from ..scene.assembly import Assembly
from .base import InstancesLike, RenderStats

try:
    import nvdiffrast.torch as dr
except ImportError as exc:  # pragma: no cover
    dr = None
    _IMPORT_ERROR = exc


class NvdiffrastRenderer:
    """Batched perspective silhouette rasterizer with analytic antialiasing.

    * ``render_silhouettes`` concatenates every instance into one triangle soup
      and rasterizes all views in one batched call (z-buffered union).
    * ``render_instance_silhouettes`` uses nvdiffrast's instanced mode: one
      call per distinct mesh, batch = instances x views, chunked by memory.
    Gradients w.r.t. transformed vertices come from ``dr.antialias``.
    """

    def __init__(self, library: MeshLibrary, device="cuda", max_batch_pixels: int = 1 << 24, max_triangles: int = 1_000_000):
        if dr is None:  # pragma: no cover
            raise ImportError(f"nvdiffrast is required for NvdiffrastRenderer: {_IMPORT_ERROR}")
        self.library = library
        self.device = torch.device(device)
        self.ctx = dr.RasterizeCudaContext(device=self.device)
        self.max_batch_pixels = max_batch_pixels
        # the CUDA rasterizer overflows ("subtriangle count overflow" / illegal access) beyond a few
        # million triangles per call: larger scenes (e.g. the original meshes) are rendered in
        # chunks and composited with max (exact for binary silhouettes)
        self.max_triangles = int(max_triangles)
        self.stats = RenderStats()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _as_assembly(instances: InstancesLike, device) -> Assembly:
        if isinstance(instances, Assembly):
            return instances
        return Assembly.from_instances(list(instances), device=device)

    def _mvps(self, cameras: Sequence[PerspectiveCamera], resolution) -> torch.Tensor:
        H, W = resolution
        return torch.stack([c.torch_mvp(self.device, aspect=W / H) for c in cameras])  # [V,4,4]

    @staticmethod
    def _check_res(resolution):
        H, W = int(resolution[0]), int(resolution[1])
        if H % 8 or W % 8:
            raise ValueError(f"nvdiffrast CUDA rasterizer needs resolutions divisible by 8, got {resolution}")
        return H, W

    def _transformed(self, assembly: Assembly, idx: torch.Tensor, mesh_id: int, lod: str):
        verts, faces = self.library.proxy(mesh_id, lod)
        R = assembly.rotation_matrices()[idx]
        s = torch.exp(assembly.log_scale[idx])
        t = assembly.translation[idx]
        world = torch.einsum("vj,kij->kvi", verts, R) * s[:, None, None] + t[:, None, :]  # [k,Nv,3]
        return world, faces

    # ------------------------------------------------------------ API
    def render_silhouettes(self, assembly: Assembly, cameras, resolution, lod: str = "proxy") -> torch.Tensor:
        H, W = self._check_res(resolution)
        V = len(cameras)
        self.stats.record("union", V, V * H * W)
        if len(assembly) == 0:
            return torch.zeros(V, H, W, device=self.device)
        tri_counts = torch.tensor([len(self.library.proxy(m, lod)[1]) for m in range(len(self.library))], device=self.device)
        per_inst = tri_counts[assembly.mesh_ids]
        if int(per_inst.sum()) > self.max_triangles and len(assembly) > 1:
            groups, cur, acc = [], [], 0
            for i, n in enumerate(per_inst.tolist()):
                if cur and acc + n > self.max_triangles:
                    groups.append(cur)
                    cur, acc = [], 0
                cur.append(i)
                acc += n
            groups.append(cur)
            out = None
            for g in groups:
                m = self._render_union(assembly[g], cameras, (H, W), lod)
                out = m if out is None else torch.maximum(out, m)
            return out
        return self._render_union(assembly, cameras, (H, W), lod)

    def _render_union(self, assembly: Assembly, cameras, resolution, lod: str) -> torch.Tensor:
        H, W = resolution
        all_pos, all_tri, offset = [], [], 0
        for m in torch.unique(assembly.mesh_ids).tolist():
            idx = (assembly.mesh_ids == m).nonzero(as_tuple=True)[0]
            world, faces = self._transformed(assembly, idx, m, lod)
            k, nv = world.shape[:2]
            all_pos.append(world.reshape(-1, 3))
            tri = faces[None].long() + (torch.arange(k, device=self.device) * nv)[:, None, None] + offset
            all_tri.append(tri.reshape(-1, 3))
            offset += k * nv
        pos = torch.cat(all_pos)
        tri = torch.cat(all_tri).int().contiguous()
        pos_h = torch.cat([pos, torch.ones_like(pos[:, :1])], dim=-1)
        clip = torch.einsum("ni,vji->vnj", pos_h, self._mvps(cameras, (H, W))).contiguous()  # [V,N,4]
        rast, _ = dr.rasterize(self.ctx, clip, tri, (H, W))
        mask = (rast[..., 3:] > 0).float()
        mask = dr.antialias(mask, rast, clip, tri)
        return mask[..., 0].flip(1)

    def render_instance_silhouettes(self, instances: InstancesLike, cameras, resolution, lod: str = "proxy") -> torch.Tensor:
        H, W = self._check_res(resolution)
        a = self._as_assembly(instances, self.device)
        V, N = len(cameras), len(a)
        self.stats.record("instances", N * V, N * V * H * W)
        if N == 0:
            return torch.zeros(0, V, H, W, device=self.device)
        mvps = self._mvps(cameras, (H, W))
        pix_chunk = max(1, self.max_batch_pixels // (V * H * W))
        pieces, order = [], []
        for m in torch.unique(a.mesh_ids).tolist():
            idx_all = (a.mesh_ids == m).nonzero(as_tuple=True)[0]
            n_tri = len(self.library.proxy(m, lod)[1])
            chunk = max(1, min(pix_chunk, self.max_triangles // max(1, n_tri * V)))
            s0 = 0
            while s0 < len(idx_all):
                idx = idx_all[s0 : s0 + chunk]
                try:
                    pieces.append(self._instances_chunk(a, idx, m, lod, mvps, V, H, W))
                except RuntimeError as exc:  # rasterizer buffer allocation failed: halve the batch
                    if chunk > 1 and ("cudaMalloc" in str(exc) or "out of memory" in str(exc)):
                        torch.cuda.empty_cache()
                        chunk = max(1, chunk // 2)
                        continue
                    raise
                order.append(idx)
                s0 += len(idx)
        out = torch.cat(pieces)
        inv = torch.empty(N, dtype=torch.long, device=self.device)
        inv[torch.cat(order)] = torch.arange(N, device=self.device)
        return out[inv]

    def _instances_chunk(self, a: Assembly, idx, m: int, lod: str, mvps, V: int, H: int, W: int) -> torch.Tensor:
        world, faces = self._transformed(a, idx, m, lod)  # [k,Nv,3]
        k = world.shape[0]
        pos_h = torch.cat([world, torch.ones_like(world[..., :1])], dim=-1)
        clip = torch.einsum("kni,vji->kvnj", pos_h, mvps).reshape(k * V, -1, 4).contiguous()
        rast, _ = dr.rasterize(self.ctx, clip, faces, (H, W))
        mask = (rast[..., 3:] > 0).float()
        mask = dr.antialias(mask, rast, clip, faces)
        return mask[..., 0].flip(1).reshape(k, V, H, W)

    @torch.no_grad()
    def render_instance_ids(self, assembly: Assembly, cameras, resolution, lod: str = "proxy") -> torch.Tensor:
        """Rasterize the union once and map each winning triangle back to its instance."""
        H, W = self._check_res(resolution)
        V = len(cameras)
        self.stats.record("instance_ids", V, V * H * W)
        if len(assembly) == 0:
            return torch.full((V, H, W), -1, dtype=torch.long, device=self.device)
        all_pos, all_tri, owners, offset = [], [], [], 0
        for m in torch.unique(assembly.mesh_ids).tolist():
            idx = (assembly.mesh_ids == m).nonzero(as_tuple=True)[0]
            world, faces = self._transformed(assembly, idx, m, lod)
            k, nv = world.shape[:2]
            all_pos.append(world.reshape(-1, 3))
            tri = faces[None].long() + (torch.arange(k, device=self.device) * nv)[:, None, None] + offset
            all_tri.append(tri.reshape(-1, 3))
            owners.append(idx.repeat_interleave(len(faces)))
            offset += k * nv
        pos, tri, owner = torch.cat(all_pos), torch.cat(all_tri).int().contiguous(), torch.cat(owners)
        pos_h = torch.cat([pos, torch.ones_like(pos[:, :1])], -1)
        clip = torch.einsum("ni,vji->vnj", pos_h, self._mvps(cameras, (H, W))).contiguous()
        rast, _ = dr.rasterize(self.ctx, clip, tri, (H, W))
        fid = rast[..., 3].long() - 1
        out = torch.full_like(fid, -1)
        hit = fid >= 0
        out[hit] = owner[fid[hit]]
        return out.flip(1)

    def render_rgba(self, assembly: Assembly, cameras, resolution, lod: str = "proxy") -> torch.Tensor:
        """Depth-tested, antialiased flat instance colors with geometry/color gradients.

        RGB is premultiplied by coverage; uncovered pixels have zero alpha.
        Unlike silhouette unions, this preserves the color of the visible front piece.
        """
        H, W = self._check_res(resolution)
        self.stats.record("rgba", len(cameras), len(cameras) * H * W)
        if len(assembly) == 0:
            return torch.zeros(len(cameras), H, W, 4, device=self.device)
        positions, triangles, attributes, offset = [], [], [], 0
        for m in torch.unique(assembly.mesh_ids).tolist():
            idx = (assembly.mesh_ids == m).nonzero(as_tuple=True)[0]
            world, faces = self._transformed(assembly, idx, m, lod)
            k, nv = world.shape[:2]
            positions.append(world.reshape(-1, 3))
            triangles.append((faces[None].long() + (torch.arange(k, device=self.device) * nv)[:, None, None]
                              + offset).reshape(-1, 3))
            colors = (assembly.colors[idx] if assembly.colors is not None
                      else torch.full((k, 3), 0.65, device=self.device))
            attr = torch.cat([colors, torch.ones(k, 1, device=self.device)], -1)
            attributes.append(attr[:, None].expand(k, nv, 4).reshape(-1, 4))
            offset += k * nv
        pos = torch.cat(positions)
        tri = torch.cat(triangles).int().contiguous()
        pos_h = torch.cat([pos, torch.ones_like(pos[:, :1])], -1)
        clip = torch.einsum("ni,vji->vnj", pos_h, self._mvps(cameras, (H, W))).contiguous()
        rast, _ = dr.rasterize(self.ctx, clip, tri, (H, W))
        rgba, _ = dr.interpolate(torch.cat(attributes)[None].contiguous(), rast, tri)
        return dr.antialias(rgba, rast, clip, tri).flip(1)

    # ------------------------------------------------------------ previews (server only)
    @torch.no_grad()
    def render_mesh_preview(self, mesh_id: int, resolution: int = 128) -> np.ndarray:
        """Normal-shaded RGBA uint8 thumbnail of a canonical mesh (3/4 view)."""
        verts, faces = self.library.proxy(mesh_id, "proxy")
        cam = PerspectiveCamera(position=(2.2, 1.6, 2.8), look_at=(0, 0, 0), fov_y_deg=40, near=0.1, far=20)
        mvp = cam.torch_mvp(self.device)
        pos = torch.cat([verts, torch.ones_like(verts[:, :1])], -1) @ mvp.T
        rast, _ = dr.rasterize(self.ctx, pos[None].contiguous(), faces, (resolution, resolution))
        tri = verts[faces.long()]
        fn = F.normalize(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1)
        fid = rast[0, ..., 3].long()
        n = fn[(fid - 1).clamp_min(0)]
        light = F.normalize(torch.tensor([0.4, 0.8, 0.6], device=self.device), dim=0)
        shade = (n @ light).abs() * 0.75 + 0.25
        alpha = (fid > 0).float()
        rgb = torch.stack([shade * 0.85, shade * 0.9, shade * 1.0], -1) * alpha[..., None]
        img = torch.cat([rgb, alpha[..., None]], -1).flip(0)
        return (img.clamp(0, 1) * 255).byte().cpu().numpy()
