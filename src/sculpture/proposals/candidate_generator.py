"""Target-informed candidate placements c = (mesh_id, t, R, s).

Shared by all methods (beam proposals, initial / pricing columns, SDF-ray
initialization and reseeding). Each generated candidate is an independent
object; the result is returned as an Assembly whose rows are candidates.

POSITION     sampled from Omega's voxel centers, weighted by how much of the
             candidate's projection lands in uncovered target regions (with a
             small exploration probability of unweighted / box sampling).
MODEL+ORIENT the utility map around the projected position is matched against
             the silhouette bank of every mesh (discrete argmax / top-k sample).
SCALE        projected radius ≈ f * s * rho / z, initialized from the distance
             transform of the uncovered region (inscribed radius).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..camera.projection import project_points, ray_box_intersection
from ..geometry.rotation import matrix_to_rot6d, random_rotations
from ..scene.assembly import Assembly
from .residual import inscribed_radius_map
from .silhouette_bank import FRAME


@dataclass
class GeneratorOptions:
    explore_probability: float = 0.1
    bank_probability: float = 0.8
    scale_jitter: tuple[float, float] = (0.8, 1.5)
    top_k: int = 4
    allowed_meshes: torch.Tensor | None = None


class CandidateGenerator:
    def __init__(self, ctx, max_points: int = 200_000):
        self.ctx = ctx
        self.device = ctx.device
        g = torch.Generator(device="cpu").manual_seed(int(ctx.cfg.seed))
        self.points = ctx.hull.inside_centers(strict=ctx.strict, max_points=max_points, generator=g)  # [P,3]
        self.voxel = torch.tensor(ctx.hull.voxel_size, dtype=torch.float32, device=self.device)
        self.bmin = torch.tensor(ctx.hull.bmin, dtype=torch.float32, device=self.device)
        self.bmax = torch.tensor(ctx.hull.bmax, dtype=torch.float32, device=self.device)
        # normalized projections of all hull points (u/W, v/H in [0,1]) and depths
        uv, depth = [], []
        for cam in ctx.cameras:
            p, z, _ = project_points(self.points, cam, (1, 1))
            uv.append(p + 0.5)  # with W=H=1, pixel coordinate + 0.5 is the normalized position
            depth.append(z)
        self.uv = torch.stack(uv)        # [V,P,2]
        self.depth = torch.stack(depth)  # [V,P]
        self.phi = ctx.hull.query_sdf(self.points)  # inscribed sphere radius = -phi

    # ------------------------------------------------------------ helpers
    def _pixel_lookup(self, maps: torch.Tensor, uv_norm: torch.Tensor) -> torch.Tensor:
        """maps [V,H,W], uv_norm [V,P,2] -> [V,P] nearest-pixel values."""
        V, H, W = maps.shape
        col = (uv_norm[..., 0] * W).long().clamp(0, W - 1)
        row = (uv_norm[..., 1] * H).long().clamp(0, H - 1)
        return torch.stack([maps[v][row[v], col[v]] for v in range(V)])

    def sample_positions(self, n: int, U: torch.Tensor, gen: torch.Generator, explore_p: float) -> torch.Tensor:
        u = self._pixel_lookup(U, self.uv).clamp_min(0)  # [V,P]
        w = u.mean(0) + 2.0 * u.prod(0) + 1e-4
        idx = torch.multinomial(w, n, replacement=True, generator=gen)
        pos = self.points[idx] + (torch.rand(n, 3, device=self.device, generator=gen) - 0.5) * self.voxel
        n_exp = int(round(explore_p * n))
        if n_exp > 0:
            # exploration: unweighted hull points with a wider jitter, and some anywhere in B
            idx2 = torch.randint(0, len(self.points), (n_exp,), device=self.device, generator=gen)
            p2 = self.points[idx2] + (torch.rand(n_exp, 3, device=self.device, generator=gen) - 0.5) * 4 * self.voxel
            n_box = n_exp // 4
            if n_box:
                p2[:n_box] = self.bmin + torch.rand(n_box, 3, device=self.device, generator=gen) * (self.bmax - self.bmin)
            pos[-n_exp:] = p2
        return pos

    def _patches(self, W: torch.Tensor, view: torch.Tensor, uv_px: torch.Tensor, half_px: torch.Tensor, d: int) -> torch.Tensor:
        """Sample utility patches on the bank frame: -> [n, d, d]."""
        V, H, Wd = W.shape
        n = uv_px.shape[0]
        lin = ((torch.arange(d, device=self.device) + 0.5) / d * 2 - 1)  # [-1,1] cell centers
        oy, ox = torch.meshgrid(lin, lin, indexing="ij")
        cols = uv_px[:, 0, None, None] + ox[None] * half_px[:, None, None]
        rows = uv_px[:, 1, None, None] + oy[None] * half_px[:, None, None]
        grid = torch.stack([(cols + 0.5) / Wd * 2 - 1, (rows + 0.5) / H * 2 - 1], -1)  # [n,d,d,2]
        out = torch.zeros(n, d, d, device=self.device)
        for v in range(V):
            sel = (view == v).nonzero(as_tuple=True)[0]
            if len(sel):
                out[sel] = F.grid_sample(W[v][None, None].expand(len(sel), 1, H, Wd), grid[sel], align_corners=False, padding_mode="zeros")[:, 0]
        return out

    def _slide_to_depth(self, pos: torch.Tensor, view: torch.Tensor, z: torch.Tensor, z_des: torch.Tensor):
        """Move points along their reference camera's ray to depth z_des (clamped to the
        hull's box). The projection in that view is unchanged."""
        new_pos, new_z = pos.clone(), z.clone()
        for v, cam in enumerate(self.ctx.cameras):
            sel = (view == v).nonzero(as_tuple=True)[0]
            if len(sel) == 0:
                continue
            eye = cam.torch_eye(self.device)
            d = pos[sel] - eye
            dist = d.norm(dim=-1).clamp_min(1e-6)
            dirs = d / dist[:, None]
            tn, tf, hit = ray_box_intersection(eye.expand_as(dirs), dirs, self.bmin, self.bmax)
            t_des = dist * z_des[sel] / z[sel]
            margin = 0.02 * (tf - tn).clamp_min(0)
            t_new = torch.where(hit, torch.minimum(torch.maximum(t_des, tn + margin), tf - margin), dist)
            new_pos[sel] = eye + dirs * t_new[:, None]
            new_z[sel] = z[sel] * t_new / dist
        return new_pos, new_z

    # ------------------------------------------------------------ main API
    @torch.no_grad()
    def generate(
        self,
        n: int,
        W: torch.Tensor,
        U: torch.Tensor,
        gen: torch.Generator,
        opts: GeneratorOptions | None = None,
        positions: torch.Tensor | None = None,
    ) -> Assembly:
        """n candidates guided by utility map W [V,h,w] and uncovered map U [V,h,w]."""
        opts = opts or GeneratorOptions()
        ctx = self.ctx
        V, h, w = W.shape
        bank = ctx.bank
        pos = positions if positions is not None else self.sample_positions(n, U, gen, opts.explore_probability)
        n = pos.shape[0]

        # reference view per candidate, weighted by remaining uncovered mass
        mass = U.sum((-1, -2)) + 1e-6
        view = torch.multinomial(mass / mass.sum(), n, replacement=True, generator=gen)
        uv_px = torch.zeros(n, 2, device=self.device)
        z = torch.zeros(n, device=self.device)
        f = torch.zeros(n, device=self.device)
        for v, cam in enumerate(ctx.cameras):
            sel = (view == v).nonzero(as_tuple=True)[0]
            if len(sel):
                p, zz, _ = project_points(pos[sel], cam, (h, w))
                uv_px[sel], z[sel] = p, zz.clamp_min(1e-3)
                f[sel] = cam.focal_px(h)

        # scale from inscribed radius of the uncovered region (fallback: target interior)
        rad_u = inscribed_radius_map(U)
        rad_t = ctx.targets.fg_distance(h)
        col = uv_px[:, 0].round().long().clamp(0, w - 1)
        row = uv_px[:, 1].round().long().clamp(0, h - 1)
        r_px = torch.maximum(rad_u[view, row, col], 0.5 * rad_t[view, row, col]).clamp_min(1.0)
        jitter = opts.scale_jitter[0] + (opts.scale_jitter[1] - opts.scale_jitter[0]) * torch.rand(n, device=self.device, generator=gen)
        if ctx.constraints.scale_fixed:
            # fixed size: the desired projected radius determines the DEPTH instead (z = f s / r)
            s = torch.full((n,), float(ctx.cfg.scale.fixed), device=self.device)
            z_des = f * s / (r_px * jitter)
            pos, z = self._slide_to_depth(pos, view, z, z_des)
        else:
            s = r_px * z / f * jitter
            smin, smax = float(ctx.cfg.scale.min), float(ctx.cfg.scale.max)
            s = s.clamp(smin, smax)

        # mesh + orientation
        M = len(ctx.library)
        mesh = torch.randint(0, M, (n,), device=self.device, generator=gen)
        R = random_rotations(n, gen, device=self.device)
        use_bank = torch.rand(n, device=self.device, generator=gen) < opts.bank_probability
        if use_bank.any():
            bi = use_bank.nonzero(as_tuple=True)[0]
            half = FRAME * f[bi] * s[bi] / z[bi]
            patches = self._patches(W, view[bi], uv_px[bi], half, bank.d)
            scores = bank.score_patches(patches, opts.allowed_meshes)  # [nb, M*K]
            k = min(opts.top_k, scores.shape[1])
            top_s, top_i = scores.topk(k, dim=-1)
            pick = torch.randint(0, k, (len(bi),), device=self.device, generator=gen)
            flat = top_i[torch.arange(len(bi), device=self.device), pick]
            m_idx, r_idx = bank.entry(flat)
            mesh[bi] = m_idx
            for v, cam in enumerate(ctx.cameras):
                sv = (view[bi] == v).nonzero(as_tuple=True)[0]
                if len(sv):
                    R[bi[sv]] = bank.world_rotation(r_idx[sv], cam)
        elif opts.allowed_meshes is not None:
            allowed = opts.allowed_meshes.nonzero(as_tuple=True)[0]
            mesh = allowed[torch.randint(0, len(allowed), (n,), device=self.device, generator=gen)]
        return Assembly(mesh, pos, matrix_to_rot6d(R), torch.log(s))
