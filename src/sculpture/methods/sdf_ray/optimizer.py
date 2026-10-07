"""METHOD 3: SDF ray packing + raster polish.

A continuous 3D formulation. Instead of rasterized image boundaries, optimize
whether every target-foreground camera ray intersects the (soft) union of the
object SDFs:

    d_r      = softmin_k phi_X(r(t_k))        (stratified samples of the ray ∩ Omega)
    L_ray    = mean_r  tau * softplus(d_r / tau)       foreground rays should hit
    L_bg     = mean_r  tau * softplus(-d_r / tau)      background rays should miss
    + containment ReLU(phi_Omega)^2, SDF collisions ReLU(-phi_j(x))^2, scale range

(the spec's softplus(d/tau) is scaled by tau so the gradient scale does not
change while tau is annealed). This gives long-range 3D gradients: a ray
"feels" an object long before its silhouette touches the pixel. Weak objects
are periodically reseeded into uncovered ray regions and objects are added on
plateaus. Finally the shared nvdiffrast multiscale loss polishes the result.
"""

from __future__ import annotations

import logging

import torch

from ...geometry.rotation import matrix_to_rot6d, rot6d_to_matrix
from ...loss.silhouette import secant_utility
from ...proposals.candidate_generator import GeneratorOptions
from ...proposals.residual import uncovered
from ...refinement.continuous import refine_assembly
from ...scene.assembly import Assembly
from ...scene.serialization import assembly_from_dict, assembly_to_dict
from ..base import OptimizationMethod
from .rays import build_ray_sets
from .union_sdf import instance_sdf, softmin_st

log = logging.getLogger(__name__)


class SDFRayOptimizer(OptimizationMethod):
    name = "sdf_ray"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._restored: dict | None = None
        self.current: Assembly | None = None

    def checkpoint_state(self) -> dict:
        return {"current": assembly_to_dict(self.current, self.ctx.library)} if self.current is not None else {}

    def restore_state(self, state: dict) -> None:
        self._restored = state

    # ------------------------------------------------------------ initialization
    @torch.no_grad()
    def greedy_init(self, n_objects: int, gen: torch.Generator, base: Assembly | None = None) -> Assembly:
        """Target-informed init: pool of residual-guided / bank-retrieved candidates,
        greedily selected by exact union utility at low resolution."""
        ctx, cfg = self.ctx, self.cfg
        res = 64
        I = ctx.targets.soft(res)
        base = base if base is not None else Assembly.empty(ctx.device)
        R = ctx.render(base, res) if len(base) else torch.zeros_like(I)
        W = secant_utility(R, I, ctx.loss.w_cov, ctx.loss.w_neg)
        pool = ctx.generator.generate(max(n_objects * int(cfg.init_pool_factor), 32), W, uncovered(R, I), gen,
                                      GeneratorOptions(explore_probability=0.1, bank_probability=0.8))
        if ctx.strict:
            pool, valid = ctx.constraints.project_inside(pool)
            pool = pool[valid]
        S = (ctx.render_instances(pool, res, lod="coarse") > 0.5).float()
        # collision-aware: never pick a candidate that intersects the base or an earlier pick
        excluded = torch.zeros(len(pool), dtype=torch.bool, device=ctx.device)
        partners: dict[int, list[int]] = {}
        if ctx.constraints.hard_collisions and len(pool) > 1:
            if len(base):
                cp, _ = ctx.constraints.conflicts(pool, base)
                excluded[cp[:, 0].unique()] = True
            pp, _ = ctx.constraints.conflicts(pool)
            for i, j in pp.tolist():
                partners.setdefault(i, []).append(j)
                partners.setdefault(j, []).append(i)
        chosen = []
        for _ in range(min(n_objects, len(pool))):
            W = secant_utility(R, I, ctx.loss.w_cov, ctx.loss.w_neg)
            gain = (W[None] * S * (1 - R)[None]).sum((1, 2, 3))
            gain[excluded] = -float("inf")
            if chosen:
                gain[torch.tensor(chosen, device=ctx.device)] = -float("inf")
            c = int(gain.argmax())
            if gain[c] <= 0:
                break
            chosen.append(c)
            for j in partners.get(c, []):
                excluded[j] = True
            R = torch.maximum(R, S[c])
        if not chosen:
            return Assembly.empty(ctx.device)
        return pool[torch.tensor(chosen, device=ctx.device)]

    # ------------------------------------------------------------ ray objective
    def ray_losses(self, a: Assembly, fg, bg, tau: float, gen, n_fg: int, n_bg: int, k: int):
        ctx, cfg = self.ctx, self.cfg
        sdf = ctx.constraints.sdf
        margin = 4.0 * max(tau, float(cfg.union_temperature))
        out = {}
        total = torch.zeros((), device=ctx.device)
        for name, rays, n, sign, w in (("fg", fg, n_fg, 1.0, float(cfg.w_ray)), ("bg", bg, n_bg, -1.0, float(cfg.w_bg))):
            if len(rays) == 0 or n == 0 or w == 0:
                out[name] = 0.0
                continue
            idx = torch.randint(0, len(rays), (min(n, len(rays)),), device=ctx.device, generator=gen)
            sub = rays.subset(idx)
            pts = sub.sample_points(k, gen).reshape(-1, 3)
            phi_i = instance_sdf(sdf, a, pts, margin)                           # [N, Q]
            phi = softmin_st(phi_i, float(cfg.union_temperature), dim=0)            # [Q]
            d = softmin_st(phi.view(len(sub), k), float(cfg.ray_temperature), dim=1)  # [R]
            L = (tau * torch.nn.functional.softplus(sign * d / tau)).mean()
            total = total + w * L
            out[name] = float(L.detach())
            out[f"{name}_hit"] = float((d < 0).float().mean())
            if name == "fg":
                with torch.no_grad():
                    owner = phi_i.view(len(a), len(sub), k).amin(-1).argmin(0)  # object closest to each ray
                    hit = d < 0
                    out["ownership"] = torch.bincount(owner[hit], minlength=len(a))
        return total, out

    # ------------------------------------------------------------ main loop
    def optimize(self) -> Assembly:
        ctx, cfg = self.ctx, self.cfg
        self.phase = "rays"
        fg, bg, stats = build_ray_sets(ctx, int(cfg.ray_resolution))
        self.info.update({f"ray_{k}": v for k, v in stats.items()})
        gen = ctx.generator_torch(self.seed * 31337 + 5)
        if self._restored and self._restored.get("current"):
            a = assembly_from_dict(self._restored["current"], ctx.library, ctx.device)
        else:
            self.phase = "init"
            a = self.greedy_init(int(cfg.initial_objects), gen)
        log.info("[sdf_ray] %d fg rays (%d unsatisfiable), %d bg rays, %d initial objects", stats["fg_rays"], stats["fg_unsatisfiable"], stats["bg_rays"], len(a))

        iters = int(cfg.iterations)
        polish_reserve = 0.25 * self.max_runtime
        p, opt = self._new_optimizer(a)
        last_hit = 0.0
        it0 = self.iteration
        for it in range(it0, iters):
            if self.out_of_time(reserve=polish_reserve):
                break
            self.phase = "ray_packing"
            frac = it / max(iters - 1, 1)
            tau = float(cfg.tau_start) * (float(cfg.tau_end) / float(cfg.tau_start)) ** frac
            gen_it = ctx.generator_torch(self.seed * 31337 + 1000 + it)
            L_ray, terms = self.ray_losses(p, fg, bg, tau, gen_it, int(cfg.rays_per_view) * ctx.num_views,
                                           int(cfg.bg_rays_per_view) * ctx.num_views, int(cfg.samples_per_ray))
            # moderate clearance weight while packing (the ray loss is O(0.05)); the shared
            # final stage resolves any remaining intersections
            pen, pen_terms = ctx.constraints.penalty(p, collisions=True, lambda_collision=float(cfg.get("lambda_collision", 30.0)))
            loss = L_ray + pen
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            self.iteration = it + 1
            self.current = p.detach()
            if it % 10 == 9:
                with torch.no_grad():
                    p.rot6d.copy_(matrix_to_rot6d(rot6d_to_matrix(p.rot6d)))
                    if ctx.strict and it >= iters // 2:  # projected steps in the second half
                        proj, _ = ctx.constraints.project_inside(p.detach())
                        p.translation.copy_(proj.translation)
                        p.log_scale.copy_(proj.log_scale)

            # periodic reseeding of weak objects / growth on plateaus
            if int(cfg.reseed_every) > 0 and (it + 1) % int(cfg.reseed_every) == 0 and it + 1 < iters:
                a = p.detach()
                own = terms.get("ownership")
                hit = terms.get("fg_hit", 0.0)
                a = self._reseed(a, own, gen_it)
                # grow when ray coverage plateaus (piece count is not penalized)
                if hit - last_hit < float(cfg.plateau_tolerance) and hit < 0.995 and len(a) < int(ctx.cfg.max_objects):
                    n_add = min(int(cfg.add_objects_on_plateau), int(ctx.cfg.max_objects) - len(a))
                    extra = self.greedy_init(n_add, gen_it, base=a)
                    a = Assembly.concat([a, extra]) if len(extra) else a
                    self.info["objects_added"] = self.info.get("objects_added", 0) + len(extra)
                last_hit = hit
                p, opt = self._new_optimizer(a)
            if it % 20 == 0 or it + 1 == iters:
                self.log_iteration(p.detach(), ray_loss=float(L_ray.detach()), tau=tau, fg_hit=terms.get("fg_hit"),
                                   bg_hit=terms.get("bg_hit"), **pen_terms)

        a = p.detach().normalized_rotations()
        if ctx.strict:
            a, _ = ctx.constraints.project_inside(a)
        self.consider_best(a)
        self.info["objects_before_polish"] = len(a)

        # raster polish with the shared nvdiffrast multiscale loss (determines reported fidelity)
        self.phase = "raster_polish"
        if len(a) and int(cfg.polish_steps) > 0:
            r = refine_assembly(ctx, a, int(cfg.polish_steps), progress=(0.3, 1.0), deadline=None)
            a = r.assembly
            self.info["polish_initial_loss"] = r.initial_loss
            self.info["polish_final_loss"] = r.loss
        self.consider_best(a)
        self.iteration += 1
        self.log_iteration(self.best_assembly, force_report=True)
        return self.finalize(self.best_assembly)

    def _new_optimizer(self, a: Assembly):
        cfg = self.cfg
        p = a.params()
        opt = torch.optim.Adam([{"params": [p.translation], "lr": float(cfg.lr_translation)},
                                {"params": [p.rot6d], "lr": float(cfg.lr_rotation)},
                                {"params": [p.log_scale], "lr": float(cfg.lr_log_scale) * self.ctx.constraints.log_scale_lr_factor}])
        return p, opt

    @torch.no_grad()
    def _reseed(self, a: Assembly, ownership: torch.Tensor | None, gen: torch.Generator) -> Assembly:
        """Teleport the weakest objects (fewest owned foreground rays) into uncovered regions."""
        if ownership is None or len(a) < 2:
            return a
        n = max(1, int(round(float(self.cfg.reseed_fraction) * len(a))))
        weak = torch.argsort(ownership.float())[:n]
        weak = weak[ownership[weak] <= ownership.float().median() * 0.25]
        if len(weak) == 0:
            return a
        keep = torch.ones(len(a), dtype=torch.bool, device=a.device)
        keep[weak] = False
        kept = a[keep]
        new = self.greedy_init(len(weak), gen, base=kept)
        self.info["reseeded"] = self.info.get("reseeded", 0) + len(new)
        return Assembly.concat([kept, new]) if len(new) else a
