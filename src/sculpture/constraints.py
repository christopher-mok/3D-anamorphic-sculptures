"""ConstraintEvaluator: one place for every physical / geometric constraint.

Methods can consume constraints as
  * differentiable penalties  (``penalty``),
  * hard rejection            (``valid_mask``),
  * combinatorial conflicts   (``conflicts``)  e.g. MILP pair cuts.
"""

from __future__ import annotations

import math

import torch

from .geometry.collisions import reference_intersections
from .loss.collision import collision_loss, colliding_pairs, cross_collision_loss
from .loss.containment import containment_per_object
from .scene.assembly import Assembly


class ConstraintEvaluator:
    def __init__(self, library, hull, cfg):
        self.library = library
        self.hull = hull
        c = cfg.constraints
        self.lambda_contain = float(c.lambda_contain)
        self.lambda_collision = float(c.lambda_collision)
        self.lambda_scale = float(c.lambda_scale)
        self.contain_tol = float(c.containment_tolerance)
        # clearance margin (world units): objects closer than this count as intersecting
        self.collision_margin = float(c.get("collision_margin", 0.005))
        self.hard_collisions = bool(c.get("hard_collisions", True))
        self.n_contain = int(c.contain_samples)
        self.n_collision = int(c.collision_samples)
        self.log_smin = math.log(float(cfg.scale.min))
        self.log_smax = math.log(float(cfg.scale.max))
        # fixed-scale mode: every instance has the same size; apparent size comes from depth only
        # scale is not a parameter in "fixed" (one uniform size) and "native" (each model keeps the
        # size of its source file x native_factor) modes
        self.scale_mode = str(cfg.scale.get("mode", "free"))
        self.scale_fixed = self.scale_mode in ("fixed", "native")
        self.fixed_log_scale = math.log(float(cfg.scale.get("fixed", 0.15)))
        if self.scale_mode == "native":
            factor = float(cfg.scale.get("native_factor", 0.3))
            radii = [1.0 / float(e.data["norm_scale"]) for e in library.entries]
            self.fixed_log_scales = torch.log(torch.tensor(radii, device=library.device) * factor)
        else:
            self.fixed_log_scales = torch.full((len(library),), self.fixed_log_scale, device=library.device)
        self.bmin = torch.tensor(cfg.bounding_volume.min, dtype=torch.float32, device=library.device)
        self.bmax = torch.tensor(cfg.bounding_volume.max, dtype=torch.float32, device=library.device)
        self.sdf_resolution = int(cfg.meshes.sdf_resolution)
        self.camera_eye: torch.Tensor | None = None  # set by build_context (first camera)

    @property
    def log_scale_lr_factor(self) -> float:
        """0 in fixed-scale mode: optimizers then never change log_scale."""
        return 0.0 if self.scale_fixed else 1.0

    @property
    def sdf(self):
        return self.library.sdf(self.sdf_resolution)

    # ------------------------------------------------------------ differentiable
    def scale_penalty_per_object(self, a: Assembly) -> torch.Tensor:
        if self.scale_fixed:
            return torch.zeros_like(a.log_scale)
        ls = a.log_scale
        return torch.relu(ls - self.log_smax) ** 2 + torch.relu(self.log_smin - ls) ** 2

    def bounds_penalty_per_object(self, a: Assembly) -> torch.Tensor:
        t = a.translation
        return (torch.relu(t - self.bmax) ** 2 + torch.relu(self.bmin - t) ** 2).sum(-1)

    def penalty(self, a: Assembly, collisions: bool = True, lambda_collision: float | None = None) -> tuple[torch.Tensor, dict]:
        """Weighted sum of containment, collision, scale and bounds penalties."""
        dev = self.library.device
        if len(a) == 0:
            z = torch.zeros((), device=dev)
            return z, {"contain": 0.0, "collision": 0.0, "scale": 0.0}
        contain, _ = containment_per_object(a, self.library, self.hull, self.n_contain)
        contain = contain.mean()
        scale = (self.scale_penalty_per_object(a) + self.bounds_penalty_per_object(a)).mean()
        coll = collision_loss(a, self.library, self.sdf, self.n_collision, self.collision_margin) if (collisions and self.lambda_collision > 0) else torch.zeros((), device=dev)
        lam_c = self.lambda_collision if lambda_collision is None else float(lambda_collision)
        total = self.lambda_contain * contain + lam_c * coll + self.lambda_scale * scale
        return total, {"contain": float(contain.detach()), "collision": float(coll.detach()), "scale": float(scale.detach())}

    def candidate_penalty(self, cands: Assembly, fixed: Assembly | None = None, collisions: bool = True) -> torch.Tensor:
        """Per-candidate penalty [K]; each candidate is independent, ``fixed`` objects are frozen."""
        contain, _ = containment_per_object(cands, self.library, self.hull, min(self.n_contain, 64))
        pen = self.lambda_contain * contain + self.lambda_scale * (self.scale_penalty_per_object(cands) + self.bounds_penalty_per_object(cands))
        if collisions and fixed is not None and len(fixed) > 0 and self.lambda_collision > 0:
            pen = pen + self.lambda_collision * cross_collision_loss(cands, fixed.detach(), self.library, self.sdf, 96, self.collision_margin)
        return pen

    # ------------------------------------------------------------ hard
    @torch.no_grad()
    def containment_violation(self, a: Assembly) -> torch.Tensor:
        """Max phi_Omega over surface samples per object [N] (world units)."""
        _, mx = containment_per_object(a, self.library, self.hull, self.library.samples.shape[1])
        return mx

    @torch.no_grad()
    def valid_mask(self, a: Assembly, strict: bool = True) -> torch.Tensor:
        """Hard per-object validity: containment (strict mode) and scale range."""
        if self.scale_fixed:
            ok = torch.ones(len(a), dtype=torch.bool, device=a.device)
        else:
            ok = (a.log_scale >= self.log_smin - 1e-4) & (a.log_scale <= self.log_smax + 1e-4)
        if strict and self.hull.mode == "strict":
            ok &= self.containment_violation(a) <= self.contain_tol
        return ok

    def project_inside(self, a: Assembly, iters: int = 8) -> tuple[Assembly, torch.Tensor]:
        """Hard feasibility projection into Omega (strict mode).

        1. Move object centers inside Omega along -grad(phi) until phi(t) < -voxel/2.
        2. Shrink each scale to the largest factor in a fixed ladder whose surface
           samples satisfy the containment tolerance.
        Returns (projected assembly, valid mask). Rotation is unchanged.
        """
        if len(a) == 0:
            return a, torch.zeros(0, dtype=torch.bool, device=self.library.device)
        hull = self.hull
        vs = hull.mean_voxel_size
        t = a.translation.detach().clone()
        for _ in range(iters):
            with torch.enable_grad():
                tt = t.clone().requires_grad_(True)
                phi = hull.query_sdf(tt)
                (g,) = torch.autograd.grad(phi.sum(), tt)
            need = phi.detach() > -0.5 * vs
            if not need.any():
                break
            g = torch.nn.functional.normalize(g, dim=-1)
            t[need] = t[need] - (phi.detach()[need] + vs)[:, None] * g[need]
        if self.scale_fixed:
            out, valid = self._project_fixed_scale(a, t)
            return self._keep_locked(a, out, valid)
        with torch.no_grad():
            alphas = torch.tensor([1.0, 0.92, 0.85, 0.78, 0.7, 0.62, 0.55, 0.47, 0.4, 0.32, 0.25, 0.18], device=t.device)
            N, A = len(a), len(alphas)
            ls = a.log_scale.detach()[:, None] + torch.log(alphas)[None]  # [N,A]
            ls = torch.maximum(ls, torch.full_like(ls, self.log_smin))
            rep = Assembly(a.mesh_ids.repeat_interleave(A), t.repeat_interleave(A, 0), a.rot6d.detach().repeat_interleave(A, 0), ls.reshape(-1))
            ok = (self.containment_violation(rep) <= self.contain_tol).reshape(N, A)
            first = torch.where(ok.any(1), ok.float().argmax(1), torch.full((N,), A - 1, device=t.device))
            new_ls = ls[torch.arange(N, device=t.device), first]
            valid = ok.any(1)
        return self._keep_locked(a, a.with_params(translation=t, log_scale=new_ls).detach(), valid)

    @staticmethod
    def _keep_locked(orig: Assembly, out: Assembly, valid: torch.Tensor):
        """Locked pieces are user decisions: restore them unchanged and treat them as valid."""
        if not bool(orig.locked.any()):
            return out, valid
        lk = orig.locked
        t, ls, r6 = out.translation.clone(), out.log_scale.clone(), out.rot6d.clone()
        t[lk], ls[lk], r6[lk] = orig.translation.detach()[lk], orig.log_scale.detach()[lk], orig.rot6d.detach()[lk]
        return out.with_params(translation=t, rot6d=r6, log_scale=ls), valid | lk

    def _project_fixed_scale(self, a: Assembly, t: torch.Tensor, iters: int = 12):
        """Fixed-scale feasibility: the size may not change, so move the object instead.

        1. Gradient steps on the containment violation w.r.t. translation (mostly lateral).
        2. If still violating: a depth ladder along the ray from the first camera (keeps the
           object's projection in that view; farther away = smaller in the image).
        """
        ls = self.fixed_log_scales[a.mesh_ids].clone()
        b = a.with_params(translation=t, log_scale=ls).detach()
        n = min(self.n_contain, 128)
        for _ in range(iters):
            with torch.enable_grad():
                tt = b.translation.clone().requires_grad_(True)
                per, mx = containment_per_object(b.with_params(translation=tt), self.library, self.hull, n)
                (g,) = torch.autograd.grad(per.sum(), tt)
            need = mx.detach() > 0.5 * self.contain_tol
            if not need.any():
                break
            step = (mx.detach() + 0.5 * self.contain_tol).clamp(max=0.2)
            b.translation[need] -= step[need, None] * torch.nn.functional.normalize(g[need], dim=-1)
        with torch.no_grad():
            viol = self.containment_violation(b)
            bad = viol > self.contain_tol
            if bad.any() and self.camera_eye is not None:
                eye = self.camera_eye
                factors = torch.tensor([1.04, 1.08, 1.15, 1.25, 1.4, 1.6, 1.85, 2.2, 0.96, 0.92, 0.85], device=t.device)
                idx = bad.nonzero(as_tuple=True)[0]
                d = b.translation[idx] - eye
                cand_t = eye + d[:, None, :] * factors[None, :, None]  # [k, F, 3]
                k, F = cand_t.shape[:2]
                rep = b[idx.repeat_interleave(F)].with_params(translation=cand_t.reshape(-1, 3))
                ok = (self.containment_violation(rep) <= self.contain_tol).reshape(k, F)
                has = ok.any(1)
                first = ok.float().argmax(1)
                b.translation[idx[has]] = cand_t[has, first[has]]
                viol = self.containment_violation(b)
        return b, viol <= self.contain_tol

    @torch.no_grad()
    def conflicts(self, a: Assembly, b: Assembly | None = None):
        """Colliding pairs (combinatorial conflicts) and penetration depths."""
        return colliding_pairs(a, self.library, self.sdf, self.collision_margin, B=b)

    @torch.no_grad()
    def report(self, a: Assembly) -> dict:
        if len(a) == 0:
            return {"collisions": 0, "clearance_violations": 0, "containment_violations": 0, "max_containment_violation": 0.0}
        pairs, _ = self.conflicts(a)
        ref = reference_intersections(a, self.library)
        viol = self.containment_violation(a)
        return {
            # authoritative count: dense winding-number reference (actual interpenetration)
            "collisions": len(ref),
            # conservative detector: pairs closer than collision_margin (includes near-contacts)
            "clearance_violations": int(pairs.shape[0]),
            "containment_violations": int((viol > self.contain_tol).sum().item()),
            "max_containment_violation": float(viol.max().item()),
        }
