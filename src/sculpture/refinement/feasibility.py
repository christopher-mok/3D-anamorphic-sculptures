"""Shared final stage: make an assembly intersection-free.

"Intersecting" uses the conservative clearance test (ConstraintEvaluator.conflicts:
pieces closer than ``constraints.collision_margin``), whose recall against the
dense winding-number reference was verified to be complete for margins
>= 0.0025 on all saved demo results.

1. SEPARATE: a few rounds of joint refinement in which only the pieces involved in a
   conflict move, with the clearance penalty weight boosted.
2. FIX: for every remaining conflicting pair, the weaker contributor (marginal image
   contribution) is shrunk (free scale) or nudged along the first camera's ray
   (fixed scale), choosing the clear state with the best image loss.
3. REMOVE: only if no clear state exists ("deletion is only for invalid geometry").
"""

from __future__ import annotations

import logging

import torch

from ..geometry.collisions import reference_intersections
from ..loss.silhouette import soft_union
from ..scene.assembly import Assembly
from .continuous import refine_assembly
from .diversity import marginal_contributions, replace_row

log = logging.getLogger(__name__)


def _conflicts(ctx, a: Assembly) -> torch.Tensor:
    """Union of the conservative clearance test and the dense reference check, so the
    stage guarantees zero true intersections whatever collision_margin is set to."""
    det, _ = ctx.constraints.conflicts(a)
    ref = reference_intersections(a, ctx.library)
    pairs = {tuple(sorted(p)) for p in det.tolist()} | set(ref)
    return torch.tensor(sorted(pairs), dtype=torch.long, device=ctx.device).reshape(-1, 2)


def _fix_candidates(ctx, obj: Assembly) -> Assembly:
    cons = ctx.constraints
    parts = []
    if not cons.scale_fixed:
        alphas = torch.tensor([0.93, 0.86, 0.78, 0.7, 0.6, 0.5, 0.4, 0.3], device=ctx.device)
        ls = torch.maximum(obj.log_scale + torch.log(alphas), torch.full_like(alphas, cons.log_smin))
        k = len(alphas)
        parts.append(Assembly(obj.mesh_ids.expand(k).clone(), obj.translation.expand(k, 3).clone(), obj.rot6d.expand(k, 6).clone(), ls))
    if cons.camera_eye is not None:
        f = torch.tensor([1.02, 0.98, 1.04, 0.96, 1.07, 0.93, 1.12, 0.88], device=ctx.device)
        eye = cons.camera_eye
        t = eye + (obj.translation - eye) * f[:, None]
        k = len(f)
        parts.append(Assembly(obj.mesh_ids.expand(k).clone(), t, obj.rot6d.expand(k, 6).clone(), obj.log_scale.expand(k).clone()))
    return Assembly.concat(parts)


def resolve_intersections(ctx, a: Assembly, rounds: int = 6, steps: int = 30) -> tuple[Assembly, dict]:
    cons = ctx.constraints
    a = a.detach()
    pairs = _conflicts(ctx, a)
    stats = {"initial_conflicts": int(pairs.shape[0]), "separation_rounds": 0, "fixed": 0, "removed": 0}
    if pairs.shape[0] == 0:
        stats["final_conflicts"] = 0
        return a, stats

    # 1. local separation with a boosted clearance penalty
    for _ in range(rounds):
        if pairs.shape[0] == 0:
            break
        mask = torch.zeros(len(a), device=ctx.device)
        mask[pairs.flatten().unique()] = 1.0
        mask[a.locked] = 0.0
        r = refine_assembly(ctx, a, steps, progress=(1.0, 1.0), trainable=mask, lr_scale=0.5,
                            collision_weight=cons.lambda_collision * 30.0)
        a = r.assembly
        pairs = _conflicts(ctx, a)
        stats["separation_rounds"] += 1

    # 2./3. per-pair fix of the weaker piece, removal as a last resort
    res = int(ctx.cfg.beam.get("lookahead_resolution", 128))
    for _ in range(8):
        if pairs.shape[0] == 0:
            break
        contrib = marginal_contributions(ctx, a, res)
        remove: set[int] = set()
        handled: set[int] = set()
        for i, j in pairs.tolist():
            if i in handled or j in handled or i in remove or j in remove:
                continue
            li, lj = bool(a.locked[i]), bool(a.locked[j])
            if li and lj:  # both locked by the user: leave as is (reported)
                continue
            victim = j if li else i if lj else (i if contrib[i] <= contrib[j] else j)
            handled.add(victim)
            obj, others = a[victim], a.without(victim)
            cands = _fix_candidates(ctx, obj)
            ok = cons.valid_mask(cands)
            cp, _ = cons.conflicts(cands, others)
            if cp.shape[0]:
                ok[cp[:, 0].unique()] = False
            for k in ok.nonzero(as_tuple=True)[0].tolist():  # verify survivors with the reference
                trial = Assembly.concat([others, cands[k]])
                if any(j == len(others) for p in reference_intersections(trial, ctx.library) for j in p):
                    ok[k] = False
            if ok.any():
                with torch.no_grad():
                    R_o = ctx.render(others, res)
                    L = ctx.loss(soft_union(R_o[None], ctx.render_instances(cands, res)), 1.0)
                L = torch.where(ok, L, torch.full_like(L, float("inf")))
                a = replace_row(a, victim, cands[int(L.argmin())])
                stats["fixed"] += 1
            else:
                remove.add(victim)
        if remove:
            keep = torch.ones(len(a), dtype=torch.bool, device=ctx.device)
            keep[torch.tensor(sorted(remove), device=ctx.device)] = False
            a = a[keep]
            stats["removed"] += len(remove)
        pairs = _conflicts(ctx, a)
    stats["final_conflicts"] = int(pairs.shape[0])
    log.info("intersection resolution: %s", stats)
    return a, stats
