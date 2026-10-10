"""Shared type-swap stage that moves the piece-type counts toward the desired
distribution (more "randomness") while protecting image fidelity.

Objective  J = L_image + penalty + weight * deficit(type counts).
Repeatedly take a piece of the most over-represented type (weakest marginal
contributors first) and try to turn it into one of the most under-represented
types: silhouette-compatible bank orientations of the new mesh at the same
place, each locally optimized with the rest of the sculpture frozen. A swap is
accepted only if J decreases and the new piece does not intersect any other.
"""

from __future__ import annotations

import logging

import torch

from ..geometry.rotation import matrix_to_rot6d
from ..loss.diversity import assembly_diversity, target_distribution, type_counts
from ..scene.assembly import Assembly
from .local import lookahead

log = logging.getLogger(__name__)


@torch.no_grad()
def marginal_contributions(ctx, a: Assembly, res: int) -> torch.Tensor:
    """L(X without O_i) - L(X) from binary instance masks: [N]."""
    S = ctx.render_instances(a, res) > 0.5
    C = S.sum(0)
    L_full = ctx.loss((C > 0).float(), 1.0)
    out = [ctx.loss(((C[None] - S[s : s + 64].int()) > 0).float(), 1.0) - L_full for s in range(0, len(a), 64)]
    return torch.cat(out) if out else torch.zeros(0, device=ctx.device)


def replace_row(a: Assembly, i: int, new: Assembly) -> Assembly:
    return Assembly(
        torch.cat([a.mesh_ids[:i], new.mesh_ids, a.mesh_ids[i + 1 :]]),
        torch.cat([a.translation[:i], new.translation, a.translation[i + 1 :]]),
        torch.cat([a.rot6d[:i], new.rot6d, a.rot6d[i + 1 :]]),
        torch.cat([a.log_scale[:i], new.log_scale, a.log_scale[i + 1 :]]),
        torch.cat([a.locked[:i], new.locked, a.locked[i + 1 :]]),
    )


def _swap_candidates(ctx, obj: Assembly, types: list[int], per_type: int) -> Assembly:
    bank = ctx.bank
    cam = ctx.cameras[0]
    R_cam = cam.torch_rotation(ctx.device)[None] @ obj.rotation_matrices()
    flat = int(bank.flat(obj.mesh_ids, bank.nearest_rotation(R_cam))[0])
    parts = []
    for m in types:
        rot_idx = bank.similar_in_mesh(flat, m, per_type)
        R = bank.world_rotation(rot_idx, cam)
        k = len(rot_idx)
        parts.append(Assembly(torch.full((k,), m, device=ctx.device), obj.translation.expand(k, 3).clone(),
                              matrix_to_rot6d(R), obj.log_scale.expand(k).clone()))
    return Assembly.concat(parts)


def rebalance_types(ctx, a: Assembly, deadline_s: float | None = None) -> tuple[Assembly, dict]:
    import time

    cfg = ctx.cfg.diversity
    weight = float(cfg.weight)
    q = target_distribution(ctx.cfg, ctx.library)
    stats = {"swaps": 0, "tried": 0}
    before = assembly_diversity(a.mesh_ids, q)
    stats["randomness_before"], stats["deficit_before"] = before["randomness"], before["deficit"]
    if weight <= 0 or len(a) < 2:
        return a, stats
    res = int(ctx.cfg.beam.get("lookahead_resolution", 128))
    M = len(ctx.library)
    a = a.detach()
    t_end = None if deadline_s is None else time.time() + deadline_s
    tried_objects: set[int] = set()
    for _ in range(int(cfg.max_swaps)):
        if t_end is not None and time.time() > t_end:
            break
        counts = type_counts(a.mesh_ids, M)
        n = counts.sum()
        excess = counts - q * n  # >0 over-represented
        over = [int(m) for m in torch.argsort(excess, descending=True).tolist() if excess[m] > 0.5]
        under = [int(m) for m in torch.argsort(excess).tolist() if excess[m] < -0.5 and q[m] > 0]
        if not over or not under:
            break
        contrib = marginal_contributions(ctx, a, res)
        # weakest pieces of the most over-represented types not tried yet
        cand_objs = [i for i in torch.argsort(contrib).tolist()
                     if int(a.mesh_ids[i]) in over[:2] and i not in tried_objects and not bool(a.locked[i])]
        if not cand_objs:
            break
        i = cand_objs[0]
        tried_objects.add(i)
        stats["tried"] += 1
        obj = a[i]
        rest = a.without(i)
        with torch.no_grad():
            R_rest = ctx.render(rest, res)
        alts = Assembly.concat([obj, _swap_candidates(ctx, obj, under[: int(cfg.types_per_object)], int(cfg.orientations_per_type))])
        opt, L, pen, _ = lookahead(ctx, rest, R_rest, alts, int(cfg.swap_steps), res)
        if ctx.strict:
            opt, valid = ctx.constraints.project_inside(opt)
            with torch.no_grad():
                from ..loss.silhouette import soft_union

                L = ctx.loss(soft_union(R_rest[None], ctx.render_instances(opt, res)), 1.0)
                pen = ctx.constraints.candidate_penalty(opt, fixed=rest)
        else:
            valid = torch.ones(len(opt), dtype=torch.bool, device=ctx.device)
        # diversity term of the whole assembly after each alternative
        new_counts = counts[None].repeat(len(opt), 1)
        new_counts[:, int(obj.mesh_ids)] -= 1
        new_counts[torch.arange(len(opt)), opt.mesh_ids] += 1
        from ..loss.diversity import diversity_stats

        J = L + pen + weight * diversity_stats(new_counts, q)["deficit"]
        if ctx.constraints.hard_collisions:
            pairs, _ = ctx.constraints.conflicts(opt, rest)
            bad = torch.zeros(len(opt), dtype=torch.bool, device=ctx.device)
            if pairs.shape[0]:
                bad[pairs[:, 0].unique()] = True
            valid = valid & (~bad | (torch.arange(len(opt), device=ctx.device) == 0))
        J = torch.where(valid, J, torch.full_like(J, float("inf")))
        best = int(J.argmin())
        if best != 0 and J[best] < J[0] - 1e-6:
            a = replace_row(a, i, opt[best])
            stats["swaps"] += 1
            tried_objects.discard(i)
    after = assembly_diversity(a.mesh_ids, q)
    stats["randomness_after"], stats["deficit_after"] = after["randomness"], after["deficit"]
    log.info("diversity swaps: %d accepted / %d tried, randomness %.3f -> %.3f", stats["swaps"], stats["tried"],
             before["randomness"], after["randomness"])
    return a, stats
