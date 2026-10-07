"""Repair moves for weak objects: SWAP, RESEED, LARGE POSE JUMP (Method 1).

Deletion is used only to undo harmful states (negative marginal contribution
that no repair can fix), never as a simplicity operation.
"""

from __future__ import annotations

import logging

import torch

from ...geometry.rotation import matrix_to_rot6d, perturb_rotations
from ...loss.silhouette import secant_utility
from ...proposals.candidate_generator import GeneratorOptions
from ...proposals.residual import uncovered
from ...proposals.utility_map import approximate_utility
from ...scene.assembly import Assembly
from .beam import lookahead

log = logging.getLogger(__name__)


@torch.no_grad()
def marginal_contributions(ctx, assembly: Assembly, res: int, chunk: int = 64) -> torch.Tensor:
    """contribution_i = L(X \\ O_i) - L(X), from binary instance masks (approximate)."""
    if len(assembly) == 0:
        return torch.zeros(0, device=ctx.device)
    S = ctx.render_instances(assembly, res) > 0.5  # [N,V,h,w]
    C = S.sum(0)
    L_full = ctx.loss((C > 0).float(), 1.0)
    out = []
    for s in range(0, len(assembly), chunk):
        R_minus = ((C[None] - S[s : s + chunk].int()) > 0).float()
        out.append(ctx.loss(R_minus, 1.0) - L_full)
    return torch.cat(out)


def replace_row(a: Assembly, i: int, new: Assembly) -> Assembly:
    return Assembly(
        torch.cat([a.mesh_ids[:i], new.mesh_ids, a.mesh_ids[i + 1 :]]),
        torch.cat([a.translation[:i], new.translation, a.translation[i + 1 :]]),
        torch.cat([a.rot6d[:i], new.rot6d, a.rot6d[i + 1 :]]),
        torch.cat([a.log_scale[:i], new.log_scale, a.log_scale[i + 1 :]]),
    )


def _swap_candidates(ctx, obj: Assembly, k: int) -> Assembly:
    """Silhouette-compatible neighbours of other meshes, same position/scale."""
    bank = ctx.bank
    cam = ctx.cameras[0]
    Rv = cam.torch_rotation(ctx.device)
    R_cam = Rv[None] @ obj.rotation_matrices()
    rot_idx = bank.nearest_rotation(R_cam)
    nb, _ = bank.similar_entries(bank.flat(obj.mesh_ids, rot_idx))
    nb = nb[0, :k]
    if nb.numel() == 0:
        return Assembly.empty(ctx.device)
    m, r = bank.entry(nb)
    R = bank.world_rotation(r, cam)
    n = len(m)
    return Assembly(m, obj.translation.expand(n, 3).clone(), matrix_to_rot6d(R), obj.log_scale.expand(n).clone())


def repair_assembly(ctx, assembly: Assembly, cfg, gen: torch.Generator, res: int, cand_res: int, steps: int):
    """Try to improve the weakest objects. Returns (assembly, stats dict)."""
    stats = {"swap": 0, "reseed": 0, "pose_jump": 0, "delete": 0, "tried": 0}
    if len(assembly) < 1:
        return assembly, stats
    contrib = marginal_contributions(ctx, assembly, res)
    med = contrib.median().item()
    order = torch.argsort(contrib).tolist()
    weak = [i for i in order[: int(cfg.repair_max_objects)] if contrib[i].item() < max(1e-6, 0.25 * med)]
    to_delete = []
    I_c = ctx.targets.soft(cand_res)
    for i in weak:
        stats["tried"] += 1
        obj = assembly[i]
        base = assembly.without(i)
        with torch.no_grad():
            R_base = ctx.render(base, res)
            R_base_c = ctx.render(base, cand_res)
            W = secant_utility(R_base_c, I_c, ctx.loss.w_cov, ctx.loss.w_neg)
            U = uncovered(R_base_c, I_c)
        alts, kinds = [obj], ["keep"]
        sw = _swap_candidates(ctx, obj, 4)
        alts.append(sw)
        kinds += ["swap"] * len(sw)
        # RESEED: teleport into the current residual (any model)
        pool = ctx.generator.generate(int(cfg.repair_candidates), W, U, gen)
        with torch.no_grad():
            util = approximate_utility(W, ctx.render_instances(pool, cand_res, lod="coarse"))
        rs = pool[util.topk(min(4, len(pool))).indices]
        alts.append(rs)
        kinds += ["reseed"] * len(rs)
        # LARGE POSE JUMP: bank orientations of the same mesh at the same spot + random perturbations
        allowed = torch.zeros(len(ctx.library), dtype=torch.bool, device=ctx.device)
        allowed[obj.mesh_ids] = True
        pj = ctx.generator.generate(4, W, U, gen, GeneratorOptions(bank_probability=1.0, allowed_meshes=allowed),
                                    positions=obj.translation.expand(4, 3).clone())
        pj = pj.with_params(log_scale=obj.log_scale.expand(4).clone())
        R_rand = perturb_rotations(obj.rotation_matrices().expand(4, 3, 3), 0.8, gen)
        pr = Assembly(obj.mesh_ids.expand(4).clone(), obj.translation.expand(4, 3).clone(), matrix_to_rot6d(R_rand), obj.log_scale.expand(4).clone())
        alts += [pj, pr]
        kinds += ["pose_jump"] * 8
        cands = Assembly.concat(alts)
        opt, L, pen, _ = lookahead(ctx, base, R_base, cands, steps, res)
        score = L + pen
        best = int(score.argmin())
        keep_score = float(score[0])
        with torch.no_grad():
            base_score = float(ctx.loss(R_base, 1.0))
        if best != 0 and float(score[best]) < keep_score - 1e-6:
            assembly = replace_row(assembly, i, opt[best])
            stats[kinds[best]] += 1
        else:
            assembly = replace_row(assembly, i, opt[0])  # locally improved original
            if contrib[i].item() < 0 and base_score < keep_score:
                to_delete.append(i)  # harmful and unfixable -> undo
    if to_delete:
        keep = torch.ones(len(assembly), dtype=torch.bool, device=ctx.device)
        keep[torch.tensor(to_delete, device=ctx.device)] = False
        assembly = assembly[keep]
        stats["delete"] = len(to_delete)
    return assembly, stats
