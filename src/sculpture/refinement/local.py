"""Frozen-context local optimization of independent candidate placements.

Shared utility (beam lookahead, beam repair, diversity swaps): every candidate
is optimized on its own against the image loss of (frozen context ∪ candidate)
plus its constraint penalty. Candidates are independent, so one batched Adam
run optimizes all of them at once.
"""

from __future__ import annotations

import torch

from ..geometry.rotation import matrix_to_rot6d, rot6d_to_matrix
from ..loss.silhouette import soft_union
from ..scene.assembly import Assembly


def lookahead(ctx, parent: Assembly, R_parent: torch.Tensor, cands: Assembly, steps: int, res: int, progress: float = 0.8):
    """Locally optimize each candidate (t, rot6d, log_scale) with the parent frozen.

    Candidates are independent: the summed objective has a block-diagonal
    Hessian, so one batched Adam run optimizes all of them at once.
    Returns (optimized candidates, image loss [K], penalty [K], union masks [K,V,h,w]).
    """
    rc = ctx.cfg.refine
    p = cands.params()
    opt = torch.optim.Adam(
        [{"params": [p.translation], "lr": rc.lr_translation}, {"params": [p.rot6d], "lr": rc.lr_rotation}, {"params": [p.log_scale], "lr": rc.lr_log_scale * ctx.constraints.log_scale_lr_factor}]
    )
    K = len(cands)
    best_score = torch.full((K,), float("inf"), device=ctx.device)
    best_t, best_r, best_s = p.translation.detach().clone(), p.rot6d.detach().clone(), p.log_scale.detach().clone()
    best_L = torch.zeros(K, device=ctx.device)
    best_pen = torch.zeros(K, device=ctx.device)
    best_U = None
    Rp = R_parent.detach()[None]
    for step in range(steps + 1):
        S = ctx.render_instances(p, res)
        U = soft_union(Rp, S)
        L_sched = ctx.loss(U, progress)
        pen = ctx.constraints.candidate_penalty(p, fixed=parent)
        with torch.no_grad():
            L_fin = ctx.loss(U, 1.0)
            score = L_fin + pen
            better = score < best_score
            best_score = torch.where(better, score, best_score)
            best_t[better] = p.translation[better]
            best_r[better] = p.rot6d[better]
            best_s[better] = p.log_scale[better]
            best_L = torch.where(better, L_fin, best_L)
            best_pen = torch.where(better, pen, best_pen)
            best_U = U.detach().clone() if best_U is None else torch.where(better[:, None, None, None], U, best_U)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        (L_sched + pen).sum().backward()
        opt.step()
    best_r = matrix_to_rot6d(rot6d_to_matrix(best_r))
    return Assembly(cands.mesh_ids, best_t, best_r, best_s), best_L, best_pen, best_U
