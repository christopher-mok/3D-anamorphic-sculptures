"""Approximate pricing: continuous placements maximizing the dual image weight.

The pricing problem  max_c  sum_p W_dual(p) S_c(p) - sigma  is nonconvex in
(t, R, s) and discrete in the mesh. It is solved *heuristically*: many
silhouette-bank-initialized placements are optimized in parallel with
nvdiffrast gradients. Consequently the LP bound is rigorous only over the
discovered columns, not over the continuous placement space.
"""

from __future__ import annotations

import torch

from ...proposals.candidate_generator import GeneratorOptions
from ...scene.assembly import Assembly
from ...targets.pyramids import downsample_area


def price_columns(ctx, W_dual: torch.Tensor, sigma: float, n: int, steps: int, render_res: int, gen: torch.Generator, opts: GeneratorOptions | None = None):
    """Returns (placements Assembly, exact reduced profit [n] on binary masks, binary masks [n, V*res*res])."""
    V, res, _ = W_dual.shape
    W_up = torch.nn.functional.interpolate(W_dual[None], size=(render_res, render_res), mode="nearest")[0]
    positive = W_dual.clamp_min(0)
    U = torch.nn.functional.interpolate((positive / positive.max().clamp_min(1e-12))[None], size=(render_res, render_res), mode="nearest")[0]
    cands = ctx.generator.generate(n, W_up, U, gen, opts)

    # normalized objective so that penalties have a comparable scale
    norm = positive.sum().clamp_min(1e-12)
    Wn = W_dual / norm
    rc = ctx.cfg.refine
    p = cands.params()
    opt = torch.optim.Adam([{"params": [p.translation], "lr": rc.lr_translation}, {"params": [p.rot6d], "lr": rc.lr_rotation},
                            {"params": [p.log_scale], "lr": rc.lr_log_scale * ctx.constraints.log_scale_lr_factor}])
    for _ in range(steps):
        S = downsample_area(ctx.render_instances(p, render_res), res)
        gain = (Wn[None] * S).sum((1, 2, 3))
        pen = ctx.constraints.candidate_penalty(p, collisions=False)
        loss = (-gain + pen).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    out = p.detach().normalized_rotations()
    valid = torch.ones(len(out), dtype=torch.bool, device=ctx.device)
    if ctx.strict:
        out, valid = ctx.constraints.project_inside(out)
    with torch.no_grad():
        S = downsample_area(ctx.render_instances(out, render_res), res) >= 0.5
        profit = (W_dual[None] * S).sum((1, 2, 3)) - sigma
        profit = torch.where(valid, profit, torch.full_like(profit, -float("inf")))
    return out, profit, S.reshape(len(out), -1)
