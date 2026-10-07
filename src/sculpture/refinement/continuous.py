"""Fixed-structure continuous refinement (Milestone 3 baseline).

Jointly optimizes translation / rot6d / log_scale of a FIXED set of instances
against the shared multiscale silhouette loss plus constraint penalties. The
discrete structure (which meshes, how many) never changes here. All three
methods use this as their final raster polish so reported fidelity is
produced by the same objective.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..geometry.rotation import matrix_to_rot6d, rot6d_to_matrix
from ..scene.assembly import Assembly


@dataclass
class RefineResult:
    assembly: Assembly
    loss: float               # image loss at progress=1 (working resolution)
    total: float              # loss + penalties
    initial_loss: float
    steps: int
    history: list = field(default_factory=list)


def refine_assembly(
    ctx,
    assembly: Assembly,
    steps: int,
    progress: tuple[float, float] = (0.3, 1.0),
    resolution: int | None = None,
    trainable: torch.Tensor | None = None,
    collisions: bool = True,
    lr_scale: float = 1.0,
    deadline: float | None = None,
    project: bool = True,
    collision_weight: float | None = None,
    callback: Callable[[int, dict, Assembly], None] | None = None,
) -> RefineResult:
    """Adam on all continuous parameters; returns the best state seen."""
    res = int(resolution or ctx.working_resolution)
    if len(assembly) == 0 or steps <= 0:
        with torch.no_grad():
            L = float(ctx.loss(ctx.render(assembly, res), 1.0))
        return RefineResult(assembly.detach(), L, L, L, 0)
    rc = ctx.cfg.refine
    a = assembly.params()
    opt = torch.optim.Adam(
        [
            {"params": [a.translation], "lr": rc.lr_translation * lr_scale},
            {"params": [a.rot6d], "lr": rc.lr_rotation * lr_scale},
            {"params": [a.log_scale], "lr": rc.lr_log_scale * lr_scale * ctx.constraints.log_scale_lr_factor},
        ]
    )
    mask = None if trainable is None else trainable.to(a.device).float()
    best = None
    history = []
    initial = None
    for it in range(steps):
        p = progress[0] + (progress[1] - progress[0]) * it / max(steps - 1, 1)
        R = ctx.render(a, res)
        L_sched = ctx.loss(R, p)
        with torch.no_grad():
            L_final = float(ctx.loss(R, 1.0))
        pen, pen_terms = ctx.constraints.penalty(a, collisions=collisions, lambda_collision=collision_weight)
        total = L_sched + pen
        score = L_final + float(pen.detach())
        if initial is None:
            initial = L_final
        if best is None or score < best[0]:
            best = (score, L_final, a.detach())
        history.append({"step": it, "loss": L_final, "scheduled": float(L_sched.detach()), **pen_terms})
        if callback is not None:
            callback(it, history[-1], a)
        if deadline is not None and time.time() > deadline:
            break
        opt.zero_grad(set_to_none=True)
        total.backward()
        if mask is not None:
            a.translation.grad *= mask[:, None]
            a.rot6d.grad *= mask[:, None]
            a.log_scale.grad *= mask
        opt.step()
        if it % 10 == 9:  # keep the 6D parameters well conditioned
            with torch.no_grad():
                a.rot6d.copy_(matrix_to_rot6d(rot6d_to_matrix(a.rot6d)))
                if project and ctx.strict:  # projected gradient: stay (nearly) feasible
                    proj, _ = ctx.constraints.project_inside(a.detach())
                    a.translation.copy_(proj.translation)
                    a.log_scale.copy_(proj.log_scale)
    # evaluate the final state too
    with torch.no_grad():
        R = ctx.render(a, res)
        L_final = float(ctx.loss(R, 1.0))
        pen, _ = ctx.constraints.penalty(a, collisions=collisions, lambda_collision=collision_weight)
        if L_final + float(pen) < best[0]:
            best = (L_final + float(pen), L_final, a.detach())
    out = best[2].normalized_rotations()
    if project and ctx.strict:
        out, _ = ctx.constraints.project_inside(out)
        with torch.no_grad():
            L_proj = float(ctx.loss(ctx.render(out, res), 1.0))
        best = (best[0] - best[1] + L_proj, L_proj, out)
    return RefineResult(out, best[1], best[0], initial, len(history), history)
