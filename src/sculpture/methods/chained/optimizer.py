"""METHOD 4: Chained  SDF ray packing -> column-generation hole filling -> beam growth/repair.

A merged approach that runs the three independent solvers in sequence, each with a share of
the time budget, handing the current assembly from one to the next:

1. SDF ray packing  (wide-basin continuous 3D packing; fast global structure),
2. column generation in ANCHOR mode (the incoming pieces are fixed columns; LP duals price new
   placements and the MILP fills the remaining holes around them),
3. beam constructive search rooted at that assembly (growth, local lookahead, repairs).

The standalone methods are untouched (each stage IS the standalone implementation, run as an
intermediate stage that skips the shared final stages); the shared final stages (diversity
swaps, intersection resolution, original-mesh substitution) run once at the end of the chain.
"""

from __future__ import annotations

import dataclasses
import logging

from ...config import _wrap, deep_merge
from ...scene.assembly import Assembly
from ..base import OptimizationMethod, ProgressReporter

log = logging.getLogger(__name__)

STAGES = (("sdf_ray", "sdf_fraction"), ("column_generation", "cg_fraction"), ("beam", None))


class _StageReporter(ProgressReporter):
    """Forwards a stage's progress as the chained method's progress."""

    def __init__(self, parent: ProgressReporter, stage: str, owner: "ChainedOptimizer"):
        self.parent, self.stage, self.owner = parent, stage, owner

    def update(self, method, event):
        event = dict(event)
        event["phase"] = f"{self.stage}: {event.get('phase', '')}"
        event["runtime_s"] = self.owner.elapsed
        event["info"] = {**self.owner.info, **{f"{self.stage}_{k}": v for k, v in (event.get("info") or {}).items()
                                                if isinstance(v, (int, float, str, bool))}}
        self.parent.update(self.owner.name, event)

    def cancelled(self):
        return self.parent.cancelled()


class ChainedOptimizer(OptimizationMethod):
    name = "chained"

    def optimize(self) -> Assembly:
        from .. import get_method

        ctx, cfg = self.ctx, self.cfg
        ctx.bank, ctx.generator  # build shared lazy state once, before the stage contexts are copied
        total = self.max_runtime
        current = self.initial_assembly()
        stages = []
        for i, (name, frac_key) in enumerate(STAGES):
            left = self.time_left()
            if left < 3:
                break
            budget = left if frac_key is None else min(left, max(5.0, float(cfg.get(frac_key, 0.3)) * total))
            over = {name: {"max_runtime_s": budget}}
            if name == "column_generation":
                over[name]["init_mode"] = "anchor" if current is not None and len(current) else "columns"
            sub_ctx = dataclasses.replace(ctx, cfg=_wrap(deep_merge(ctx.cfg, over)), initial_assembly=current)
            self.phase = f"stage {i + 1}: {name}"
            stage = get_method(name)(sub_ctx, self.output_dir / f"stage{i + 1}_{name}", _StageReporter(self.reporter, name, self))
            stage.intermediate = True
            res = stage.run()
            if res.status == "cancelled":
                self.best_assembly = res.assembly
                raise _cancelled()
            current = res.assembly
            q = self.quick_eval(current)
            stages.append({"stage": name, "budget_s": round(budget, 1), "runtime_s": round(res.runtime_s, 1),
                           "objects": len(current), "loss": q["loss"], "min_view_iou": q["min_view_iou"]})
            self.info["stages"] = stages
            log.info("[chained] after %s: %d pieces, loss %.4f, min IoU %.4f", name, len(current), q["loss"], q["min_view_iou"])
            self.iteration += 1
            self.log_iteration(current, force_report=True)
        if current is None:
            current = Assembly.empty(ctx.device)
        self.best_assembly = current
        return self.finalize(current)


def _cancelled():
    from ..base import OptimizationCancelled

    return OptimizationCancelled()
