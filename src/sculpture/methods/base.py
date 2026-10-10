"""Common scaffolding for optimization methods (NOT shared search logic).

Every method subclasses OptimizationMethod and implements ``optimize()``.
The base class provides: timing / budgets, best-so-far tracking, CSV logging,
throttled progress reporting, checkpoints and cancellation.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..context import ProblemContext, set_seed
from ..evaluation.metrics import mask_metrics
from ..scene.assembly import Assembly
from ..scene.serialization import assembly_from_dict, assembly_to_dict

log = logging.getLogger(__name__)


class OptimizationCancelled(Exception):
    pass


class ProgressReporter:
    """Receives progress events. Subclass to stream to UI / logs."""

    def update(self, method: str, event: dict) -> None:  # pragma: no cover - default no-op
        pass

    def cancelled(self) -> bool:
        return False


class LoggingReporter(ProgressReporter):
    def __init__(self, every_s: float = 5.0):
        self.every_s = every_s
        self._last = 0.0

    def update(self, method, event):
        now = time.time()
        if now - self._last >= self.every_s or event.get("final"):
            self._last = now
            ious = ", ".join(f"{x:.3f}" for x in event.get("view_iou", []))
            log.info("[%s] %s it=%s t=%.1fs loss=%s iou=[%s] n=%s", method, event.get("phase"), event.get("iteration"),
                     event.get("runtime_s", 0), _fmt(event.get("loss")), ious, event.get("object_count"))


def _fmt(x):
    return "None" if x is None else f"{x:.5f}"


@dataclass
class MethodResult:
    method: str
    assembly: Assembly
    status: str                       # done | cancelled | failed
    runtime_s: float
    renderer_calls: int
    info: dict = field(default_factory=dict)


class CSVLogger:
    def __init__(self, path: Path, append: bool = False):
        self.path = Path(path)
        self.fields: list[str] | None = None
        self.rows: list[dict] = []
        if append and self.path.exists():
            with open(self.path, newline="") as f:
                self.rows = list(csv.DictReader(f))
            if self.rows:
                self.fields = list(self.rows[0].keys())

    def log(self, row: dict) -> None:
        row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
        self.rows.append(row)
        if self.fields is None or any(k not in self.fields for k in row):
            self.fields = list(dict.fromkeys((self.fields or []) + list(row.keys())))
            self._rewrite()
        else:
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, self.fields).writerow(row)

    def _rewrite(self) -> None:
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, self.fields)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)


class OptimizationMethod(ABC):
    name: str = "base"

    def __init__(self, ctx: ProblemContext, output_dir, reporter: ProgressReporter | None = None, resume: bool = False):
        self.ctx = ctx
        self.cfg = ctx.cfg[self.name]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reporter = reporter or LoggingReporter()
        self.resume = resume
        self.logger = CSVLogger(self.output_dir / "optimization.csv", append=resume)
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.best_assembly: Assembly = Assembly.empty(ctx.device)
        self.best_loss = float("inf")
        self.iteration = 0
        self.phase = "init"
        self._t0 = time.time()
        self._time_offset = 0.0
        self._calls0 = ctx.renderer.stats.calls
        self._last_report = 0.0
        self._last_ckpt = time.time()
        self.info: dict[str, Any] = {}
        # set by the chained method: run as an intermediate stage (skip the shared final stages)
        self.intermediate = False

    # ------------------------------------------------------------ budgets
    @property
    def elapsed(self) -> float:
        return time.time() - self._t0 + self._time_offset

    @property
    def max_runtime(self) -> float:
        return float(self.cfg.get("max_runtime_s", 300))

    def time_left(self) -> float:
        return self.max_runtime - self.elapsed

    def out_of_time(self, reserve: float = 0.0) -> bool:
        return self.time_left() <= reserve

    @property
    def renderer_calls(self) -> int:
        return self.ctx.renderer.stats.calls - self._calls0

    def check_cancel(self) -> None:
        if self.reporter.cancelled():
            raise OptimizationCancelled()

    @property
    def seed(self) -> int:
        return int(self.ctx.cfg.seed)

    # ------------------------------------------------------------ evaluation helpers
    @torch.no_grad()
    def quick_eval(self, assembly: Assembly, resolution: int | None = None) -> dict:
        """Loss (progress=1) and binary IoUs at the working resolution."""
        res = resolution or self.ctx.working_resolution
        R = self.ctx.render(assembly, res)
        m = mask_metrics(R, self.ctx.targets.mask(res))
        m["loss"] = float(self.ctx.loss(R, 1.0))
        return m

    def consider_best(self, assembly: Assembly, loss: float | None = None) -> bool:
        """Track best-so-far by working-resolution loss (+ constraint penalty)."""
        if loss is None:
            q = self.quick_eval(assembly)
            with torch.no_grad():
                pen, _ = self.ctx.constraints.penalty(assembly.detach(), collisions=False)
            loss = q["loss"] + float(pen)
        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_assembly = assembly.detach()
            return True
        return False

    # ------------------------------------------------------------ logging / progress
    def log_iteration(self, assembly: Assembly | None, force_report: bool = False, **fields) -> None:
        row = {"iteration": self.iteration, "phase": self.phase, "runtime_s": round(self.elapsed, 3),
               "renderer_calls": self.renderer_calls, **fields}
        q = None
        now = time.time()
        report_due = force_report or (now - self._last_report >= float(self.ctx.cfg.progress.preview_interval_s))
        if assembly is not None and ("view_iou" not in row or report_due):
            q = self.quick_eval(assembly)
            row.setdefault("loss", q["loss"])
            row.setdefault("view_iou", [round(x, 5) for x in q["view_iou"]])
            row.setdefault("object_count", len(assembly))
        self.logger.log(row)
        if report_due:
            self._last_report = now
            event = {
                "phase": self.phase, "iteration": self.iteration, "runtime_s": self.elapsed,
                "loss": row.get("loss"), "view_iou": row.get("view_iou", []),
                "object_count": row.get("object_count", len(assembly) if assembly is not None else 0),
                "best_metric": None, "info": dict(self.info),
            }
            if len(self.best_assembly) > 0 or assembly is not None:
                best_q = self.quick_eval(self.best_assembly if len(self.best_assembly) else assembly)
                event["best_metric"] = best_q["min_view_iou"]
            if assembly is not None:
                with torch.no_grad():
                    pr = int(self.ctx.cfg.progress.preview_resolution)
                    V0 = self.ctx.num_primary_views or self.ctx.num_views
                    event["preview"] = self.ctx.render(assembly.detach(), pr).cpu().numpy()[:V0]
                event["assembly"] = assembly_to_dict(assembly, self.ctx.library)
            self.reporter.update(self.name, event)
        self.maybe_checkpoint()
        self.check_cancel()

    # ------------------------------------------------------------ checkpoints
    def checkpoint_state(self) -> dict:
        """Method-specific state; override and call super()."""
        return {}

    def restore_state(self, state: dict) -> None:
        """Method-specific restore; override."""

    def maybe_checkpoint(self, force: bool = False) -> None:
        if not force and time.time() - self._last_ckpt < float(self.ctx.cfg.progress.checkpoint_interval_s):
            return
        self._last_ckpt = time.time()
        state = {
            "method": self.name,
            "iteration": self.iteration,
            "phase": self.phase,
            "elapsed": self.elapsed,
            "best_loss": self.best_loss,
            "best_assembly": assembly_to_dict(self.best_assembly, self.ctx.library),
            "info": self.info,
            # random generators are re-seeded from (seed, iteration) each round, so
            # resuming reproduces the remaining rounds without storing RNG state
            "state": self.checkpoint_state(),
        }
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, default=_json_default))
        os.replace(tmp, self.checkpoint_path)

    def load_checkpoint(self) -> bool:
        if not (self.resume and self.checkpoint_path.exists()):
            return False
        state = json.loads(self.checkpoint_path.read_text())
        self.iteration = int(state["iteration"])
        self.phase = state.get("phase", "resume")
        self._time_offset = float(state.get("elapsed", 0.0))
        self.best_loss = float(state["best_loss"])
        self.best_assembly = assembly_from_dict(state["best_assembly"], self.ctx.library, self.ctx.device)
        self.info.update(state.get("info", {}))
        self.restore_state(state.get("state", {}))
        log.info("[%s] resumed from iteration %d (best loss %.5f)", self.name, self.iteration, self.best_loss)
        return True

    def initial_assembly(self) -> Assembly | None:
        """Starting assembly from lock-and-rerun editing / a previous chained stage (or None)."""
        a = getattr(self.ctx, "initial_assembly", None)
        return None if a is None or len(a) == 0 else a.detach()

    # ------------------------------------------------------------ shared final stages
    def finalize(self, a: Assembly) -> Assembly:
        """Shared post-processing applied identically by every method:
        1. diversity type swaps (if diversity.weight > 0),
        2. intersection resolution (if constraints.resolve_intersections).
        The result is the method's final answer (it replaces best-so-far, which was
        tracked without the discrete diversity / hard collision requirements)."""
        from ..refinement.diversity import rebalance_types
        from ..refinement.feasibility import resolve_intersections

        ctx = self.ctx
        a = a.detach()
        if self.intermediate:  # chained stage: keep it feasible, leave the final stages to the chain
            if ctx.strict and len(a):
                a, valid = ctx.constraints.project_inside(a)
                a = a[valid]
            self.best_assembly = a
            return a
        if ctx.strict and len(a):
            # strict containment: project, and drop pieces that cannot be made to fit (invalid geometry)
            a, valid = ctx.constraints.project_inside(a)
            if not bool(valid.all()):
                self.info["removed_outside_hull"] = int((~valid).sum())
                a = a[valid]
        if float(ctx.cfg.diversity.weight) > 0 and ctx.cfg.diversity.get("swap_stage", True) and len(a) > 1:
            self.phase = "diversity_swaps"
            a, st = rebalance_types(ctx, a, deadline_s=max(15.0, 0.15 * self.max_runtime))
            self.info["diversity_swaps"] = st
        if ctx.constraints.hard_collisions and ctx.cfg.constraints.get("resolve_intersections", True) and len(a) > 1:
            self.phase = "resolve_intersections"
            a, st = resolve_intersections(ctx, a, int(ctx.cfg.constraints.get("resolve_rounds", 6)))
            self.info["intersections"] = st
        if ctx.strict and len(a):
            # separation may nudge a piece slightly out of Omega: project once more, drop pieces that
            # cannot fit, and re-resolve if the projection created a new contact
            a, valid = ctx.constraints.project_inside(a)
            if not bool(valid.all()):
                self.info["removed_outside_hull"] = self.info.get("removed_outside_hull", 0) + int((~valid).sum())
                a = a[valid]
            if ctx.constraints.hard_collisions and len(a) > 1 and ctx.constraints.conflicts(a)[0].shape[0]:
                a, st2 = resolve_intersections(ctx, a, 1)
                self.info["intersections_after_projection"] = st2
        self.best_assembly = a
        self.best_loss = self.quick_eval(a)["loss"]
        self.phase = "final"
        self.iteration += 1
        self.log_iteration(a, force_report=True)
        return a

    # ------------------------------------------------------------ entry point
    @abstractmethod
    def optimize(self) -> Assembly:
        """Run the search; return the final assembly."""

    def run(self) -> MethodResult:
        set_seed(self.seed)
        self._t0 = time.time()
        self._calls0 = self.ctx.renderer.stats.calls
        resumed = self.load_checkpoint()
        self.info["resumed"] = resumed
        status = "done"
        try:
            final = self.optimize()
        except OptimizationCancelled:
            status = "cancelled"
            final = self.best_assembly
        if len(final) == 0 and len(self.best_assembly) > 0:
            final = self.best_assembly
        self.maybe_checkpoint(force=True)
        self.phase = "done"
        return MethodResult(self.name, final.detach(), status, self.elapsed, self.renderer_calls, dict(self.info))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(type(o))
