"""METHOD 2: Column-generation coverage MILP.

Global combinatorial selection over a dynamically generated, finite set of
placements (columns). Loop: solve the LP relaxation -> pixel duals ->
image-space pricing weights -> differentiable (nvdiffrast) pricing of new
placements -> add columns with positive reduced profit. Collisions enter as
lazy pairwise conflict cuts. The final binary selection is solved as a MILP
and then polished with the shared continuous raster objective.

IMPORTANT: pricing is solved heuristically (non-globally). The LP objective
is therefore a rigorous bound only over the *discovered* columns, NOT a
global optimality certificate for the continuous placement space.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ...loss.silhouette import secant_utility
from ...proposals.candidate_generator import GeneratorOptions
from ...refinement.continuous import refine_assembly
from ...scene.assembly import Assembly
from ...scene.serialization import load_assembly_json
from ..base import OptimizationMethod
from .conflicts import find_conflicts
from .master import ColumnPool, MasterProblem, greedy_selection
from .pricing import price_columns
from .solver_base import make_solver

log = logging.getLogger(__name__)

NOTE = ("pricing is solved approximately (non-global); the LP bound is rigorous over discovered columns only, "
        "not a global proof for the continuous placement space")


class ColumnGenerationOptimizer(OptimizationMethod):
    name = "column_generation"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._restored: dict | None = None
        self.pool: ColumnPool | None = None
        self.master: MasterProblem | None = None

    # ------------------------------------------------------------ checkpointing
    def checkpoint_state(self) -> dict:
        if self.pool is None:
            return {}
        path = self.output_dir / "columns.npz"
        np.savez_compressed(path, **self.pool.state())
        return {"columns_file": str(path), "conflicts": sorted(self.master.conflicts) if self.master else []}

    def restore_state(self, state: dict) -> None:
        self._restored = state

    # ------------------------------------------------------------ helpers
    def _initial_columns(self, gen: torch.Generator) -> None:
        ctx, cfg, pool = self.ctx, self.cfg, self.pool
        rres = pool.rres
        I = ctx.targets.soft(rres)
        W0 = secant_utility(torch.zeros_like(I), I, ctx.loss.w_cov, ctx.loss.w_neg)
        n_total = int(cfg.initial_columns)
        added = 0
        for s in range(0, n_total, 1024):
            n = min(1024, n_total - s)
            a = ctx.generator.generate(n, W0, I, gen, GeneratorOptions(explore_probability=0.15, bank_probability=0.8))
            if ctx.strict:
                a, valid = ctx.constraints.project_inside(a)
                a = a[valid]
            added += pool.add(a, "init")
        seed = cfg.get("seed_assembly")
        if seed:
            sa = load_assembly_json(seed, ctx.library, ctx.device)
            added += pool.add(sa, "seed")
            self.info["seed_assembly"] = str(seed)
        self.info["initial_columns"] = len(pool)

    def _solve_lp_with_conflicts(self, solver):
        ctx, cfg, master, pool = self.ctx, self.cfg, self.master, self.pool
        for _ in range(int(cfg.max_conflict_rounds)):
            lp, layout = master.build()
            res = solver.solve_lp(lp, time_limit=float(cfg.lp_time_limit_s))
            if not ctx.constraints.hard_collisions:
                return lp, layout, res
            # lazy but batched: test every pair inside the LP support (not the whole pool);
            # re-solve only if a new conflict is actually violated by the LP point
            support = np.flatnonzero(res.x[: layout.C] > 1e-3)
            new = find_conflicts(ctx, pool, support, master.conflicts)
            master.conflicts.update(new)
            if not any(res.x[i] + res.x[j] > 1.0 + 1e-6 for i, j in new) or self.out_of_time():
                return lp, layout, res
        return lp, layout, res

    def _solve_milp_with_conflicts(self, solver, share: float = 0.6):
        """MILP + lazy conflict cuts. ``share`` of the remaining runtime is the budget for the
        WHOLE phase (all conflict re-solves together), not per solve."""
        import time as _time

        ctx, cfg, master, pool = self.ctx, self.cfg, self.master, self.pool
        rounds = 0
        phase_end = _time.time() + max(5.0, min(float(cfg.milp_time_limit_s), self.time_left() * share))
        while True:
            rounds += 1
            lp, layout = master.build()
            budget = max(2.0, phase_end - _time.time())
            res = solver.solve_milp(lp, time_limit=budget, gap=float(cfg.milp_gap))
            if res.x is None:  # no incumbent in time: conflict-free greedy rounding of the LP
                log.warning("[column_generation] MILP returned no solution (%s); using greedy LP rounding", res.status)
                lres = solver.solve_lp(lp, time_limit=float(cfg.lp_time_limit_s))
                x = greedy_selection(master, lres.x, master.max_objects)
                res.x = np.concatenate([x, np.zeros(len(lp.c) - len(x))])
                res.status = "greedy_fallback (" + res.status + ")"
            sel = np.flatnonzero(res.x[: layout.C] > 0.5)
            if not ctx.constraints.hard_collisions:
                break
            violated = [p for p in find_conflicts(ctx, pool, sel, master.conflicts)
                        if not (p[0] in master.fixed and p[1] in master.fixed)]
            if not violated:
                break
            master.conflicts.update(violated)
            if rounds >= int(cfg.max_conflict_rounds) or self.out_of_time() or _time.time() >= phase_end - 2.0:
                # cut budget exhausted: drop one column of each remaining colliding pair
                keep = set(int(c) for c in sel)
                for i, j in violated:
                    if i in keep and j in keep:
                        keep.discard(j if j not in master.fixed else i)
                x = np.zeros_like(res.x)
                x[np.array(sorted(keep), dtype=int)] = 1
                res.x = x
                res.status += " + conflict repair"
                break
            # cut the violated pairs plus every conflict between the selection and the pool
            master.conflicts.update(find_conflicts(ctx, pool, sel, master.conflicts, against=np.arange(len(pool))))
        return lp, layout, res, rounds

    # ------------------------------------------------------------ main
    def optimize(self) -> Assembly:
        ctx, cfg = self.ctx, self.cfg
        solver = make_solver(cfg.solver)
        self.info.update({"solver": solver.name, "note": NOTE})
        log.info("[column_generation] master solver: %s. NOTE: %s", solver.name, NOTE)
        self.pool = pool = ColumnPool(ctx, cfg.resolution, cfg.render_resolution, cfg.coverage_threshold,
                                      cfg.get("fg_coverage_threshold"), cfg.get("bg_spill_threshold"))
        self.master = master = MasterProblem(pool, ctx.loss.w_cov, ctx.loss.w_neg, int(ctx.cfg.max_objects))

        gen = ctx.generator_torch(self.seed * 7919 + 1)
        if self._restored and self._restored.get("columns_file") and Path(self._restored["columns_file"]).exists():
            with np.load(self._restored["columns_file"]) as z:
                a = Assembly(torch.tensor(z["mesh_ids"]), torch.tensor(z["translation"]), torch.tensor(z["rot6d"]), torch.tensor(z["log_scale"])).to(ctx.device)
            pool.add(a, "resumed")
            master.conflicts.update(tuple(p) for p in self._restored.get("conflicts", []))
        else:
            self.phase = "initial_columns"
            self._initial_columns(gen)
        log.info("[column_generation] %d initial columns", len(pool))

        no_improve = 0
        rounds = 0
        n_anchor = int(cfg.get("anchor_rounds", 2))
        while rounds < int(cfg.pricing_rounds):
            # reserve time for the final LP (+ conflict rounds), the MILP and the anchored rounds
            reserve = (min(float(cfg.milp_time_limit_s), 0.4 * self.max_runtime) + min(float(cfg.lp_time_limit_s), 0.15 * self.max_runtime)
                       + n_anchor * float(cfg.get("anchor_time_fraction", 0.12)) * self.max_runtime + 5)
            if self.out_of_time(reserve=reserve) or len(pool) >= int(cfg.max_columns):
                break
            self.phase = "pricing"
            added = self._pricing_round(solver, rounds)
            no_improve = no_improve + 1 if added == 0 else 0
            rounds += 1
            if no_improve >= int(cfg.patience_rounds):
                break
        self.info["pricing_rounds"] = rounds

        assembly = self._integer_assembly(solver, milp_share=float(cfg.get("milp_time_share", 0.25)))
        self.consider_best(assembly)
        self.log_iteration(assembly, force_report=True)

        # continuous polish with the shared raster loss (same objective as the other methods)
        self.phase = "raster_polish"
        if len(assembly) and int(cfg.polish_steps) > 0:
            r = refine_assembly(ctx, assembly, int(cfg.polish_steps), progress=(0.6, 1.0))
            assembly = r.assembly
            self.info["polish_initial_loss"] = r.initial_loss
            self.info["polish_final_loss"] = r.loss
        self.consider_best(assembly)
        self.iteration += 1
        self.log_iteration(self.best_assembly, force_report=True)

        # anchored re-optimization: polished pieces become fixed columns, new columns are priced
        # against what is still uncovered, and the MILP fills the holes around the anchors
        stale = 0
        for k in range(int(cfg.get("max_anchor_rounds", 6))):
            if self.out_of_time(reserve=0.05 * self.max_runtime + 3) or len(self.best_assembly) == 0:
                break
            if stale >= int(cfg.get("anchor_patience", 2)):
                break
            self.phase = f"anchored_{k + 1}"
            before = self.best_loss
            assembly = self._anchored_round(solver, self.best_assembly, k)
            self.consider_best(assembly)
            self.info.setdefault("anchored_rounds", []).append(
                {"loss_before": before, "loss_after": self.best_loss, "objects": len(self.best_assembly)})
            stale = stale + 1 if self.best_loss > before - 1e-4 else 0
            self.iteration += 1
            self.log_iteration(self.best_assembly, force_report=True)
        self.master.fixed = set()
        return self.finalize(self.best_assembly)

    # ------------------------------------------------------------ building blocks
    def _pricing_round(self, solver, rounds: int) -> int:
        """One column-generation iteration: LP -> duals -> pricing -> add columns."""
        ctx, cfg, pool, master = self.ctx, self.cfg, self.pool, self.master
        lp, layout, res = self._solve_lp_with_conflicts(solver)
        lp_obj = -res.objective
        W, sigma = master.pricing_map(lp, layout, res)
        W_t = torch.tensor(W, dtype=torch.float32, device=ctx.device).reshape(pool.V, pool.res, pool.res)
        beta = float(cfg.dual_blend)
        if beta > 0:  # exploration: blend in the residual utility of the rounded LP solution
            sel = np.flatnonzero(res.x[: layout.C] > 0.5)
            with torch.no_grad():
                R = ctx.render(pool.assembly(sel), pool.res) if len(sel) else torch.zeros_like(W_t)
                Wr = secant_utility(R, ctx.targets.soft(pool.res), ctx.loss.w_cov, ctx.loss.w_neg)
            W_t = (1 - beta) * W_t + beta * Wr
        gen = ctx.generator_torch(self.seed * 7919 + 17 * (rounds + 1) + self.iteration)
        cands, profit, masks = price_columns(ctx, W_t, sigma, int(cfg.pricing_batch), int(cfg.pricing_steps), pool.rres, gen, pool=pool)
        good = torch.nonzero(profit > float(cfg.min_reduced_profit), as_tuple=True)[0]
        good = good[torch.argsort(profit[good], descending=True)][: int(cfg.columns_per_round)]
        added = pool.add(cands[good], "pricing", masks=masks[good]) if len(good) else 0
        self.iteration += 1
        snap = pool.assembly(np.flatnonzero(res.x[: layout.C] > 0.5))  # LP rounding as a progress snapshot
        if master.fixed:  # anchored LP: not a bound for the original problem
            self.info.update({"anchored_lp_loss": master.loss_from_objective(lp_obj), "columns": len(pool), "conflicts": len(master.conflicts)})
        else:
            self.info.update({"lp_objective": lp_obj, "lp_loss_bound": master.loss_from_objective(lp_obj),
                              "columns": len(pool), "conflicts": len(master.conflicts)})
        self.log_iteration(snap, lp_objective=lp_obj, lp_loss_bound=master.loss_from_objective(lp_obj),
                           columns=len(pool), added=added, best_profit=float(profit.max()) if len(profit) else 0.0, sigma=sigma)
        return added

    def _integer_assembly(self, solver, milp_share: float) -> Assembly:
        """Final LP bound, MILP (+ greedy LP rounding as second incumbent) -> assembly."""
        ctx, pool, master = self.ctx, self.pool, self.master
        self.phase = "final_lp"
        lp, layout, res = self._solve_lp_with_conflicts(solver)
        lp_obj = -res.objective
        self.phase = "milp"
        lp, layout, mres, conflict_rounds = self._solve_milp_with_conflicts(solver, share=milp_share)
        sel = np.flatnonzero(mres.x[: layout.C] > 0.5)
        int_obj = master.objective_value(mres.x[: layout.C])
        # second incumbent: conflict-free greedy rounding of the final LP; keep the better one
        # (a time-limited MILP incumbent can be far from optimal on large instances)
        gx = greedy_selection(master, res.x[: layout.C], master.max_objects)
        if ctx.constraints.hard_collisions:
            for i, j in find_conflicts(ctx, pool, np.flatnonzero(gx > 0.5), set()):
                if gx[i] > 0.5 and gx[j] > 0.5 and not (i in master.fixed and j in master.fixed):
                    gx[j if j not in master.fixed else i] = 0.0
        g_obj = master.objective_value(gx)
        incumbent = "milp"
        if g_obj > int_obj:
            sel, int_obj, incumbent = np.flatnonzero(gx > 0.5), g_obj, "greedy_lp_rounding"
        if master.fixed:  # anchored LP/MILP: not a bound for the original problem -> separate keys
            self.info.update({"anchored_lp_loss": master.loss_from_objective(lp_obj),
                              "anchored_integer_loss": master.loss_from_objective(int_obj), "anchored_incumbent": incumbent})
            log.info("[column_generation] anchored: LP loss %.5f, integer loss %.5f (%s), %d columns selected",
                     master.loss_from_objective(lp_obj), master.loss_from_objective(int_obj), incumbent, len(sel))
            return pool.assembly(sel)
        self.info.update({
            "lp_objective": lp_obj, "lp_loss_bound": master.loss_from_objective(lp_obj),
            "integer_objective": int_obj, "integer_loss": master.loss_from_objective(int_obj),
            "greedy_objective": g_obj, "incumbent": incumbent,
            "milp_status": mres.status, "milp_gap": mres.gap, "columns": len(pool),
            "conflicts": len(master.conflicts), "conflict_rounds": conflict_rounds, "selected": int(len(sel)),
            "milp_resolution": pool.res,
        })
        log.info("[column_generation] LP obj %.5f (loss bound %.5f over discovered columns), integer obj %.5f (%s), %d/%d columns selected",
                 lp_obj, master.loss_from_objective(lp_obj), int_obj, incumbent, len(sel), len(pool))
        return pool.assembly(sel)

    def _anchored_round(self, solver, incumbent: Assembly, k: int) -> Assembly:
        """Fix the polished pieces as columns (x = 1), price new columns against what is still
        uncovered, re-solve, and polish the additions."""
        ctx, cfg, pool, master = self.ctx, self.cfg, self.pool, self.master
        n0 = len(pool)
        added = pool.add(incumbent, f"anchor{k}", dedupe=False)
        anchors = list(range(n0, n0 + added))
        master.fixed = set(anchors)
        if ctx.constraints.hard_collisions:
            master.conflicts.update(find_conflicts(ctx, pool, np.array(anchors), master.conflicts, against=np.arange(len(pool))))
        for r in range(int(cfg.get("anchor_pricing_rounds", 2))):
            if self.out_of_time(reserve=0.08 * self.max_runtime + 3):
                break
            self.phase = f"anchored_{k + 1}_pricing"
            self._pricing_round(solver, 1000 * (k + 1) + r)
        assembly = self._integer_assembly(solver, milp_share=float(cfg.get("milp_time_share", 0.25)))
        steps = int(cfg.get("anchor_polish_steps", max(20, int(cfg.polish_steps) // 2)))
        if len(assembly) > len(incumbent) and steps > 0:
            assembly = refine_assembly(ctx, assembly, steps, progress=(0.8, 1.0), lr_scale=0.5).assembly
        return assembly
