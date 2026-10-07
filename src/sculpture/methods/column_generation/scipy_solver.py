"""Open-source master solver: HiGHS through SciPy (linprog / milp)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .solver_base import LinearProgram, LPResult, MasterSolver, MILPResult


class ScipySolver(MasterSolver):
    name = "scipy-highs"

    def solve_lp(self, lp: LinearProgram, time_limit: float | None = None) -> LPResult:
        opts = {"presolve": True}
        if time_limit:
            opts["time_limit"] = float(time_limit)
        bounds = (0.0, 1.0) if lp.lb is None else np.column_stack([lp.lower(), np.ones(len(lp.c))])
        res = linprog(lp.c, A_ub=lp.A_ub, b_ub=lp.b_ub, bounds=bounds, method="highs", options=opts)
        if res.x is None:
            raise RuntimeError(f"LP failed: {res.message}")
        duals = np.asarray(res.ineqlin.marginals) if getattr(res, "ineqlin", None) is not None else None
        return LPResult("optimal" if res.status == 0 else str(res.message), float(res.fun), np.asarray(res.x), duals)

    def solve_milp(self, lp: LinearProgram, time_limit: float | None = None, gap: float | None = None) -> MILPResult:
        n = len(lp.c)
        integrality = np.zeros(n)
        integrality[: lp.n_int] = 1
        opts = {"disp": False, "presolve": True}
        if time_limit:
            opts["time_limit"] = float(time_limit)
        if gap is not None:
            opts["mip_rel_gap"] = float(gap)
        cons = LinearConstraint(lp.A_ub, -np.inf, lp.b_ub)
        res = milp(lp.c, constraints=cons, integrality=integrality, bounds=Bounds(lp.lower(), np.ones(n)), options=opts)
        if res.x is None:  # e.g. time limit before the first incumbent
            return MILPResult(f"no_solution: {res.message}", float("inf"), None, None, None)
        bound = getattr(res, "mip_dual_bound", None)
        g = getattr(res, "mip_gap", None)
        return MILPResult("optimal" if res.status == 0 else str(res.message), float(res.fun), np.asarray(res.x), bound, g)
