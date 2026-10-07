"""Gurobi master solver (used automatically when gurobipy + a license are present).

NOTE: not exercised by the test suite in environments without Gurobi; the
SciPy/HiGHS backend is the tested reference implementation.
"""

from __future__ import annotations

import numpy as np

from .solver_base import LinearProgram, LPResult, MasterSolver, MILPResult

import gurobipy as gp  # noqa: E402  (ImportError -> caller falls back to SciPy)
from gurobipy import GRB  # noqa: E402


class GurobiSolver(MasterSolver):
    name = "gurobi"

    def __init__(self):
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

    def _model(self, lp: LinearProgram, integer: bool):
        m = gp.Model(env=self.env)
        n = len(lp.c)
        vtype = np.full(n, GRB.CONTINUOUS)
        if integer:
            vtype[: lp.n_int] = GRB.BINARY
        v = m.addMVar(n, lb=0.0, ub=1.0, vtype=vtype.tolist(), obj=lp.c)
        cons = m.addMConstr(lp.A_ub, v, "<", lp.b_ub)
        m.ModelSense = GRB.MINIMIZE
        return m, v, cons

    def solve_lp(self, lp: LinearProgram, time_limit: float | None = None) -> LPResult:
        m, v, cons = self._model(lp, integer=False)
        if time_limit:
            m.Params.TimeLimit = float(time_limit)
        m.optimize()
        if m.SolCount == 0:
            raise RuntimeError(f"Gurobi LP failed with status {m.Status}")
        return LPResult("optimal" if m.Status == GRB.OPTIMAL else str(m.Status), float(m.ObjVal), np.asarray(v.X), np.asarray(cons.Pi))

    def solve_milp(self, lp: LinearProgram, time_limit: float | None = None, gap: float | None = None) -> MILPResult:
        m, v, _ = self._model(lp, integer=True)
        if time_limit:
            m.Params.TimeLimit = float(time_limit)
        if gap is not None:
            m.Params.MIPGap = float(gap)
        m.optimize()
        if m.SolCount == 0:
            return MILPResult(f"no_solution: status {m.Status}", float("inf"), None, None, None)
        return MILPResult("optimal" if m.Status == GRB.OPTIMAL else str(m.Status), float(m.ObjVal), np.asarray(v.X), float(m.ObjBound), float(m.MIPGap))
