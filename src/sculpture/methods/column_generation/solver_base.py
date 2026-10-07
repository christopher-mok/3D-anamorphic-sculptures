"""MasterSolver interface: LP relaxation with duals + binary MILP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class LinearProgram:
    """minimize c^T v  s.t.  A_ub v <= b_ub,  0 <= v <= 1;  first n_int vars binary in the MILP."""

    c: np.ndarray
    A_ub: sp.csr_matrix
    b_ub: np.ndarray
    n_int: int
    row_groups: dict  # name -> slice of rows (coverage, spill, cardinality, conflicts)


@dataclass
class LPResult:
    status: str
    objective: float          # minimization objective
    x: np.ndarray
    duals: np.ndarray | None  # marginals d(obj)/d(b_ub)  (<= 0 for <= rows)


@dataclass
class MILPResult:
    status: str
    objective: float
    x: np.ndarray
    bound: float | None       # best dual bound reported by the solver
    gap: float | None


class MasterSolver(ABC):
    name = "base"

    @abstractmethod
    def solve_lp(self, lp: LinearProgram, time_limit: float | None = None) -> LPResult: ...

    @abstractmethod
    def solve_milp(self, lp: LinearProgram, time_limit: float | None = None, gap: float | None = None) -> MILPResult: ...


def make_solver(kind: str = "auto") -> MasterSolver:
    if kind in ("auto", "gurobi"):
        try:
            from .gurobi_solver import GurobiSolver

            return GurobiSolver()
        except Exception as exc:
            if kind == "gurobi":
                raise RuntimeError(f"Gurobi requested but unavailable: {exc}") from exc
    from .scipy_solver import ScipySolver

    return ScipySolver()
