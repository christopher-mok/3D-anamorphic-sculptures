"""Lazy collision conflicts x_i + x_j <= 1 between columns."""

from __future__ import annotations

import numpy as np

from .master import ColumnPool


def find_conflicts(ctx, pool: ColumnPool, selected: np.ndarray, known: set[tuple[int, int]], against: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Geometric collisions not yet constrained.

    against=None: pairs within ``selected``.
    against=idx : pairs (i in selected, j in idx), e.g. the selected columns vs.
                  the whole pool, which anticipates the solver's next choices.
    """
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) == 0:
        return []
    a = pool.assembly(selected)
    if against is None:
        if len(selected) < 2:
            return []
        pairs, _ = ctx.constraints.conflicts(a)
        other = selected
    else:
        other = np.asarray(against, dtype=np.int64)
        pairs, _ = ctx.constraints.conflicts(a, pool.assembly(other))
    out = set()
    for i, j in pairs.cpu().numpy():
        ci, cj = int(selected[i]), int(other[j])
        if ci == cj:
            continue
        key = (min(ci, cj), max(ci, cj))
        if key not in known:
            out.add(key)
    return sorted(out)
