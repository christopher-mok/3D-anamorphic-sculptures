import numpy as np
import pytest
import torch

from sculpture.evaluation.metrics import evaluate_assembly
from sculpture.loss.silhouette import image_loss
from sculpture.methods.column_generation.master import ColumnPool, MasterProblem, greedy_selection
from sculpture.methods.column_generation.optimizer import ColumnGenerationOptimizer
from sculpture.methods.column_generation.scipy_solver import ScipySolver

from conftest import ctx_with, known_assembly, requires_cuda

SMALL = dict(max_runtime_s=40, resolution=32, render_resolution=64, initial_columns=400, pricing_rounds=3,
             pricing_batch=64, pricing_steps=10, columns_per_round=32, milp_time_limit_s=15, polish_steps=30, solver="scipy")


@requires_cuda
def test_master_objective_equals_image_loss(ctx, library):
    """For binary masks the master objective is exactly w_cov - L_image at the MILP resolution."""
    pool = ColumnPool(ctx, 32, 64)
    pool.add(known_assembly(library), "gt")
    master = MasterProblem(pool, 1.0, 4.0, 10)
    x = np.ones(len(pool))
    union = torch.zeros(pool.P, dtype=torch.bool)
    for r in pool.rows:
        union[torch.as_tensor(r)] = True
    R = union.float().reshape(2, 32, 32).to(ctx.device)
    L = image_loss(R, ctx.targets.mask(32), 1.0, 4.0).item()
    assert abs(master.loss_from_objective(master.objective_value(x)) - L) < 1e-6


@requires_cuda
def test_lp_bound_and_greedy_incumbent(ctx):
    pool = ColumnPool(ctx, 32, 64)
    I = ctx.targets.soft(64)
    pool.add(ctx.generator.generate(200, I, I, ctx.generator_torch(0)), "test")
    master = MasterProblem(pool, 1.0, 4.0, 50)
    lp, layout = master.build()
    solver = ScipySolver()
    lres = solver.solve_lp(lp)
    mres = solver.solve_milp(lp, time_limit=20)
    lp_obj, int_obj = -lres.objective, master.objective_value(mres.x[: layout.C])
    greedy_obj = master.objective_value(greedy_selection(master, lres.x, 50))
    assert int_obj <= lp_obj + 1e-6          # LP relaxation bounds the integer optimum
    assert greedy_obj <= int_obj + 1e-6      # greedy is a feasible (worse or equal) incumbent
    W, sigma = master.pricing_map(lp, layout, lres)
    assert W.shape == (pool.P,) and sigma >= 0


@requires_cuda
@pytest.mark.slow
def test_column_generation_improves(ctx, tmp_path):
    c = ctx_with(ctx, column_generation=SMALL)
    res = ColumnGenerationOptimizer(c, tmp_path).run()
    info = res.info
    assert info["columns"] >= info["initial_columns"]
    assert info["lp_loss_bound"] <= info["integer_loss"] + 1e-6
    m = evaluate_assembly(c, res.assembly)
    assert m["loss"] < 0.6 and m["min_view_iou"] > 0.5
