"""Fixed scale, unbounded view axis, piece-type diversity, intersection handling."""
import math

import pytest
import torch

from sculpture.geometry.collisions import reference_intersections
from sculpture.geometry.rotation import matrix_to_rot6d, random_rotations
from sculpture.loss.diversity import assembly_diversity, diversity_stats
from sculpture.refinement.continuous import refine_assembly
from sculpture.refinement.diversity import rebalance_types
from sculpture.refinement.feasibility import resolve_intersections
from sculpture.scene.assembly import Assembly

from conftest import ctx_with, known_assembly, requires_cuda


def _crowded(library, n=30, seed=0, spread=0.35, scale=0.22):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, len(library), (n,), generator=g)
    t = (torch.rand(n, 3, generator=g) - 0.5) * 2 * spread
    R = random_rotations(n, g)
    return Assembly(ids.cuda(), t.cuda(), matrix_to_rot6d(R).cuda(), torch.full((n,), math.log(scale)).cuda())


# ------------------------------------------------------------------ intersections
@requires_cuda
def test_conservative_detector_contains_reference(ctx, library):
    """Every truly interpenetrating pair (dense winding-number reference) is flagged."""
    for seed in range(3):
        a = _crowded(library, seed=seed)
        ref = set(reference_intersections(a, library))
        assert ref, "test assembly should contain intersections"
        det, _ = ctx.constraints.conflicts(a)
        det = {tuple(sorted(p)) for p in det.tolist()}
        assert ref <= det, ref - det


@requires_cuda
def test_resolve_intersections_reaches_zero(ctx, library):
    a = _crowded(library, n=24, seed=5, spread=0.3, scale=0.2)
    assert reference_intersections(a, library)
    b, stats = resolve_intersections(ctx, a, rounds=3, steps=20)
    assert stats["final_conflicts"] == 0
    assert ctx.constraints.conflicts(b)[0].shape[0] == 0
    assert reference_intersections(b, library) == []
    assert len(b) >= len(a) // 2  # mostly separated / shrunk, not deleted


@requires_cuda
def test_report_counts_reference_intersections(ctx, library):
    a = known_assembly(library)
    clash = Assembly.concat([a, a[0].with_params(translation=a.translation[:1] + 0.05)])
    rep = ctx.constraints.report(clash)
    assert rep["collisions"] >= 1 and rep["clearance_violations"] >= rep["collisions"]


# ------------------------------------------------------------------ fixed scale
@requires_cuda
def test_fixed_scale_never_changes(ctx, library):
    c = ctx_with(ctx, scale={"mode": "fixed", "fixed": 0.2})
    from sculpture.constraints import ConstraintEvaluator

    cons = ConstraintEvaluator(c.library, c.hull, c.cfg)
    cons.camera_eye = c.cameras[0].torch_eye("cuda")
    import dataclasses

    c = dataclasses.replace(c, constraints=cons, _generator=None)
    I = c.targets.soft(64)
    cands = c.generator.generate(64, I, I, c.generator_torch(0))
    assert torch.allclose(cands.scales(), torch.full((64,), 0.2, device="cuda"))
    proj, valid = cons.project_inside(cands)
    assert torch.allclose(proj.scales(), torch.full((64,), 0.2, device="cuda"), atol=1e-6)
    assert valid.float().mean() > 0.3
    r = refine_assembly(c, proj[valid][:6], steps=15)
    assert torch.allclose(r.assembly.scales(), torch.full((len(r.assembly),), 0.2, device="cuda"), atol=1e-5)


@requires_cuda
def test_fixed_scale_depth_from_desired_size(ctx):
    """Smaller uncovered regions -> candidates placed farther from the camera."""
    c = ctx_with(ctx, scale={"mode": "fixed", "fixed": 0.15})
    from sculpture.constraints import ConstraintEvaluator

    cons = ConstraintEvaluator(c.library, c.hull, c.cfg)
    import dataclasses

    c = dataclasses.replace(c, constraints=cons, _generator=None)
    gen = c.generator
    pos = torch.zeros(2, 3, device="cuda")
    view = torch.zeros(2, dtype=torch.long, device="cuda")
    z = torch.full((2,), 5.0, device="cuda")
    new, z_new = gen._slide_to_depth(pos, view, z, torch.tensor([4.5, 5.5], device="cuda"))
    assert z_new[0] < z_new[1] and new[0, 2] > new[1, 2]  # camera 0 sits on +z
    assert torch.allclose(new[:, :2], pos[:, :2], atol=1e-5)  # stays on the same camera ray


# ------------------------------------------------------------------ unbounded view axis
@requires_cuda
def test_unbounded_view_axis_single_view(cfg, library, renderer, circle_ctx):
    import copy

    from sculpture.config import _wrap, deep_merge
    from sculpture.context import build_context

    c2 = _wrap(deep_merge(copy.deepcopy(cfg), {"bounding_volume": {"unbounded_view_axis": True, "view_axis_near": 2.0, "view_axis_far": 10.0}}))
    ctx_u = build_context(c2, None, circle_ctx.targets.paths, library=library, renderer=renderer)
    bmin, bmax = ctx_u.hull.bmin, ctx_u.hull.bmax
    assert bmax[2] >= 2.9 and bmin[2] <= -4.9          # camera at z=5: depths 2..10 -> z in [-5, 3]
    assert ctx_u.cameras[0].far >= 10.0
    # the hull is much deeper than the default cube
    inside = ctx_u.hull.inside_centers(strict=True)
    assert inside[:, 2].min() < -2.0 and inside[:, 2].max() > 1.5
    assert ctx_u.compat.strict_coverage_upper_bound[0] > 0.95


# ------------------------------------------------------------------ diversity
def test_diversity_stats():
    q = torch.ones(4) / 4
    uni = diversity_stats(torch.tensor([5.0, 5.0, 5.0, 5.0]), q)
    one = diversity_stats(torch.tensor([20.0, 0.0, 0.0, 0.0]), q)
    assert abs(float(uni["randomness"]) - 1.0) < 1e-6 and abs(float(uni["deficit"])) < 1e-6
    assert abs(float(one["randomness"])) < 1e-6 and abs(float(one["deficit"]) - 1.0) < 1e-6
    skew = diversity_stats(torch.tensor([10.0, 6.0, 3.0, 1.0]), q)
    assert 0 < float(skew["deficit"]) < 1 and 0 < float(skew["randomness"]) < 1


@requires_cuda
@pytest.mark.slow
def test_rebalance_types_increases_randomness(ctx, library):
    from sculpture.loss.diversity import target_distribution

    c = ctx_with(ctx, diversity={"weight": 0.3, "max_swaps": 12, "swap_steps": 8})
    # all-cube assembly covering the targets reasonably
    cube = library.by_name("cube").mesh_id
    g = torch.Generator().manual_seed(1)
    I = c.targets.soft(64)
    cands = c.generator.generate(256, I, I, c.generator_torch(3))
    a = cands[cands.mesh_ids == cands.mesh_ids.mode().values][:10]
    a = Assembly(torch.full((len(a),), cube, device="cuda"), a.translation, a.rot6d, a.log_scale)
    q = target_distribution(c.cfg, library)
    before = assembly_diversity(a.mesh_ids, q)["randomness"]
    b, stats = rebalance_types(c, a)
    after = assembly_diversity(b.mesh_ids, q)["randomness"]
    assert stats["swaps"] >= 1 and after > before
