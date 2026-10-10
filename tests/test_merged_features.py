"""Chained method, lock-and-rerun, viewing zone, reveal, native sizes, multi-start, MILP multi-add,
ray-meet candidate placement."""
import dataclasses
import math

import numpy as np
import pytest
import torch

from sculpture.evaluation.metrics import evaluate_assembly
from sculpture.scene.assembly import Assembly

from conftest import ctx_with, known_assembly, requires_cuda


# ------------------------------------------------------------------ lock-and-rerun
@requires_cuda
def test_locked_pieces_never_move(ctx, library):
    from sculpture.refinement.continuous import refine_assembly
    from sculpture.refinement.feasibility import resolve_intersections

    a = known_assembly(library)
    a.locked[0] = True
    # push the locked cube and an unlocked piece off their optimum: only the unlocked one may move back
    a.translation[0, 0] += 0.05
    a.translation[1, 1] += 0.05
    r = refine_assembly(ctx, a, 25)
    assert torch.allclose(r.assembly.translation[0], a.translation[0]) and bool(r.assembly.locked[0])
    assert not torch.allclose(r.assembly.translation[1], a.translation[1])
    proj, valid = ctx.constraints.project_inside(a)
    assert torch.allclose(proj.translation[0], a.translation[0]) and bool(valid[0])
    clash = Assembly.concat([a, a[1].with_params(translation=a.translation[1:2] + 0.02)])
    clash.locked[1] = True
    b, _ = resolve_intersections(ctx, clash, rounds=1, steps=5)
    assert torch.allclose(b.translation[b.locked], clash.translation[clash.locked])


@requires_cuda
def test_load_initial_assembly_lock_and_discard(ctx, library, tmp_path):
    from sculpture.context import load_initial_assembly
    from sculpture.scene.serialization import save_assembly_json

    save_assembly_json(known_assembly(library), tmp_path / "a.json", library)
    a = load_initial_assembly(ctx, tmp_path / "a.json", locked=[0, 2], keep_unlocked=True)
    assert a.locked.tolist() == [True, False, True]
    b = load_initial_assembly(ctx, tmp_path / "a.json", locked=[1], keep_unlocked=False)
    assert len(b) == 1 and bool(b.locked[0])


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("method", ["beam", "sdf_ray", "column_generation"])
def test_rerun_keeps_locked(ctx, library, tmp_path, method):
    from sculpture.methods import get_method

    small = {"beam": dict(max_runtime_s=10, beam_width=1, candidates_per_branch=64, utility_top_k=8, lookahead_top_k=4,
                          lookahead_steps=4, lookahead_resolution=64, final_refine_steps=5),
             "sdf_ray": dict(max_runtime_s=10, initial_objects=6, ray_resolution=64, iterations=30, polish_steps=5),
             "column_generation": dict(max_runtime_s=15, resolution=32, render_resolution=64, initial_columns=200, pricing_rounds=2,
                                       pricing_batch=32, pricing_steps=5, milp_time_limit_s=5, polish_steps=5)}[method]
    init = known_assembly(library)
    init.locked[:] = True
    c = dataclasses.replace(ctx_with(ctx, **{method: small}), initial_assembly=init)
    res = get_method(method)(c, tmp_path).run()
    a = res.assembly
    assert int(a.locked.sum()) == 3
    for i in range(3):  # every locked piece is still there, unchanged
        d = (a.translation[a.locked] - init.translation[i]).norm(dim=-1).min()
        assert d < 1e-5


# ------------------------------------------------------------------ chained
@requires_cuda
@pytest.mark.slow
def test_chained_runs_all_stages(ctx, tmp_path):
    from sculpture.methods.chained.optimizer import ChainedOptimizer

    c = ctx_with(ctx, chained={"max_runtime_s": 36},
                 sdf_ray=dict(initial_objects=8, ray_resolution=64, rays_per_view=512, iterations=60, polish_steps=10),
                 column_generation=dict(resolution=32, render_resolution=64, initial_columns=200, pricing_batch=32,
                                        pricing_steps=5, milp_time_limit_s=5, polish_steps=10),
                 beam=dict(beam_width=1, candidates_per_branch=64, utility_top_k=8, lookahead_top_k=4, lookahead_steps=4,
                           lookahead_resolution=64, final_refine_steps=10))
    res = ChainedOptimizer(c, tmp_path).run()
    stages = res.info["stages"]
    assert [s["stage"] for s in stages] == ["sdf_ray", "column_generation", "beam"]
    m = evaluate_assembly(c, res.assembly)
    assert m["min_view_iou"] > 0.5 and m["collisions"] == 0
    assert stages[-1]["loss"] <= stages[0]["loss"] + 0.02  # later stages do not undo earlier progress


# ------------------------------------------------------------------ viewing zone / reveal
@requires_cuda
def test_viewing_zone_cameras(cfg, library, renderer, synthetic_targets):
    import copy

    from sculpture.config import _wrap, deep_merge
    from sculpture.context import build_context

    c = _wrap(deep_merge(copy.deepcopy(cfg), {"viewing_zone": {"radius": 0.2, "samples": 3}}))
    z = build_context(c, None, synthetic_targets, library=library, renderer=renderer)
    assert z.num_primary_views == 2 and z.num_views == 2 + 2 * 3
    for k in range(3):
        cam, base = z.cameras[2 + k], z.cameras[0]
        off = cam.eye - base.eye
        assert abs(np.linalg.norm(off) - 0.2) < 1e-6 and abs(off @ base.forward()) < 1e-6
    m = evaluate_assembly(z, known_assembly(library))
    assert len(m["view_iou"]) == 2 and m["zone_min_iou"] <= m["min_view_iou"] + 1e-6


@requires_cuda
def test_reveal_cameras_and_similarity(ctx, library):
    rev = ctx.reveal
    assert rev is not None and len(rev.cameras) > 0
    prim = [(c.eye - np.asarray(c.look_at)) / np.linalg.norm(c.eye - np.asarray(c.look_at)) for c in ctx.cameras[:2]]
    for cam in rev.cameras:
        d = (cam.eye - np.asarray(cam.look_at)) / np.linalg.norm(cam.eye - np.asarray(cam.look_at))
        assert min(math.degrees(math.acos(np.clip(d @ p, -1, 1))) for p in prim) >= 24.9
    s = rev.similarity(known_assembly(library))
    assert ((s >= 0) & (s <= 1)).all()


# ------------------------------------------------------------------ native sizes / coarse-to-fine
@requires_cuda
def test_native_sizes_and_large_mask(tmp_path, cfg):
    import copy

    import trimesh

    from sculpture.config import _wrap, deep_merge
    from sculpture.constraints import ConstraintEvaluator
    from sculpture.geometry.mesh_library import MeshLibrary

    trimesh.creation.box((2.0, 2.0, 2.0)).export(tmp_path / "big.obj")
    trimesh.creation.box((0.2, 0.2, 0.2)).export(tmp_path / "small.obj")
    lib = MeshLibrary.from_folder(tmp_path, cfg.meshes, device="cuda", cache_dir=tmp_path / "cache")
    c = _wrap(deep_merge(copy.deepcopy(cfg), {"scale": {"mode": "native", "native_factor": 0.1}}))
    cons = ConstraintEvaluator(lib, None, c)
    s = torch.exp(cons.fixed_log_scales)
    assert abs(float(s[lib.by_name("big").mesh_id]) - 0.1 * math.sqrt(3)) < 1e-4
    assert abs(float(s[lib.by_name("small").mesh_id]) - 0.01 * math.sqrt(3)) < 1e-5
    assert cons.scale_fixed and cons.log_scale_lr_factor == 0.0


# ------------------------------------------------------------------ MILP multi-add / ray meet
@requires_cuda
def test_ray_meet_positions_land_in_uncovered_regions(ctx):
    I = ctx.targets.soft(64)
    U = I.clone()
    U[:, :, :32] = 0  # left half of every view already covered
    pts = ctx.generator.sample_ray_meet(400, U, ctx.generator_torch(0))
    assert pts is not None and len(pts) > 100
    from sculpture.camera.projection import project_points

    ok = torch.ones(len(pts), dtype=torch.bool, device=pts.device)
    for v, cam in enumerate(ctx.cameras):
        uv, _, _ = project_points(pts, cam, (64, 64))
        c = uv[:, 0].round().long().clamp(0, 63)
        r = uv[:, 1].round().long().clamp(0, 63)
        ok &= U[v][r, c] > 0
    assert ok.float().mean() > 0.8
