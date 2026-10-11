from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sculpture.config import _wrap, load_config
from sculpture.loss.color_image import ColorImageLoss
from sculpture.targets.target import TargetSet


def test_color_objective_distinguishes_images_with_identical_silhouettes():
    rgb = np.zeros((32, 32, 3), np.float32)
    rgb[:, :16, 0] = 1
    rgb[:, 16:, 2] = 1
    targets = TargetSet([np.ones((32, 32), bool)], "cpu", colors=[rgb])
    ctx = SimpleNamespace(targets=targets, num_views=1, cfg=_wrap({"targets": {"color": {}}}))
    objective = ColorImageLoss(ctx)
    perfect = torch.cat([torch.tensor(rgb)[None], torch.ones(1, 32, 32, 1)], -1)
    swapped = perfect.flip(2)
    solid = perfect.clone()
    solid[..., :3] = torch.tensor([0.5, 0, 0.5])
    empty = torch.zeros_like(perfect)
    assert float(objective(perfect)) < 1e-7
    assert float(objective(swapped)) > float(objective(solid)) > float(objective(perfect))
    white = perfect.clone()
    white[..., :3] = 1
    assert float(objective(white)) < float(objective(empty))
    variable = solid.detach().requires_grad_(True)
    objective(variable).backward()
    assert torch.isfinite(variable.grad).all() and variable.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_depth_tested_color_renderer_has_geometry_and_color_gradients(library, renderer, cameras):
    from sculpture.scene.assembly import Assembly

    a = Assembly.from_matrices([0, 0], [[0, 0, 0.3], [0, 0, -0.3]],
                              torch.eye(3).repeat(2, 1, 1), [0.3, 0.3]).to("cuda")
    a.colors = torch.tensor([[1., 0, 0], [0, 0, 1.]], device="cuda")
    p = a.params()
    p.colors.requires_grad_(True)
    rgba = renderer.render_rgba(p, cameras[:1], (64, 64), lod="coarse")
    assert rgba[0, 32, 32, 0] > 0.99 and rgba[0, 32, 32, 2] < 0.01
    x = torch.arange(64, device="cuda").float()[None, None, :, None]
    (rgba * x).sum().backward()
    assert torch.isfinite(p.translation.grad).all() and p.translation.grad.abs().sum() > 0
    assert p.colors.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_chained_full_color_adds_pieces_despite_diversity_and_reduces_error(tmp_path, library, renderer):
    from sculpture.context import build_context
    from sculpture.methods.chained.optimizer import ChainedOptimizer

    rgb = np.zeros((64, 64, 3), np.uint8)
    rgb[:, :32] = (230, 30, 20)
    rgb[:, 32:] = (20, 40, 230)
    path = tmp_path / "two_colors.png"
    Image.fromarray(rgb).save(path)
    cfg = load_config("fast", overrides={
        "max_objects": 12,
        "targets": {"mask_background": False, "working_resolution": 64},
        "hull": {"resolution": 32, "compatibility_resolution": 64},
        "diversity": {"weight": 0.3},
        "chained": {"max_runtime_s": 12},
        "beam": {"candidates_per_branch": 96, "lookahead_top_k": 6, "global_refine_every": 4,
                 "global_refine_steps": 4, "final_refine_steps": 4},
        "silhouette_bank": {"num_orientations": 16},
        "constraints": {"resolve_intersections": False},
    })
    ctx = build_context(cfg, None, [path], library=library, renderer=renderer)
    result = ChainedOptimizer(ctx, tmp_path / "run").run()
    assert len(result.assembly) >= 2
    assert result.assembly.colors is not None
    assert result.info["final_image_loss"] < result.info["initial_image_loss"] * 0.9
    assert result.assembly.colors[:, 0].max() - result.assembly.colors[:, 0].min() > 0.4
    assert result.info["objective"] == "multiscale_color_edges_coverage"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_refinement_preserves_fixed_colors_and_locked_geometry(circle_ctx, tmp_path):
    import dataclasses
    from sculpture.config import deep_merge
    from sculpture.methods.color_constructive import ColorConstructiveOptimizer
    from sculpture.scene.assembly import Assembly

    ctx = dataclasses.replace(circle_ctx, cfg=_wrap(deep_merge(circle_ctx.cfg,
        {"targets": {"color": {"fixed_models": {"cube": "#00ff00"}}}})))
    a = Assembly.from_matrices([ctx.library.by_name("cube").mesh_id, ctx.library.by_name("sphere").mesh_id],
                              [[-0.15, 0, 0], [0.15, 0, 0]], torch.eye(3).repeat(2, 1, 1), [0.1, 0.1]).to("cuda")
    a.colors = torch.tensor([[0., 1, 0], [0.3, 0.2, 0.1]], device="cuda")
    a.locked[1] = True
    method = ColorConstructiveOptimizer(ctx, tmp_path / "fixed")
    method.objective = ColorImageLoss(ctx)
    refined = method.refine(a, 3, 64)
    assert torch.equal(refined.colors, a.colors)
    assert torch.equal(refined.translation[1], a.translation[1])
    assert torch.equal(refined.log_scale[1], a.log_scale[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_collision_repairs_preserve_instance_colors(circle_ctx):
    from sculpture.refinement.feasibility import resolve_intersections
    from sculpture.scene.assembly import Assembly

    cube = circle_ctx.library.by_name("cube").mesh_id
    a = Assembly.from_matrices([cube, cube], [[0, 0, 0], [0.1, 0, 0]],
                              torch.eye(3).repeat(2, 1, 1), [0.2, 0.2]).to("cuda")
    a.colors = torch.tensor([[1., 0, 0], [0, 0, 1.]], device="cuda")
    a.locked[0] = True
    repaired, stats = resolve_intersections(circle_ctx, a, rounds=0)
    assert repaired.colors is not None
    assert torch.equal(repaired.colors[0], a.colors[0])
    assert torch.equal(repaired.translation[0], a.translation[0])
    assert stats["final_conflicts"] == 0
