import pytest
import torch

from sculpture.evaluation.metrics import evaluate_assembly
from sculpture.methods.sdf_ray.optimizer import SDFRayOptimizer
from sculpture.methods.sdf_ray.rays import build_ray_sets
from sculpture.methods.sdf_ray.union_sdf import instance_sdf, softmin_st
from sculpture.scene.assembly import Assembly

from conftest import ctx_with, known_assembly, requires_cuda

SMALL = dict(max_runtime_s=40, initial_objects=12, ray_resolution=64, rays_per_view=1024, bg_rays_per_view=512,
             samples_per_ray=16, iterations=80, reseed_every=40, polish_steps=30)


@requires_cuda
def test_instance_sdf_sign_and_scaling(ctx, library):
    a = known_assembly(library)
    phi = instance_sdf(ctx.constraints.sdf, a, a.translation.clone(), margin=10.0)
    assert (phi.diagonal() < 0).all()  # object centers lie inside their objects
    # isotropic scale multiplies distances: sphere of world radius 0.5, point at distance 0.9
    sph = library.by_name("sphere").mesh_id
    b = Assembly.from_matrices(torch.tensor([sph], device="cuda"), torch.zeros(1, 3, device="cuda"),
                               torch.eye(3, device="cuda")[None], torch.tensor([0.5], device="cuda"))
    d = instance_sdf(ctx.constraints.sdf, b, torch.tensor([[0.9, 0.0, 0.0]], device="cuda"), margin=10.0)
    assert abs(d.item() - 0.4) < 0.03


def test_straight_through_min_is_exact():
    x = torch.tensor([[0.3, -0.1, 0.5], [0.2, 0.4, 0.6]], requires_grad=True)
    y = softmin_st(x, 0.05, dim=1)
    assert torch.allclose(y, torch.tensor([-0.1, 0.2]))
    y.sum().backward()
    assert (x.grad > 0).all()  # the soft gradient reaches every term


@requires_cuda
def test_ground_truth_satisfies_foreground_rays(ctx, library):
    fg, _, stats = build_ray_sets(ctx, 64)
    assert stats["fg_unsatisfiable"] < 0.05 * stats["fg_rays"]
    a = known_assembly(library)
    pts = fg.sample_points(64, stratified=False).reshape(-1, 3)
    phi = instance_sdf(ctx.constraints.sdf, a, pts, margin=0.5).amin(0).view(len(fg), 64)
    assert (phi.amin(1) < 0.02).float().mean() > 0.95


@requires_cuda
@pytest.mark.slow
def test_sdf_ray_improves(ctx, tmp_path):
    c = ctx_with(ctx, sdf_ray=SMALL)
    opt = SDFRayOptimizer(c, tmp_path)
    L0 = evaluate_assembly(c, opt.greedy_init(12, c.generator_torch(1)))["loss"]
    res = opt.run()
    m = evaluate_assembly(c, res.assembly)
    assert m["loss"] < L0
    assert m["min_view_iou"] > 0.5
