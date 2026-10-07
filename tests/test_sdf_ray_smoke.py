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
def test_volumetric_overlap_term(ctx, library):
    """Soft pairwise overlap volume: > 0 when pieces interpenetrate, ~0 when apart, and its
    gradient pushes the pieces apart."""
    sph = library.by_name("sphere").mesh_id
    g = torch.Generator().manual_seed(0)
    pts = (torch.rand(20000, 3, generator=g) - 0.5).cuda() * 1.2

    def overlap(dx):
        t = torch.tensor([[-dx / 2, 0.0, 0.0], [dx / 2, 0.0, 0.0]], device="cuda", requires_grad=True)
        a = Assembly.from_matrices(torch.tensor([sph, sph], device="cuda"), t, torch.eye(3, device="cuda").expand(2, 3, 3),
                                   torch.tensor([0.25, 0.25], device="cuda"))
        occ = torch.sigmoid(-instance_sdf(ctx.constraints.sdf, a, pts, margin=0.3) / 0.01)
        s1 = occ.sum(0)
        L = (0.5 * (s1 * s1 - (occ * occ).sum(0))).mean()
        L.backward()
        return float(L), t.grad

    L_in, grad = overlap(0.2)
    L_out, _ = overlap(0.6)
    assert L_in > 1e-3 and L_out < 1e-5
    assert grad[0, 0] > 0 and grad[1, 0] < 0  # descent moves sphere 0 left and sphere 1 right


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
