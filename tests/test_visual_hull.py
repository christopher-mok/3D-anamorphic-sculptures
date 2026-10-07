import torch

from sculpture.camera.projection import project_to_pixel_index
from sculpture.loss.containment import containment_loss, surface_points

from conftest import known_assembly, requires_cuda


@requires_cuda
def test_hull_voxels_project_to_foreground(ctx):
    pts = ctx.hull.inside_centers(strict=True)
    assert len(pts) > 100
    res = ctx.targets.base_resolution
    for v, cam in enumerate(ctx.cameras):
        r, c, valid = project_to_pixel_index(pts, cam, (res, res))
        assert valid.all()
        assert ctx.targets.mask(res)[v][r, c].bool().all()


@requires_cuda
def test_hull_sdf_sign(ctx):
    inside = ctx.hull.inside_centers(strict=True)
    assert (ctx.hull.query_sdf(inside) <= 1e-6).float().mean() > 0.99
    far = torch.tensor([[0.0, 3.0, 0.0], [5.0, 5.0, 5.0], [-0.99, 0.99, -0.99]], device="cuda")
    assert (ctx.hull.query_sdf(far) > 0).all()


@requires_cuda
def test_consistent_pair_has_high_bound(ctx):
    # targets rendered from a real assembly => Omega reproduces almost all of each target
    assert min(ctx.compat.strict_coverage_upper_bound) > 0.95, ctx.compat.to_dict()
    assert max(ctx.compat.projected_outside_target) < 0.02
    assert not ctx.compat.warnings


@requires_cuda
def test_generating_assembly_is_contained(ctx, library):
    a = known_assembly(library)
    viol = ctx.constraints.containment_violation(a)
    assert (viol < 3 * ctx.hull.mean_voxel_size).all(), viol


@requires_cuda
def test_containment_penalizes_outside(ctx, library):
    a = known_assembly(library)
    inside_loss = containment_loss(a, library, ctx.hull)
    moved = a.with_params(translation=a.translation + torch.tensor([0.0, 0.0, 0.0], device="cuda"))
    moved.translation[0, 1] = 0.95  # push the cube up out of the hull
    outside_loss = containment_loss(moved, library, ctx.hull)
    assert outside_loss > inside_loss + 1e-4
    pts = surface_points(moved, library)[0]
    assert (torch.relu(ctx.hull.query_sdf(pts)) > 0).any()


@requires_cuda
def test_inconsistent_pair_warns(cfg, library, renderer, tmp_path):
    """A wide shape in view 0 and a short one in view 1 cannot both be matched."""
    import numpy as np
    from PIL import Image

    from sculpture.context import build_context

    a = np.full((256, 256), 255, np.uint8)
    a[30:226, 30:226] = 0
    b = np.full((256, 256), 255, np.uint8)
    b[110:146, 30:226] = 0
    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(b).save(tmp_path / "b.png")
    c = build_context(cfg, None, [tmp_path / "a.png", tmp_path / "b.png"], library=library, renderer=renderer)
    assert c.compat.strict_coverage_upper_bound[0] < 0.5
    assert c.compat.warnings
