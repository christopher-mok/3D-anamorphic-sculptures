import torch

from sculpture.geometry.rotation import matrix_to_rot6d, random_rotations
from sculpture.scene.assembly import Assembly

from conftest import requires_cuda


def _assembly(library, n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.arange(n) % len(library)
    t = (torch.rand(n, 3, generator=g) - 0.5) * 0.8
    R = random_rotations(n, g)
    return Assembly(ids.cuda(), t.cuda(), matrix_to_rot6d(R).cuda(), torch.log(torch.full((n,), 0.3)).cuda())


@requires_cuda
def test_union_render_gradients(renderer, cameras, library):
    a = _assembly(library).params()
    target = torch.zeros(2, 64, 64, device="cuda")
    target[:, 20:44, 20:44] = 1
    R = renderer.render_silhouettes(a, cameras, (64, 64))
    loss = ((R - target) ** 2).mean()
    loss.backward()
    for name, g in [("translation", a.translation.grad), ("rotation", a.rot6d.grad), ("scale", a.log_scale.grad)]:
        assert g is not None and torch.isfinite(g).all(), name
        assert g.abs().sum() > 0, f"zero gradient for {name}"


@requires_cuda
def test_instance_render_gradients_and_consistency(renderer, cameras, library):
    a = _assembly(library, n=5, seed=1).params()
    S = renderer.render_instance_silhouettes(a, cameras, (64, 64))
    assert S.shape == (5, 2, 64, 64)
    S.sum().backward()
    assert a.translation.grad.abs().sum() > 0
    assert a.rot6d.grad.abs().sum() > 0
    assert a.log_scale.grad.abs().sum() > 0
    # union of binary instance masks == binary union render
    with torch.no_grad():
        U = renderer.render_silhouettes(a, cameras, (64, 64))
        Si = renderer.render_instance_silhouettes(a, cameras, (64, 64))
        diff = ((Si > 0.5).any(0).float() - (U > 0.5).float()).abs().mean()
    assert diff < 0.01


@requires_cuda
def test_scale_gradient_sign(renderer, cameras, library):
    """Growing an object increases covered area: d(area)/d(log_scale) > 0."""
    a = Assembly.from_matrices(torch.tensor([0], device="cuda"), torch.zeros(1, 3, device="cuda"),
                               torch.eye(3, device="cuda")[None], torch.tensor([0.3], device="cuda")).params()
    renderer.render_silhouettes(a, cameras, (64, 64)).sum().backward()
    assert a.log_scale.grad.item() > 0


@requires_cuda
def test_translation_gradient_direction(renderer, cameras, library):
    """Target to the right of the object in view 0 -> gradient pushes +x."""
    a = Assembly.from_matrices(torch.tensor([1], device="cuda"), torch.zeros(1, 3, device="cuda"),
                               torch.eye(3, device="cuda")[None], torch.tensor([0.2], device="cuda")).params()
    target = torch.zeros(1, 64, 64, device="cuda")
    target[0, 24:40, 36:52] = 1
    R = renderer.render_silhouettes(a, cameras[:1], (64, 64))
    ((R - target) ** 2).sum().backward()
    assert a.translation.grad[0, 0] < 0  # gradient descent increases x
