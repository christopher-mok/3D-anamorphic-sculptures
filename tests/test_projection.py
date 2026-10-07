import math

import numpy as np
import torch

from sculpture.camera.perspective_camera import PerspectiveCamera
from sculpture.camera.projection import pixel_rays, project_points
from sculpture.scene.assembly import Assembly

from conftest import requires_cuda


def test_center_projects_to_image_center():
    cam = PerspectiveCamera(position=(0, 0, 5), look_at=(0, 0, 0), fov_y_deg=30)
    uv, depth, front = project_points(torch.zeros(1, 3), cam, (64, 64))
    assert front.all()
    assert torch.allclose(uv[0], torch.tensor([31.5, 31.5]), atol=1e-4)
    assert abs(depth.item() - 5.0) < 1e-5


def test_up_is_up_and_right_is_right():
    cam = PerspectiveCamera(position=(0, 0, 5), look_at=(0, 0, 0), fov_y_deg=30)
    uv, _, _ = project_points(torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), cam, (64, 64))
    assert uv[0, 1] < 31.5  # +Y appears above the center (smaller row)
    assert uv[1, 0] > 31.5  # +X appears right of center for a camera on +Z


def test_perspective_closer_is_larger():
    cam = PerspectiveCamera(position=(0, 0, 5), look_at=(0, 0, 0), fov_y_deg=30)
    pts = torch.tensor([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    far_uv, _, _ = project_points(pts, cam, (128, 128))
    near_uv, _, _ = project_points(pts + torch.tensor([0, 0, 2.0]), cam, (128, 128))
    assert (near_uv[1, 1] - near_uv[0, 1]) > 1.5 * (far_uv[1, 1] - far_uv[0, 1])
    # exact pinhole relation: size_px = f * size / z
    f = cam.focal_px(128)
    assert abs((far_uv[1, 1] - far_uv[0, 1]).item() - f * 1.0 / 5.0) < 1e-3


def test_rays_reproject_to_their_pixels():
    cam = PerspectiveCamera(position=(3, 2, 4), look_at=(0.2, 0, -0.1), fov_y_deg=40)
    rows = torch.tensor([0, 10, 63])
    cols = torch.tensor([5, 40, 63])
    o, d = pixel_rays(cam, (64, 64), "cpu", rows, cols)
    pts = o + 3.0 * d
    uv, _, _ = project_points(pts, cam, (64, 64))
    assert torch.allclose(uv[:, 0], cols.float(), atol=1e-3)
    assert torch.allclose(uv[:, 1], rows.float(), atol=1e-3)


@requires_cuda
def test_render_matches_projection(renderer, cameras):
    """A small sphere rendered at a projected location lights up that pixel in both views."""
    sphere_id = renderer.library.by_name("sphere").mesh_id
    t = torch.tensor([[0.3, -0.2, 0.1]], device="cuda")
    a = Assembly.from_matrices(torch.tensor([sphere_id]), t, torch.eye(3, device="cuda")[None], torch.tensor([0.08], device="cuda"))
    masks = renderer.render_silhouettes(a, cameras, (128, 128))
    for v, cam in enumerate(cameras):
        uv, _, _ = project_points(t.cpu(), cam, (128, 128))
        c, r = int(round(uv[0, 0].item())), int(round(uv[0, 1].item()))
        assert masks[v, r, c] > 0.99
        ys, xs = torch.nonzero(masks[v] > 0.5, as_tuple=True)
        assert abs(ys.float().mean().item() - uv[0, 1].item()) < 1.0
        assert abs(xs.float().mean().item() - uv[0, 0].item()) < 1.0


@requires_cuda
def test_render_closer_is_larger(renderer, cameras):
    cube = renderer.library.by_name("cube").mesh_id
    def area(z):
        a = Assembly.from_matrices(torch.tensor([cube]), torch.tensor([[0, 0, z]], device="cuda"),
                                   torch.eye(3, device="cuda")[None], torch.tensor([0.3], device="cuda"))
        return renderer.render_silhouettes(a, cameras[:1], (128, 128)).sum().item()
    assert area(0.8) > area(-0.8) * 1.3
