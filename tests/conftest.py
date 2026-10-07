import pytest
import torch
import trimesh

from sculpture.config import load_config

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (nvdiffrast)")


@pytest.fixture(scope="session")
def cfg():
    return load_config("fast")


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    trimesh.creation.box((1.0, 1.0, 1.0)).export(d / "cube.obj")
    trimesh.creation.icosphere(subdivisions=3, radius=0.5).export(d / "sphere.ply")
    trimesh.creation.box((1.6, 0.25, 0.25)).export(d / "rod.stl")
    return d


@pytest.fixture(scope="session")
def library(model_dir, cfg, tmp_path_factory):
    from sculpture.geometry.mesh_library import MeshLibrary

    return MeshLibrary.from_folder(model_dir, cfg.meshes, device="cuda", cache_dir=tmp_path_factory.mktemp("cache"))


@pytest.fixture(scope="session")
def renderer(library):
    from sculpture.rendering.nvdiffrast_renderer import NvdiffrastRenderer

    return NvdiffrastRenderer(library, "cuda")


@pytest.fixture(scope="session")
def cameras(cfg):
    from sculpture.camera.perspective_camera import cameras_from_config

    return cameras_from_config(cfg.cameras, 2)


def known_assembly(library, device="cuda"):
    """A small feasible assembly used to synthesize consistent two-view targets."""
    from sculpture.geometry.rotation import axis_angle_to_matrix
    from sculpture.scene.assembly import Assembly

    ids = torch.tensor([library.by_name("cube").mesh_id, library.by_name("sphere").mesh_id, library.by_name("rod").mesh_id], device=device)
    t = torch.tensor([[-0.35, 0.3, 0.1], [0.35, -0.25, -0.2], [0.0, 0.0, 0.3]], device=device)
    R = axis_angle_to_matrix(torch.tensor([[0.3, 0.5, 0.1], [0.0, 0.0, 0.0], [0.2, -0.4, 0.7]], device=device))
    s = torch.tensor([0.35, 0.3, 0.45], device=device)
    return Assembly.from_matrices(ids, t, R, s)


@pytest.fixture(scope="session")
def synthetic_targets(renderer, library, cameras, tmp_path_factory):
    """Two target PNGs rendered from known_assembly (guaranteed geometrically consistent)."""
    import numpy as np
    from PIL import Image

    d = tmp_path_factory.mktemp("targets")
    with torch.no_grad():
        m = renderer.render_silhouettes(known_assembly(library), cameras, (512, 512))
    paths = []
    for v in range(2):
        p = d / f"view_{v}.png"
        Image.fromarray(((m[v] < 0.5).cpu().numpy() * 255).astype(np.uint8)).save(p)
        paths.append(p)
    return paths


@pytest.fixture(scope="session")
def ctx(cfg, model_dir, synthetic_targets, library, renderer):
    from sculpture.context import build_context

    return build_context(cfg, model_dir, synthetic_targets, library=library, renderer=renderer)


def ctx_with(ctx, **sections):
    """Copy of a ProblemContext with config sections overridden (e.g. beam={...})."""
    import copy
    import dataclasses

    from sculpture.config import _wrap, deep_merge

    cfg = _wrap(deep_merge(copy.deepcopy(ctx.cfg), sections))
    return dataclasses.replace(ctx, cfg=cfg)


@pytest.fixture(scope="session")
def circle_ctx(cfg, library, renderer, tmp_path_factory):
    """Single-view circle target."""
    from PIL import Image, ImageDraw

    from sculpture.context import build_context

    d = tmp_path_factory.mktemp("circle")
    img = Image.new("L", (256, 256), 255)
    ImageDraw.Draw(img).ellipse([70, 70, 186, 186], fill=0)
    img.save(d / "circle.png")
    return build_context(cfg, None, [d / "circle.png"], library=library, renderer=renderer)
