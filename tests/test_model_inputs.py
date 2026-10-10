import sys
from types import SimpleNamespace

import numpy as np
import torch

from sculpture.config import _wrap
from sculpture.constraints import ConstraintEvaluator
from sculpture.geometry.mesh_preprocess import load_mesh
from sculpture.scene.assembly import Assembly


def test_fbx_loader_merges_nodes_and_bakes_world_transform(tmp_path, monkeypatch):
    mesh = SimpleNamespace(
        vertices=np.array([[2, 3, 4], [3, 3, 4], [3, 4, 4], [2, 4, 4]], dtype=float).reshape(-1),
        indices=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32).reshape(-1),
    )
    fake = SimpleNamespace(
        Process_Triangulate=1,
        Process_PreTransformVertices=2,
        Process_JoinIdenticalVertices=4,
        import_file=lambda *_: SimpleNamespace(meshes=[mesh]),
    )
    monkeypatch.setitem(sys.modules, "assimp_py", fake)
    path = tmp_path / "quad.fbx"
    path.write_bytes(b"mock")
    result = load_mesh(path)
    assert len(result.faces) == 2
    assert np.allclose(result.bounds[0], [2, 3, 4])
    assert np.allclose(result.bounds[1], [3, 4, 4])


def test_per_model_scale_factors_affect_ranges_and_fixed_sizes():
    entries = [
        SimpleNamespace(name="small", filename="small.obj", data={"norm_scale": 2.0}),
        SimpleNamespace(name="large", filename="large.fbx", data={"norm_scale": 4.0}),
    ]
    library = SimpleNamespace(entries=entries, device=torch.device("cpu"))
    base = {
        "scale": {"min": 0.1, "max": 1.0, "mode": "free", "fixed": 0.2,
                  "native_factor": 0.5, "model_factors": {"small": 0.5, "large.fbx": 2.0}},
        "constraints": {"lambda_contain": 1, "lambda_collision": 1, "lambda_scale": 1,
                        "containment_tolerance": 0.01, "collision_margin": 0.01,
                        "hard_collisions": False, "contain_samples": 8, "collision_samples": 8},
        "bounding_volume": {"min": [-1, -1, -1], "max": [1, 1, 1]},
        "meshes": {"sdf_resolution": 8},
    }
    cons = ConstraintEvaluator(library, None, _wrap(base))
    a = Assembly(torch.tensor([0, 1]), torch.zeros(2, 3), torch.zeros(2, 6), torch.log(torch.tensor([0.05, 2.0])))
    assert cons.valid_mask(a, strict=False).tolist() == [True, True]

    base["scale"]["mode"] = "fixed"
    fixed = ConstraintEvaluator(library, None, _wrap(base))
    assert torch.allclose(torch.exp(fixed.fixed_log_scales), torch.tensor([0.1, 0.4]))

    base["scale"]["mode"] = "native"
    native = ConstraintEvaluator(library, None, _wrap(base))
    assert torch.allclose(torch.exp(native.fixed_log_scales), torch.tensor([0.125, 0.25]))


def test_model_library_selection_uses_only_requested_inputs(tmp_path, cfg):
    import trimesh

    from sculpture.geometry.mesh_library import MeshLibrary

    trimesh.creation.box().export(tmp_path / "box.obj")
    trimesh.creation.icosphere(subdivisions=1).export(tmp_path / "sphere.ply")
    lib = MeshLibrary.from_folder(
        tmp_path, cfg.meshes, device="cpu", cache_dir=tmp_path / "cache", model_names=["sphere"],
    )
    assert lib.names == ["sphere"]
