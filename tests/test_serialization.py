import json

import torch
import trimesh

from sculpture.io.mesh_export import export_assembly
from sculpture.scene.assembly import Assembly
from sculpture.scene.serialization import assembly_from_dict, assembly_to_dict, load_assembly_json, save_assembly_json

from conftest import known_assembly, requires_cuda


@requires_cuda
def test_json_roundtrip_preserves_transforms(library, tmp_path):
    a = known_assembly(library)
    save_assembly_json(a, tmp_path / "a.json", library)
    d = json.loads((tmp_path / "a.json").read_text())
    assert d["objects"][0]["mesh_name"] == "cube" and len(d["objects"]) == 3
    b = load_assembly_json(tmp_path / "a.json", library)
    assert torch.equal(a.mesh_ids, b.mesh_ids)
    assert torch.allclose(a.translation, b.translation, atol=1e-6)
    assert torch.allclose(a.rotation_matrices(), b.rotation_matrices(), atol=1e-5)
    assert torch.allclose(a.scales(), b.scales(), atol=1e-6)
    # a rotation matrix + scale alone (no rot6d / log_scale) is also accepted
    for o in d["objects"]:
        o.pop("rotation6d")
        o.pop("log_scale")
    c = assembly_from_dict(d, library)
    assert torch.allclose(a.rotation_matrices(), c.rotation_matrices(), atol=1e-5)
    assert torch.allclose(a.scales(), c.scales(), atol=1e-5)


@requires_cuda
def test_glb_export_keeps_instances(library, tmp_path):
    a = known_assembly(library)
    export_assembly(a, library, tmp_path / "a.glb")
    scene = trimesh.load(tmp_path / "a.glb")
    assert len(scene.graph.nodes_geometry) == 3
    # the exported cube instance sits where its transform says
    node = [n for n in scene.graph.nodes_geometry if "cube" in n][0]
    T, g = scene.graph[node]
    verts = trimesh.transform_points(scene.geometry[g].vertices, T)
    assert abs(verts.mean(0) - a.translation[0].cpu().numpy()).max() < 0.05


def test_empty_assembly_roundtrip():
    a = Assembly.empty("cpu")
    assert len(assembly_from_dict(assembly_to_dict(a), device="cpu")) == 0
