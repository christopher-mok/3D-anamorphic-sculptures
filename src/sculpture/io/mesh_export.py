"""Export an assembly as GLB/GLTF.

Each instance becomes its own scene node referencing the canonical mesh with
its world transform, so per-instance transforms are retained for fabrication
while any viewer displays the sculpture in world space. ``bake=True`` instead
writes world-space vertices for every instance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from ..scene.assembly import Assembly

_PALETTE = np.array(
    [[0.85, 0.55, 0.35], [0.35, 0.6, 0.85], [0.55, 0.8, 0.45], [0.85, 0.75, 0.35], [0.7, 0.5, 0.85], [0.4, 0.8, 0.8], [0.85, 0.45, 0.6]]
)


def assembly_to_scene(assembly: Assembly, library, lod: str = "original", bake: bool = False) -> trimesh.Scene:
    scene = trimesh.Scene()
    geoms = {}
    for i, o in enumerate(assembly.detach().to_instances()):
        e = library.entries[o.mesh_id]
        color_key = tuple(float(x) for x in assembly.colors[i].cpu()) if assembly.colors is not None else None
        geom_key = (e.name, color_key)
        if geom_key not in geoms:
            m = e.canonical_trimesh(lod)
            rgb = assembly.colors[i].cpu().numpy() if assembly.colors is not None else _PALETTE[o.mesh_id % len(_PALETTE)]
            color = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile(np.append(color, 255), (len(m.faces), 1)))
            geoms[geom_key] = m
        T = o.world_matrix()
        node = f"obj_{i:04d}_{e.name}"
        if bake:
            g = geoms[geom_key].copy()
            g.apply_transform(T)
            scene.add_geometry(g, node_name=node, geom_name=node)
        else:
            scene.add_geometry(geoms[geom_key], node_name=node, geom_name=f"{e.name}_{i:04d}", transform=T)
    return scene


def export_assembly(assembly: Assembly, library, path, lod: str = "original", bake: bool = False) -> Path:
    path = Path(path)
    if len(assembly) == 0:  # trimesh cannot export empty scenes; write a minimal valid glTF
        import json as _json
        gltf = {"asset": {"version": "2.0"}, "scenes": [{"nodes": []}], "scene": 0}
        Path(path).with_suffix(".gltf").write_text(_json.dumps(gltf))
        return Path(path).with_suffix(".gltf")
    assembly_to_scene(assembly, library, lod, bake).export(path)
    return path


def export_canonical_mesh(library, mesh_id: int, path, lod: str = "proxy") -> Path:
    m = library.entries[mesh_id].canonical_trimesh(lod)
    m.export(path)
    return Path(path)
