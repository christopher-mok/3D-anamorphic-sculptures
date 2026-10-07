"""Assembly <-> JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .assembly import Assembly


def assembly_to_dict(assembly: Assembly, library=None) -> dict:
    a = assembly.detach()
    R = a.rotation_matrices().cpu().numpy()
    objs = []
    for i, o in enumerate(a.to_instances()):
        entry = library.entries[o.mesh_id] if library is not None else None
        objs.append(
            {
                "mesh_id": int(o.mesh_id),
                "mesh_name": entry.name if entry else None,
                "source_file": entry.filename if entry else None,
                "translation": [float(x) for x in o.translation],
                "rotation_matrix": [[float(x) for x in row] for row in R[i]],
                "rotation6d": [float(x) for x in o.rotation6d],
                "log_scale": float(o.log_scale),
                "scale": float(np.exp(o.log_scale)),
            }
        )
    out = {"objects": objs, "object_count": len(objs)}
    if library is not None:
        out["mesh_names"] = library.names
    return out


def assembly_from_dict(d: dict, library=None, device="cuda") -> Assembly:
    objs = d["objects"] if isinstance(d, dict) else d
    if not objs:
        return Assembly.empty(device)
    ids = []
    for o in objs:
        mid = int(o["mesh_id"])
        if library is not None and o.get("mesh_name"):
            try:  # robust to a changed model folder ordering
                mid = library.by_name(o["mesh_name"]).mesh_id
            except KeyError:
                pass
        ids.append(mid)
    t = torch.tensor([o["translation"] for o in objs], dtype=torch.float32, device=device)
    if "rotation6d" in objs[0]:
        r6 = torch.tensor([o["rotation6d"] for o in objs], dtype=torch.float32, device=device)
    else:
        from ..geometry.rotation import matrix_to_rot6d

        r6 = matrix_to_rot6d(torch.tensor([o["rotation_matrix"] for o in objs], dtype=torch.float32, device=device))
    if "log_scale" in objs[0]:
        ls = torch.tensor([o["log_scale"] for o in objs], dtype=torch.float32, device=device)
    else:
        ls = torch.log(torch.tensor([o["scale"] for o in objs], dtype=torch.float32, device=device))
    return Assembly(torch.tensor(ids, device=device), t, r6, ls)


def save_assembly_json(assembly: Assembly, path, library=None, extra: dict | None = None) -> None:
    d = assembly_to_dict(assembly, library)
    if extra:
        d.update(extra)
    Path(path).write_text(json.dumps(d, indent=2))


def load_assembly_json(path, library=None, device="cuda") -> Assembly:
    d = json.loads(Path(path).read_text())
    if "assembly" in d:  # result.json
        d = d["assembly"]
    return assembly_from_dict(d, library, device)
