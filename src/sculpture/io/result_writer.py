"""Write a method's result folder: result.json, metrics.json, assembly.glb, renders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..scene.serialization import assembly_to_dict
from .mesh_export import export_assembly


def mask_png(mask: np.ndarray, path) -> None:
    Image.fromarray(((1.0 - np.clip(mask, 0, 1)) * 255).astype(np.uint8)).save(path)


def overlay_png(render: np.ndarray, target: np.ndarray, path) -> None:
    """Gray = correct, red = missed target, blue = spill, white = background."""
    r, t = render > 0.5, target > 0.5
    img = np.full(r.shape + (3,), 255, np.uint8)
    img[r & t] = (60, 60, 60)
    img[~r & t] = (230, 60, 60)
    img[r & ~t] = (60, 110, 230)
    Image.fromarray(img).save(path)


@torch.no_grad()
def write_renders(ctx, assembly, folder, resolution: int | None = None) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    res = resolution or int(ctx.cfg.evaluation.resolution)
    V0 = ctx.num_primary_views or ctx.num_views
    R = ctx.render(assembly.detach(), res, lod=str(ctx.cfg.evaluation.get("lod", "original"))).cpu().numpy()[:V0]
    T = ctx.targets.mask(res).cpu().numpy()[:V0]
    out = []
    for v in range(R.shape[0]):
        mask_png(R[v], folder / f"view_{v}.png")
        overlay_png(R[v], T[v], folder / f"overlay_{v}.png")
        out += [folder / f"view_{v}.png", folder / f"overlay_{v}.png"]
    return out


def write_result(ctx, result, metrics: dict, folder, extra_info: dict | None = None) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": result.method,
        "status": result.status,
        "assembly": assembly_to_dict(result.assembly, ctx.library),
        "metrics": metrics,
        "info": {**result.info, **(extra_info or {})},
        "cameras": [c.to_dict() for c in ctx.cameras],
        "targets": ctx.targets.paths,
        "config": ctx.cfg.to_dict(),
    }
    (folder / "result.json").write_text(json.dumps(payload, indent=2, default=_default))
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_default))
    export_assembly(result.assembly, ctx.library, folder / "assembly.glb")
    write_renders(ctx, result.assembly, folder / "renders")
    return folder / "result.json"


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)
