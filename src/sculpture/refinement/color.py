"""Region-aware piece pruning and final visible per-instance color assignment."""

from __future__ import annotations

import numpy as np
import torch

from ..scene.assembly import Assembly
from ..targets.color import lab_to_srgb, srgb_to_lab


def _hex_rgb(value: str) -> np.ndarray:
    s = str(value).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"fixed model color must be #rrggbb, got {value!r}")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


@torch.no_grad()
def candidate_color_penalty(ctx, candidates: Assembly, masks: torch.Tensor) -> torch.Tensor:
    """Region coherence for automatic colors; target-color agreement for fixed models."""
    if ctx.targets.color_labels is None or len(candidates) == 0:
        return torch.zeros(len(candidates), device=ctx.device)
    res = masks.shape[-1]
    regions = ctx.targets.regions(res)[0]
    rgb = ctx.targets.rgb(res)[0]
    fg = ctx.targets.mask(res)[0]
    cfg = ctx.cfg.targets.color
    fixed = cfg.get("fixed_models", {})
    out = torch.zeros(len(candidates), device=ctx.device)
    for i, mid in enumerate(candidates.mesh_ids.tolist()):
        weights = masks[i, 0] * fg
        denom = weights.sum().clamp_min(1e-6)
        entry = ctx.library.entries[mid]
        value = fixed.get(entry.name, fixed.get(entry.filename))
        if value is not None:
            chosen = torch.tensor(_hex_rgb(value), device=ctx.device)
            mismatch = ((rgb - chosen) ** 2).sum(-1).sqrt() / np.sqrt(3.0)
            out[i] = float(cfg.get("fixed_placement_weight", 0.25)) * (weights * mismatch).sum() / denom
        else:
            labels = torch.unique(regions[weights > 0.1])
            labels = labels[labels >= 0]
            if len(labels) > 1:
                shares = torch.stack([(weights * (regions == label)).sum() / denom for label in labels])
                key = "coherence_weight" if ctx.targets.mask_background else "whole_image_coherence_weight"
                out[i] = float(cfg.get(key, 1.5 if not ctx.targets.mask_background else 0.12)) * (1.0 - shares.max())
    return out


@torch.no_grad()
def color_purity(ctx, assembly: Assembly, resolution: int = 128):
    """Dominant target-color-region fraction for each piece in the primary view."""
    regions = ctx.targets.regions(resolution)
    if regions is None or len(assembly) == 0:
        return torch.ones(len(assembly), device=ctx.device), torch.zeros(len(assembly), device=ctx.device)
    ids = ctx.renderer.render_instance_ids(assembly, ctx.cameras[:1], (resolution, resolution), lod="coarse")[0]
    reg = regions[0]
    purity, pixels = [], []
    for i in range(len(assembly)):
        values = reg[(ids == i) & (reg >= 0)]
        pixels.append(len(values))
        if len(values) == 0:
            purity.append(1.0)
        else:
            counts = torch.bincount(values)
            purity.append(float(counts.max()) / len(values))
    return torch.tensor(purity, device=ctx.device), torch.tensor(pixels, device=ctx.device)


@torch.no_grad()
def prune_color_incoherent(ctx, assembly: Assembly):
    """Remove broad pieces spanning unrelated palette regions so later chain stages refill them."""
    cfg = ctx.cfg.targets.color
    threshold = float(cfg.get("purity_threshold", 0.62))
    purity, pixels = color_purity(ctx, assembly, int(cfg.get("analysis_resolution", 128)))
    remove = (purity < threshold) & (pixels >= int(cfg.get("min_piece_pixels", 24))) & ~assembly.locked
    # Never erase the complete warm start: retain the most coherent visible piece.
    if len(assembly) and bool(remove.all()):
        remove[int(purity.argmax())] = False
    return assembly[~remove], {
        "removed": int(remove.sum()),
        "mean_purity": float(purity.mean()) if len(purity) else 1.0,
        "threshold": threshold,
    }


@torch.no_grad()
def assign_instance_colors(ctx, assembly: Assembly) -> tuple[Assembly, dict]:
    """Assign fixed model colors or a visibility-weighted perceptual mean per instance."""
    if len(assembly) == 0 or ctx.targets.rgb(ctx.working_resolution) is None:
        return assembly, {"assigned": 0}
    cfg = ctx.cfg.targets.color
    res = int(cfg.get("assignment_resolution", min(256, ctx.working_resolution)))
    V = min(ctx.num_primary_views or ctx.num_views, len(ctx.cameras))
    ids = ctx.renderer.render_instance_ids(assembly, ctx.cameras[:V], (res, res), lod="coarse")
    rgb = ctx.targets.rgb(res)[:V].cpu().numpy()
    masks = ctx.targets.mask(res)[:V]
    fixed = cfg.get("fixed_models", {})
    primary_w = float(cfg.get("primary_view_weight", 1.0))
    secondary_w = float(cfg.get("secondary_view_weight", 0.2))
    colors, automatic = [], 0
    for i, mid in enumerate(assembly.mesh_ids.tolist()):
        entry = ctx.library.entries[mid]
        value = fixed.get(entry.name, fixed.get(entry.filename))
        if value is not None:
            colors.append(_hex_rgb(value))
            continue
        samples, weights = [], []
        for v in range(V):
            visible = ((ids[v] == i) & (masks[v] > 0.5)).cpu().numpy()
            if visible.any():
                samples.append(srgb_to_lab(rgb[v][visible]))
                weights.append(primary_w if v == 0 else secondary_w)
        if samples:
            total = sum(w * x.sum(0) for x, w in zip(samples, weights))
            denom = sum(w * len(x) for x, w in zip(samples, weights))
            colors.append(lab_to_srgb(total / max(denom, 1e-8)))
        else:
            colors.append(np.array([0.65, 0.65, 0.65], np.float32))
        automatic += 1
    out = assembly.detach()
    out.colors = torch.tensor(np.stack(colors), dtype=torch.float32, device=ctx.device)
    return out, {"assigned": len(colors), "automatic": automatic, "fixed": len(colors) - automatic}
