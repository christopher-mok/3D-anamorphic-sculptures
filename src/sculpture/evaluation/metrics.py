"""Final evaluation: identical renderer, resolution and metrics for every method."""

from __future__ import annotations

import torch

from ..scene.assembly import Assembly


@torch.no_grad()
def mask_metrics(R: torch.Tensor, I: torch.Tensor) -> dict:
    """Binary metrics between rendered masks R and targets I, both [V, H, W] in {0,1}."""
    Rb, Ib = R > 0.5, I > 0.5
    tp = (Rb & Ib).sum((-1, -2)).float()
    fp = (Rb & ~Ib).sum((-1, -2)).float()
    fn = (~Rb & Ib).sum((-1, -2)).float()
    nI = Ib.sum((-1, -2)).float().clamp_min(1)
    nB = (~Ib).sum((-1, -2)).float().clamp_min(1)
    iou = tp / (tp + fp + fn).clamp_min(1)
    recall = tp / nI
    spill = fp / nB
    precision = tp / (tp + fp).clamp_min(1)
    return {
        "view_iou": iou.tolist(),
        "mean_iou": float(iou.mean()),
        "min_view_iou": float(iou.min()),
        "view_recall": recall.tolist(),
        "recall": float(recall.mean()),
        "view_spill": spill.tolist(),
        "spill": float(spill.mean()),
        "view_precision": precision.tolist(),
    }


@torch.no_grad()
def evaluate_assembly(ctx, assembly: Assembly, runtime_s: float = 0.0, renderer_calls: int = 0) -> dict:
    res = int(ctx.cfg.evaluation.resolution)
    a = assembly.detach()
    # final fidelity is measured with the ORIGINAL meshes substituted back (what gets exported /
    # fabricated); the proxy result is reported alongside to expose the LOD substitution error
    lod = str(ctx.cfg.evaluation.get("lod", "original"))
    V0 = ctx.num_primary_views or ctx.num_views
    R = ctx.render(a, res, lod=lod)
    T = ctx.targets.mask(res)
    m = mask_metrics(R[:V0], T[:V0])            # the user's views
    if ctx.num_views > V0:                       # viewing zone: worst / mean over every zone view
        mz = mask_metrics(R, T)
        m["zone_min_iou"], m["zone_mean_iou"] = mz["min_view_iou"], mz["mean_iou"]
    if ctx.reveal is not None:                   # how much the sculpture resembles a target from other angles
        m["off_view_similarity"] = float(ctx.reveal.similarity(a, 128).mean())
    if lod != "proxy":
        mp = mask_metrics(ctx.render(a, res, lod="proxy")[:V0], T[:V0])
        m["proxy_min_view_iou"] = mp["min_view_iou"]
        m["lod_substitution_delta"] = m["min_view_iou"] - mp["min_view_iou"]
    m["eval_lod"] = lod
    Rw = ctx.render(a, ctx.working_resolution)
    m["loss"] = float(ctx.loss(Rw, progress=1.0))
    m.update(ctx.constraints.report(a))
    from ..loss.diversity import assembly_diversity, target_distribution, type_counts

    q = target_distribution(ctx.cfg, ctx.library)
    div = assembly_diversity(a.mesh_ids, q)
    m["randomness"] = div["randomness"]          # normalized entropy of piece-type counts
    m["diversity_deficit"] = div["deficit"]      # KL(counts || desired) / log M
    counts = type_counts(a.mesh_ids, len(q)).long().tolist()
    m["type_counts"] = {name: c for name, c in zip(ctx.library.names, counts)}
    m["object_count"] = len(a)
    m["runtime_s"] = float(runtime_s)
    m["renderer_calls"] = int(renderer_calls)
    m["eval_resolution"] = res
    return m


def selection_score(metrics: dict, cfg) -> float:
    """Scalar used to pick the best method (higher is better)."""
    kind = cfg.evaluation.selection_metric
    if kind in ("min_view_iou", "mean_iou", "recall"):
        return float(metrics[kind])
    if kind == "weighted":
        return float(sum(float(w) * float(metrics[k]) for k, w in cfg.evaluation.weighted.items()))
    raise ValueError(f"unknown selection metric {kind!r}")
