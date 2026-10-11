"""One image reconstruction objective for proposal scoring, refinement and stopping."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def color_matching(ctx) -> bool:
    return (ctx.targets.color_base is not None and
            bool(ctx.cfg.targets.get("color", {}).get("enabled", True)) and
            bool(ctx.cfg.targets.get("color", {}).get("reconstruct", True)))


class ColorImageLoss:
    def __init__(self, ctx):
        self.ctx = ctx
        self.cfg = ctx.cfg.targets.color

    def view_weights(self, device):
        V = self.ctx.num_views
        weights = torch.full((V,), float(self.cfg.get("secondary_view_weight", 0.2)), device=device)
        weights[0] = float(self.cfg.get("primary_view_weight", 1.0))
        return weights / weights.sum().clamp_min(1e-8)

    @staticmethod
    def pool(x, res):
        shape = x.shape
        y = F.interpolate(x.reshape(-1, *shape[-3:]).permute(0, 3, 1, 2),
                          (res, res), mode="area").permute(0, 2, 3, 1)
        return y.reshape(*shape[:-3], res, res, shape[-1])

    def __call__(self, rgba, progress=1.0, return_terms=False):
        res = rgba.shape[-2]
        weights = self.view_weights(rgba.device)
        color = 0.0
        edge = 0.0
        levels = sorted(set([min(res, 16), min(res, 32), min(res, 64), res]))
        for level in levels:
            image = self.pool(rgba, level)
            target = self.ctx.targets.rgb(level)
            mask = self.ctx.targets.soft(level)
            # White display background is included consistently in optimization and exports.
            rendered = image[..., :3] + (1 - image[..., 3:4])
            denom = mask.sum((-1, -2)).clamp_min(1)
            error = (rendered - target).square().mean(-1)
            color = color + (((error * mask).sum((-1, -2)) / denom) * weights).sum(-1)
            if level > 1:
                dx = (rendered[..., :, 1:, :] - rendered[..., :, :-1, :]) - (target[:, :, 1:] - target[:, :, :-1])
                dy = (rendered[..., 1:, :, :] - rendered[..., :-1, :, :]) - (target[:, 1:] - target[:, :-1])
                mx = mask[:, :, 1:] * mask[:, :, :-1]
                my = mask[:, 1:] * mask[:, :-1]
                ev = (dx.square().mean(-1) * mx).sum((-1, -2)) / denom
                ev = ev + (dy.square().mean(-1) * my).sum((-1, -2)) / denom
                edge = edge + (ev * weights).sum(-1)
        color, edge = color / len(levels), edge / len(levels)
        mask = self.ctx.targets.soft(res)
        alpha = rgba[..., 3]
        coverage = (mask * (1 - alpha).square()).sum((-1, -2)) / mask.sum((-1, -2)).clamp_min(1)
        spill = ((1 - mask) * alpha.square()).sum((-1, -2)) / (1 - mask).sum((-1, -2)).clamp_min(1)
        coverage, spill = (coverage * weights).sum(-1), (spill * weights).sum(-1)
        terms = {"color": color, "edges": edge, "coverage": coverage, "spill": spill}
        total = (float(self.cfg.get("reconstruction_weight", 2.0)) * color +
                 float(self.cfg.get("edge_weight", 0.5)) * edge +
                 float(self.cfg.get("coverage_weight", 0.35)) * coverage +
                 float(self.cfg.get("spill_weight", 4.0)) * spill)
        return (total, terms) if return_terms else total

    def residual(self, rgba):
        res = rgba.shape[-2]
        mask = self.ctx.targets.soft(res)
        rendered = rgba[..., :3] + (1 - rgba[..., 3:4])
        error = (rendered - self.ctx.targets.rgb(res)).square().mean(-1)
        return mask * (error + float(self.cfg.get("coverage_weight", 0.35)) * (1 - rgba[..., 3]).square())


def render_color(ctx, assembly, res, lod="coarse"):
    return ctx.renderer.render_rgba(assembly, ctx.cameras, (res, res), lod=lod)
