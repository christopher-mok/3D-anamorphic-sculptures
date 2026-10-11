"""Constructive color reconstruction with exact visible-image acceptance.

Proposal masks are screened cheaply. Every accepted addition and every refined
assembly is evaluated with the same depth-tested RGB/edge/coverage objective.
Segmentation guides proposals only; pieces may cross boundaries when beneficial.
"""

from __future__ import annotations

import torch

from .base import OptimizationMethod
from ..loss.color_image import ColorImageLoss, render_color
from ..proposals.candidate_generator import GeneratorOptions
from ..refinement.color import _hex_rgb, assign_instance_colors
from ..scene.assembly import Assembly


class ColorConstructiveOptimizer(OptimizationMethod):
    name = "beam"  # existing quality presets and time budget apply

    @torch.no_grad()
    def candidate_colors(self, cands, masks):
        ctx = self.ctx
        res = masks.shape[-1]
        target = ctx.targets.rgb(res)
        weight = masks * ctx.targets.soft(res)[None]
        vw = self.objective.view_weights(ctx.device)[None, :, None, None]
        weight = weight * vw
        colors = (weight[..., None] * target[None]).sum((1, 2, 3)) / weight.sum((1, 2, 3))[:, None].clamp_min(1e-6)
        fixed = ctx.cfg.targets.color.get("fixed_models", {})
        for i, mid in enumerate(cands.mesh_ids.tolist()):
            entry = ctx.library.entries[mid]
            value = fixed.get(entry.name, fixed.get(entry.filename))
            if value is not None:
                colors[i] = torch.tensor(_hex_rgb(value), device=ctx.device)
        cands.colors = colors.clamp(0, 1)
        return cands

    def refine(self, assembly, steps, res):
        if not len(assembly) or steps <= 0:
            return assembly
        ctx = self.ctx
        p = assembly.params()
        p.colors.requires_grad_(True)
        opt = torch.optim.Adam([
            {"params": [p.translation], "lr": float(ctx.cfg.refine.lr_translation)},
            {"params": [p.rot6d], "lr": float(ctx.cfg.refine.lr_rotation)},
            {"params": [p.log_scale], "lr": float(ctx.cfg.refine.lr_log_scale) * ctx.constraints.log_scale_lr_factor},
            {"params": [p.colors], "lr": 0.025},
        ])
        fixed = ctx.cfg.targets.color.get("fixed_models", {})
        free_color = torch.tensor([ctx.library.entries[m].name not in fixed and ctx.library.entries[m].filename not in fixed
                                   for m in p.mesh_ids.tolist()], device=ctx.device) & ~p.locked
        original_colors = p.colors.detach().clone()
        best, best_loss = assembly.detach(), float(self.objective(render_color(ctx, assembly, res)))
        for _ in range(steps):
            self.check_cancel()
            if self.out_of_time():
                break
            image_loss = self.objective(render_color(ctx, p, res))
            penalty, _ = ctx.constraints.penalty(p, collisions=True)
            opt.zero_grad(set_to_none=True)
            (image_loss + penalty).backward()
            for parameter in (p.translation, p.rot6d, p.log_scale):
                if parameter.grad is not None:
                    parameter.grad[p.locked] = 0
            if p.colors.grad is not None:
                p.colors.grad[~free_color] = 0
            opt.step()
            with torch.no_grad():
                p.colors.clamp_(0, 1)
                p.colors[~free_color] = original_colors[~free_color]
                if ctx.constraints.scale_fixed:
                    p.log_scale[~p.locked] = ctx.constraints.fixed_log_scales[p.mesh_ids[~p.locked]]
                projected, valid = ctx.constraints.project_inside(p.detach()) if ctx.strict else (p.detach(), ctx.constraints.valid_mask(p, strict=False))
                p.translation.copy_(projected.translation)
                p.log_scale.copy_(projected.log_scale)
                feasible = bool(valid.all())
                if feasible and ctx.constraints.hard_collisions:
                    feasible = ctx.constraints.conflicts(projected)[0].numel() == 0
                score = float(self.objective(render_color(ctx, projected, res)))
                if feasible and score < best_loss:
                    best, best_loss = projected, score
        return best.normalized_rotations()

    @torch.no_grad()
    def add_piece(self, parent, res, gen, focus=None, max_pixels=None):
        ctx, cfg = self.ctx, self.cfg
        image = render_color(ctx, parent, res)
        before = float(self.objective(image))
        residual = self.objective.residual(image)
        if focus is not None:
            residual = residual * focus
        # Include residual color mismatch even when silhouette coverage is already complete.
        W = residual - 4 * (1 - ctx.targets.soft(res)) / (res * res)
        cands = ctx.generator.generate(int(cfg.candidates_per_branch), W, residual, gen,
                                       GeneratorOptions(explore_probability=float(cfg.explore_probability),
                                                        bank_probability=float(cfg.bank_probability)))
        if ctx.strict:
            cands, valid = ctx.constraints.project_inside(cands)
        else:
            valid = ctx.constraints.valid_mask(cands, strict=False)
        cands = cands[valid]
        if not len(cands):
            return parent, before, 0
        S = ctx.render_instances(cands, res, lod="coarse")
        if max_pixels is not None:
            keep = (S[:, 0].sum((-1, -2)) <= max_pixels) & (S[:, 0].sum((-1, -2)) > 0.5)
            cands, S = cands[keep], S[keep]
            if not len(cands):
                return parent, before, 0
        cands = self.candidate_colors(cands, S)
        # Screening approximation assumes candidates are in front. Acceptance below is
        # always depth-tested so a hidden piece cannot claim an image improvement.
        approximate = torch.cat([image[None, ..., :3] * (1 - S[..., None]) + S[..., None] * cands.colors[:, None, None, None],
                                 (image[None, ..., 3] + (1 - image[None, ..., 3]) * S)[..., None]], -1)
        scores = self.objective(approximate)
        indices = torch.argsort(scores)[:min(int(cfg.lookahead_top_k), len(cands))]
        # Diversity breaks near ties only; it cannot make an empty image beat useful coverage.
        if float(ctx.cfg.diversity.weight) > 0:
            counts = torch.bincount(parent.mesh_ids, minlength=len(ctx.library)).float()
            gain = (before - scores[indices]).clamp_min(0)
            bonus = float(ctx.cfg.diversity.weight) * 0.1 * gain / (1 + counts[cands.mesh_ids[indices]])
            indices = indices[torch.argsort(scores[indices] - bonus)]
        best, best_score, best_rank, accepted = parent, before, before, 0
        for i in indices.tolist():
            self.check_cancel()
            if self.out_of_time():
                break
            piece = cands[i]
            if ctx.constraints.hard_collisions and len(parent) and ctx.constraints.conflicts(piece, parent)[0].numel():
                continue
            child = Assembly.concat([parent, piece])
            exact = float(self.objective(render_color(ctx, child, res)))
            count = int((parent.mesh_ids == piece.mesh_ids[0]).sum())
            bonus = float(ctx.cfg.diversity.weight) * 0.1 * max(0, before - exact) / (1 + count)
            rank = exact - bonus
            if exact < before - 1e-7 and rank < best_rank:
                best, best_score, best_rank, accepted = child, exact, rank, 1
        return best, best_score, accepted

    @torch.no_grad()
    def split_piece(self, parent, res, gen):
        """Replace a broad, wrong-colored piece with several smaller useful pieces.

        Growth alone cannot correct a solid footprint once collision constraints prevent
        another piece from being placed in front. A replacement is committed only when its
        exact visible-image error improves on the complete original assembly.
        """
        ctx = self.ctx
        if not len(parent) or len(parent) + 3 > int(ctx.cfg.max_objects):
            return parent
        ids = ctx.renderer.render_instance_ids(parent, ctx.cameras, (res, res), lod="coarse")
        rgb = ctx.targets.rgb(res)
        mask = ctx.targets.soft(res)
        errors = []
        for i in range(len(parent)):
            visible = (ids == i) * mask
            area = visible[0].sum()
            error = ((rgb - parent.colors[i]).square().mean(-1) * visible).sum()
            errors.append(error if not parent.locked[i] and area >= 12 else error * 0)
        order = torch.argsort(torch.stack(errors), descending=True)
        before = float(self.objective(render_color(ctx, parent, res)))
        for i in order[:2].tolist():
            if float(errors[i]) <= 0.05 or self.out_of_time():
                break
            focus = (ids == i).float()
            pixels = float(focus[0].sum())
            trial = parent.without(i)
            for _ in range(4):
                if self.out_of_time():
                    break
                trial, _, added = self.add_piece(trial, res, gen, focus=focus, max_pixels=0.55 * pixels)
                if not added:
                    break
            if float(self.objective(render_color(ctx, trial, res))) < before - 1e-7:
                self.info["region_splits"] = self.info.get("region_splits", 0) + 1
                return trial
        return parent

    def optimize(self):
        ctx, cfg = self.ctx, self.cfg
        self.objective = ColorImageLoss(ctx)
        res = int(cfg.lookahead_resolution)
        current = self.best_assembly if len(self.best_assembly) else self.initial_assembly()
        current = current if current is not None else Assembly.empty(ctx.device)
        if current.colors is None:
            if len(current):
                current, _ = assign_instance_colors(ctx, current)
            else:
                current.colors = torch.empty(0, 3, device=ctx.device)
        initial_loss = float(self.objective(render_color(ctx, current, res)))
        self.info.update({"objective": "multiscale_color_edges_coverage", "initial_image_loss": initial_loss})
        self.best_assembly = current.detach()
        self.best_loss = initial_loss
        stall = 0
        stop = "max_runtime"
        while not self.out_of_time():
            if len(current) >= int(ctx.cfg.max_objects):
                stop = "max_objects"
                break
            if self.renderer_calls >= int(cfg.max_renderer_calls):
                stop = "max_renderer_calls"
                break
            self.phase = "color_growth"
            previous = float(self.objective(render_color(ctx, current, res)))
            gen = ctx.generator_torch(self.seed * 100003 + self.iteration)
            current, _, added = self.add_piece(current, res, gen)
            self.iteration += 1
            if len(current) and (not added or self.iteration % 12 == 0):
                self.phase = "color_split"
                current = self.split_piece(current, res, gen)
            every = int(cfg.global_refine_every)
            if len(current) and (not added or (every > 0 and self.iteration % every == 0)):
                self.phase = "color_refine"
                current = self.refine(current, min(20, int(cfg.global_refine_steps)), res)
            score = float(self.objective(render_color(ctx, current, res)))
            # Tiny pieces still reset patience when they actually improve the image.
            stall = stall + 1 if previous - score <= 1e-7 else 0
            self.best_assembly, self.best_loss = current.detach(), score
            self.log_iteration(current, image_loss=score, added=added, stall=stall)
            if stall >= int(cfg.patience_rounds):
                stop = "no_color_improvement"
                break
        self.phase = "color_polish"
        current = self.refine(current, min(40, int(cfg.final_refine_steps)), min(256, ctx.working_resolution))
        self.best_assembly = current
        final_loss, terms = self.objective(render_color(ctx, current, res), return_terms=True)
        self.info.update({"stop_reason": stop, "final_image_loss": float(final_loss),
                          "image_terms": {k: float(v) for k, v in terms.items()}})
        self.log_iteration(current, force_report=True)
        return self.finalize(current)
