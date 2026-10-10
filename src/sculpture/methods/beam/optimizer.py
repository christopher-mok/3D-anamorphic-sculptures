"""METHOD 1: Hull-constrained beam constructive search.

Grow assemblies from empty by repeatedly adding the mesh instance that most
improves the target projections. Several partial assemblies compete in a
beam. Proposals: residual-biased positions in Omega + silhouette-bank
mesh/orientation retrieval + distance-transform scale. Screening: cheap
utility U(c) = sum W S_c, exact union loss, then a short local lookahead
(only the candidate moves). Periodic global joint refinement and repair moves.
"""

from __future__ import annotations

import logging

import torch

from ...loss.silhouette import secant_utility, soft_union
from ...proposals.candidate_generator import GeneratorOptions
from ...proposals.residual import uncovered
from ...proposals.utility_map import approximate_utility
from ...refinement.continuous import refine_assembly
from ...scene.assembly import Assembly
from ...scene.serialization import assembly_from_dict, assembly_to_dict
from ..base import OptimizationMethod
from .beam import BeamEntry, diversity_term, lookahead, make_entry, select_diverse, select_non_conflicting
from .repair import repair_assembly

log = logging.getLogger(__name__)


class BeamSearchOptimizer(OptimizationMethod):
    name = "beam"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam: list[BeamEntry] = []
        self.stall = 0
        self._restored: dict | None = None

    # ------------------------------------------------------------ checkpointing
    def checkpoint_state(self) -> dict:
        return {
            "stall": self.stall,
            "beam": [{"assembly": assembly_to_dict(e.assembly, self.ctx.library), "lineage": e.lineage} for e in self.beam],
        }

    def restore_state(self, state: dict) -> None:
        self._restored = state

    # ------------------------------------------------------------ expansion
    def expand(self, entry: BeamEntry, gen: torch.Generator) -> list[BeamEntry]:
        ctx, cfg = self.ctx, self.cfg
        rc, rl = int(cfg.candidate_resolution), int(cfg.lookahead_resolution)
        room = int(ctx.cfg.max_objects) - len(entry.assembly)
        if room <= 0:
            return []
        I_c = ctx.targets.soft(rc)
        with torch.no_grad():
            R_c = ctx.render(entry.assembly, rc)
            W = secant_utility(R_c, I_c, ctx.loss.w_cov, ctx.loss.w_neg)
            U = uncovered(R_c, I_c)
            if U.sum() < 0.5:  # nothing left to cover
                return []
            opts = GeneratorOptions(explore_probability=float(cfg.explore_probability), bank_probability=float(cfg.bank_probability))
            cands = ctx.generator.generate(int(cfg.candidates_per_branch), W, U, gen, opts)
            # 1) cheap first-order ranking
            S = ctx.render_instances(cands, rc, lod="coarse")
            util = approximate_utility(W, S)
            top = util.topk(min(int(cfg.utility_top_k), len(cands))).indices
            # 2) exact union loss on the survivors
            L_exact = ctx.loss(soft_union(R_c[None], S[top]), 1.0)
            keep = top[L_exact.topk(min(int(cfg.lookahead_top_k), len(top)), largest=False).indices]
            R_l = ctx.render(entry.assembly, rl)
        # 3) local lookahead: optimize only the candidate, parent frozen
        opt, L, pen, Umask = lookahead(ctx, entry.assembly, R_l, cands[keep], int(cfg.lookahead_steps), rl)
        if ctx.strict:  # hard feasibility: project into Omega, re-score, reject the rest
            opt, valid = ctx.constraints.project_inside(opt)
            with torch.no_grad():
                Umask = soft_union(R_l[None], ctx.render_instances(opt, rl))
                L = ctx.loss(Umask, 1.0)
                pen = ctx.constraints.candidate_penalty(opt, fixed=entry.assembly)
            score = torch.where(valid, L + pen, torch.full_like(L, float("inf")))
        else:
            score = L + pen
        if ctx.constraints.hard_collisions and len(entry.assembly) > 0:  # reject candidates intersecting the parent
            pairs, _ = ctx.constraints.conflicts(opt, entry.assembly)
            if pairs.shape[0]:
                score[pairs[:, 0].unique()] = float("inf")
        improves = torch.isfinite(score) & (score < entry.img_loss)  # must improve the image
        w_div = float(ctx.cfg.diversity.weight)
        if w_div > 0:  # rank by image score + diversity deficit of the resulting child
            from ...loss.diversity import diversity_stats, target_distribution, type_counts

            q = target_distribution(ctx.cfg, ctx.library)
            counts = type_counts(entry.assembly.mesh_ids, len(q))[None].repeat(len(opt), 1)
            counts[torch.arange(len(opt)), opt.mesh_ids] += 1
            score = score + w_div * diversity_stats(counts, q)["deficit"]
        order = [i for i in torch.argsort(score).tolist() if improves[i]]
        children = []
        for i in order[: int(cfg.children_per_parent)]:
            child = Assembly.concat([entry.assembly, opt[i]])
            with torch.no_grad():
                pen_full, _ = ctx.constraints.penalty(child, collisions=True)
            children.append(BeamEntry(child, float(L[i]), float(pen_full), Umask[i], entry.lineage + f">a{len(child)}",
                                      diversity=diversity_term(ctx, child)))
        # 4) non-conflicting multi-add
        if cfg.multi_add and len(order) >= 2 and room >= 2:
            with torch.no_grad():
                S_opt = ctx.render_instances(opt, rl)
            n_max = min(int(cfg.multi_add_max), room)
            chosen = select_non_conflicting(S_opt, order, n_max, float(cfg.multi_add_overlap), int(cfg.multi_add_dilation_px))
            if len(chosen) >= 2:
                idx = torch.tensor(chosen, device=ctx.device)
                child = Assembly.concat([entry.assembly, opt[idx]])
                with torch.no_grad():
                    union = R_l
                    for c in chosen:
                        union = soft_union(union, S_opt[c])
                    Lm = float(ctx.loss(union, 1.0))
                    pen_full, _ = ctx.constraints.penalty(child, collisions=True)
                children.append(BeamEntry(child, Lm, float(pen_full), union, entry.lineage + f">m{len(chosen)}",
                                          diversity=diversity_term(ctx, child)))
        return children

    # ------------------------------------------------------------ global refinement + repair
    def refine_and_repair(self, gen: torch.Generator) -> None:
        ctx, cfg = self.ctx, self.cfg
        rl = int(cfg.lookahead_resolution)
        new_beam = []
        stats_total = {}
        # Only the best `refine_entries` entries are refined/repaired: profiling showed a full
        # joint refinement costs ~3 growth rounds while adding little beyond what growth adds.
        k_ref = int(cfg.get("refine_entries", 1))
        ranked = sorted(self.beam, key=lambda e: e.score)
        for rank, e in enumerate(ranked):
            if len(e.assembly) == 0 or rank >= k_ref:
                new_beam.append(e)
                continue
            self.phase = "global_refine"
            r = refine_assembly(ctx, e.assembly, int(cfg.global_refine_steps), progress=(0.3, 1.0))
            a = r.assembly
            if cfg.repair:
                self.phase = "repair"
                a, stats = repair_assembly(ctx, a, cfg, gen, rl, int(cfg.candidate_resolution), int(cfg.lookahead_steps))
                for k, v in stats.items():
                    stats_total[k] = stats_total.get(k, 0) + v
            ne = make_entry(ctx, a, rl, e.lineage + ">r")
            new_beam.append(ne if ne.score <= e.score else e)
        self.beam = new_beam
        for k, v in stats_total.items():
            self.info[f"repair_{k}"] = self.info.get(f"repair_{k}", 0) + v

    # ------------------------------------------------------------ main loop
    def _stop_reason(self) -> str | None:
        cfg = self.cfg
        if self.out_of_time():
            return "max_runtime"
        if self.renderer_calls >= int(cfg.max_renderer_calls):
            return "max_renderer_calls"
        if self.stall >= int(cfg.patience_rounds):
            return "no_improvement"
        if all(len(e.assembly) >= int(self.ctx.cfg.max_objects) for e in self.beam):
            return "max_objects"
        q = self.quick_eval(self.beam[0].assembly)
        if q["min_view_iou"] >= float(cfg.target_iou):
            return "target_iou"
        return None

    def optimize(self) -> Assembly:
        ctx, cfg = self.ctx, self.cfg
        rl = int(cfg.lookahead_resolution)
        if self._restored and self._restored.get("beam"):
            self.beam = [make_entry(ctx, assembly_from_dict(b["assembly"], ctx.library, ctx.device), rl, b.get("lineage", "resumed")) for b in self._restored["beam"]]
            self.stall = int(self._restored.get("stall", 0))
        else:
            root = self.initial_assembly()
            self.beam = [make_entry(ctx, root if root is not None else Assembly.empty(ctx.device), rl, "initial" if root is not None else "root")]
        self.info.update({"beam_width": int(cfg.beam_width)})
        stop = None
        while True:
            stop = self._stop_reason()
            if stop:
                break
            gen = ctx.generator_torch(self.seed * 100003 + self.iteration)
            self.phase = "growth"
            prev_best = self.beam[0].score
            children = []
            for e in self.beam:
                children += self.expand(e, gen)
                self.check_cancel()
            self.beam = select_diverse(children + self.beam, int(cfg.beam_width), float(cfg.diversity_iou))
            self.iteration += 1
            every = int(cfg.global_refine_every)
            stalled = self.beam and prev_best - min(e.score for e in self.beam) < float(cfg.min_improvement)
            if every > 0 and (self.iteration % every == 0 or (stalled and cfg.get("refine_on_stall", True))):
                self.refine_and_repair(gen)
                self.beam.sort(key=lambda e: e.score)
            best = self.beam[0]
            self.stall = self.stall + 1 if prev_best - best.score < float(cfg.min_improvement) else 0
            self.consider_best(best.assembly)
            self.phase = "growth"
            self.log_iteration(best.assembly, score=best.score, img_loss=best.img_loss, penalty=best.penalty,
                               beam_sizes=[len(e.assembly) for e in self.beam], children=len(children), stall=self.stall)
        self.info["stop_reason"] = stop
        # final polish of the best beam entry with the shared raster objective
        self.phase = "final_refine"
        best = min(self.beam, key=lambda e: e.score).assembly
        if len(best) > 0 and int(cfg.final_refine_steps) > 0:
            best = refine_assembly(ctx, best, int(cfg.final_refine_steps), progress=(0.6, 1.0)).assembly
        self.consider_best(best)
        self.log_iteration(self.best_assembly, force_report=True)
        return self.finalize(self.best_assembly)
