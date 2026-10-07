"""Beam state and diverse beam selection (Method 1 only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ...refinement.local import lookahead  # noqa: F401  (shared frozen-context local optimizer)
from ...scene.assembly import Assembly


@dataclass
class BeamEntry:
    assembly: Assembly
    img_loss: float           # image loss at the beam resolution (progress = 1)
    penalty: float            # constraint penalty of the whole assembly
    mask: torch.Tensor        # [V, h, w] union render at the beam resolution (detached)
    lineage: str = "root"
    history: list = field(default_factory=list)
    diversity: float = 0.0    # diversity.weight * type-distribution deficit (0 when disabled)

    @property
    def score(self) -> float:
        return self.img_loss + self.penalty + self.diversity


def diversity_term(ctx, assembly: Assembly) -> float:
    w = float(ctx.cfg.diversity.weight)
    if w <= 0 or len(assembly) == 0:
        return 0.0
    from ...loss.diversity import assembly_diversity, target_distribution

    return w * assembly_diversity(assembly.mesh_ids, target_distribution(ctx.cfg, ctx.library))["deficit"]


def make_entry(ctx, assembly: Assembly, res: int, lineage: str = "root") -> BeamEntry:
    with torch.no_grad():
        a = assembly.detach()
        R = ctx.render(a, res)
        L = float(ctx.loss(R, 1.0))
        pen, _ = ctx.constraints.penalty(a, collisions=True)
    return BeamEntry(a, L, float(pen), R, lineage, diversity=diversity_term(ctx, a))


def dilate(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Binary dilation of [..., H, W] masks by a (2px+1)^2 box."""
    if px <= 0:
        return mask
    shape = mask.shape
    m = mask.reshape(-1, 1, *shape[-2:]).float()
    return (F.max_pool2d(m, 2 * px + 1, stride=1, padding=px) > 0).reshape(shape)


def select_non_conflicting(S: torch.Tensor, order: list[int], max_adds: int, overlap: float, dilation_px: int) -> list[int]:
    """Greedy set of candidates whose (dilated) footprints do not substantially overlap in ANY view.

    S: [K, V, h, w] candidate masks; order: candidate indices by decreasing gain.
    """
    F_ = dilate(S > 0.5, dilation_px)  # [K,V,h,w]
    chosen: list[int] = []
    occupied = torch.zeros_like(F_[0])
    for c in order:
        if len(chosen) >= max_adds:
            break
        area = F_[c].sum((-1, -2)).float().clamp_min(1)
        ov = (F_[c] & occupied).sum((-1, -2)).float() / area
        if (ov <= overlap).all():
            chosen.append(c)
            occupied |= F_[c]
    return chosen


def select_diverse(pool: list[BeamEntry], width: int, iou_threshold: float) -> list[BeamEntry]:
    """Best-first selection skipping entries whose renders are near-duplicates."""
    pool = sorted(pool, key=lambda e: e.score)
    chosen: list[BeamEntry] = []
    for e in pool:
        if len(chosen) >= width:
            break
        m = e.mask > 0.5
        dup = False
        for c in chosen:
            cm = c.mask > 0.5
            inter = (m & cm).sum().item()
            union = (m | cm).sum().item()
            if union == 0 or inter / union >= iou_threshold:
                if len(e.assembly) == len(c.assembly) or union == 0:
                    dup = True
                    break
        if not dup:
            chosen.append(e)
    return chosen
