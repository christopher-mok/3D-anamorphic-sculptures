"""Multi-resolution silhouette loss with a coarse-to-fine weight schedule."""

from __future__ import annotations

import torch

from ..targets.pyramids import downsample_area
from ..targets.target import TargetSet
from .silhouette import view_losses


class MultiScaleSilhouetteLoss:
    """L = sum_l lambda_l(progress) * L_image at level l.

    Renders are produced once at the finest needed resolution and area-pooled
    to every level; targets use the identical pooling (soft at coarse levels).
    lambda(progress) linearly interpolates coarse_weights -> fine_weights.
    """

    def __init__(self, targets: TargetSet, levels, coarse_weights, fine_weights, w_cov=1.0, w_neg=4.0):
        self.targets = targets
        self.levels = [int(l) for l in levels]
        self.coarse = torch.tensor(coarse_weights, dtype=torch.float32)
        self.fine = torch.tensor(fine_weights, dtype=torch.float32)
        assert len(self.levels) == len(self.coarse) == len(self.fine)
        self.w_cov, self.w_neg = float(w_cov), float(w_neg)

    @classmethod
    def from_config(cls, targets, loss_cfg, max_resolution: int | None = None):
        levels, cw, fw = list(loss_cfg.levels), list(loss_cfg.coarse_weights), list(loss_cfg.fine_weights)
        if max_resolution is not None:
            keep = [i for i, l in enumerate(levels) if l <= max_resolution]
            levels, cw, fw = [levels[i] for i in keep], [cw[i] for i in keep], [fw[i] for i in keep]
        return cls(targets, levels, cw, fw, loss_cfg.w_cov, loss_cfg.w_neg)

    @property
    def max_level(self) -> int:
        return max(self.levels)

    def weights(self, progress: float) -> torch.Tensor:
        p = float(min(max(progress, 0.0), 1.0))
        w = (1 - p) * self.coarse + p * self.fine
        return w / w.sum()

    def __call__(self, R: torch.Tensor, progress: float = 1.0, return_terms: bool = False):
        """R: [..., V, H, W] with H >= every active level. Returns loss [...]."""
        res = R.shape[-1]
        w = self.weights(progress)
        total = 0.0
        terms = {}
        wsum = 0.0
        for l, wl in zip(self.levels, w.tolist()):
            if l > res:
                continue
            Rl = downsample_area(R, l)
            Ll = view_losses(Rl, self.targets.soft(l), self.w_cov, self.w_neg)  # [..., V]
            total = total + wl * Ll.mean(-1)
            wsum += wl
            if return_terms:
                terms[l] = Ll.detach()
        total = total / max(wsum, 1e-9)
        return (total, terms) if return_terms else total

    def single(self, R: torch.Tensor) -> torch.Tensor:
        """Plain image loss at R's own resolution (soft target), [...]"""
        res = R.shape[-1]
        return view_losses(R, self.targets.soft(res), self.w_cov, self.w_neg).mean(-1)
