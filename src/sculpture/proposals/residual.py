"""Residual maps: what part of each target is still uncovered."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def uncovered(R: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
    """U_v = I_v * (1 - R_v)."""
    return I * (1 - R)


def spill(R: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
    return (1 - I) * R


@torch.no_grad()
def inscribed_radius_map(U: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Per-pixel distance (px) to the nearest pixel NOT in {U > threshold}: [V,H,W]."""
    out = []
    for u in (U > threshold).cpu().numpy():
        if u.any():
            out.append(ndimage.distance_transform_edt(np.pad(u, 1))[1:-1, 1:-1])
        else:
            out.append(np.zeros(u.shape))
    return torch.tensor(np.stack(out), dtype=torch.float32, device=U.device)
