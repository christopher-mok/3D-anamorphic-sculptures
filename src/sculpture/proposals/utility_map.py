"""Image-space utility maps W_v(p) used to score proposals.

W > 0: adding occupancy at p helps;  W < 0: it hurts.
"""

from __future__ import annotations

import torch

from ..loss.silhouette import gradient_utility, secant_utility


@torch.no_grad()
def utility_map(R: torch.Tensor, I: torch.Tensor, w_cov: float, w_neg: float, mode: str = "secant") -> torch.Tensor:
    """mode="gradient": W = -dL/dR (first-order, as in the spec);
    mode="secant":   exact change for binary union compositing (default; the
                     gradient form gives zero spill penalty on empty background
                     because d(R^2)/dR = 0 at R = 0)."""
    if mode == "gradient":
        return gradient_utility(R, I, w_cov, w_neg)
    if mode == "secant":
        return secant_utility(R, I, w_cov, w_neg)
    raise ValueError(mode)


def approximate_utility(W: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """U(c) = sum_v sum_p W_v(p) S_c^v(p).  W [V,H,W], S [N,V,H,W] -> [N]."""
    return torch.einsum("vhw,nvhw->n", W, S)
