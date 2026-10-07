"""Image pyramids. The loss uses area (box) downsampling applied identically to
renders and targets; Gaussian pyramids are provided for analysis/proposals."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def downsample_area(x: torch.Tensor, res: int) -> torch.Tensor:
    """Area-average [..., H, W] down to [..., res, res]."""
    H, W = x.shape[-2:]
    if (H, W) == (res, res):
        return x
    lead = x.shape[:-2]
    y = x.reshape(-1, 1, H, W)
    if H % res == 0 and W % res == 0:
        y = F.avg_pool2d(y, kernel_size=(H // res, W // res))
    else:
        y = F.interpolate(y, size=(res, res), mode="area")
    return y.reshape(*lead, res, res)


def area_pyramid(x: torch.Tensor, levels) -> dict[int, torch.Tensor]:
    return {int(r): downsample_area(x, int(r)) for r in levels}


_BINOMIAL = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    H, W = x.shape[-2:]
    lead = x.shape[:-2]
    k = _BINOMIAL.to(x.device, x.dtype)
    y = x.reshape(-1, 1, H, W)
    y = F.conv2d(F.pad(y, (2, 2, 0, 0), mode="replicate"), k.view(1, 1, 1, 5))
    y = F.conv2d(F.pad(y, (0, 0, 2, 2), mode="replicate"), k.view(1, 1, 5, 1))
    return y.reshape(*lead, H, W)


def gaussian_pyramid(x: torch.Tensor, min_res: int = 16) -> list[torch.Tensor]:
    """Classic blur-and-decimate pyramid, finest first."""
    out = [x]
    while out[-1].shape[-1] // 2 >= min_res:
        b = gaussian_blur(out[-1])
        out.append(b[..., ::2, ::2])
    return out
