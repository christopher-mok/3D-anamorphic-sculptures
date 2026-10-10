"""Target silhouette loading and the per-resolution target cache."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from .distance_transform import background_distance, foreground_distance
from .pyramids import downsample_area, gaussian_pyramid


def load_target_image(path, resolution: int, threshold: float = 0.5, invert="auto", pad_fraction: float = 0.0) -> np.ndarray:
    """Load an image as a square binary foreground mask of size resolution^2.

    * EXIF orientation is normalized.
    * An alpha channel with real transparency is used as the mask directly.
    * Otherwise grayscale is thresholded; with invert="auto" the polarity is
      chosen so that the image border is background.
    * Aspect ratio is preserved by letterboxing (padding with background).
    """
    img = ImageOps.exif_transpose(Image.open(path))
    alpha = None
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
        if rgba[..., 3].min() < 0.99:
            alpha = rgba[..., 3]
    if alpha is not None:
        fg = alpha
    else:
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        if invert == "auto":
            border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
            dark_fg = border.mean() > 0.5  # bright border -> dark silhouette on light background
        else:
            dark_fg = bool(invert)
        fg = 1.0 - gray if dark_fg else gray

    h, w = fg.shape
    side = int(round(max(h, w) * (1.0 + 2.0 * pad_fraction)))
    canvas = np.zeros((side, side), np.float32)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = fg
    resized = Image.fromarray(canvas, mode="F").resize((resolution, resolution), Image.BOX if side >= resolution else Image.BILINEAR)
    return np.asarray(resized) >= threshold


class TargetSet:
    """Binary target masks for V views plus cached derived quantities.

    All masks are square; ``base_resolution`` is the finest resolution used
    anywhere (evaluation / hull carving). Coarser binary masks are derived by
    area downsampling and thresholding at 0.5; soft (area) pyramids are used by
    the multiscale loss.
    """

    def __init__(self, masks: Sequence[np.ndarray], device="cuda", paths: Sequence[str] | None = None):
        if len(masks) < 1:
            raise ValueError("at least one target image is required")
        self.device = torch.device(device)
        self.paths = list(paths) if paths else [None] * len(masks)
        self.base = torch.tensor(np.stack(masks), dtype=torch.float32, device=self.device)  # [V,S,S]
        self.base_resolution = int(self.base.shape[-1])
        for i, m in enumerate(masks):
            if m.sum() == 0:
                raise ValueError(f"target {i} has an empty foreground; check threshold / invert settings")
        self._cache: dict = {}

    @classmethod
    def from_files(cls, paths, base_resolution: int, cfg, device="cuda") -> "TargetSet":
        masks = [
            load_target_image(p, base_resolution, cfg.get("threshold", 0.5), cfg.get("invert", "auto"), cfg.get("pad_fraction", 0.0))
            for p in paths
        ]
        return cls(masks, device, [str(p) for p in paths])

    @property
    def num_views(self) -> int:
        return int(self.base.shape[0])

    def soft(self, res: int) -> torch.Tensor:
        key = ("soft", res)
        if key not in self._cache:
            self._cache[key] = downsample_area(self.base, res)
        return self._cache[key]

    def mask(self, res: int) -> torch.Tensor:
        """Binary {0,1} float masks [V, res, res]."""
        key = ("mask", res)
        if key not in self._cache:
            self._cache[key] = (self.soft(res) >= 0.5).float()
        return self._cache[key]

    def fg_distance(self, res: int) -> torch.Tensor:
        key = ("dtfg", res)
        if key not in self._cache:
            m = self.mask(res).cpu().numpy()
            self._cache[key] = torch.tensor(np.stack([foreground_distance(x) for x in m]), device=self.device)
        return self._cache[key]

    def bg_distance(self, res: int) -> torch.Tensor:
        key = ("dtbg", res)
        if key not in self._cache:
            m = self.mask(res).cpu().numpy()
            self._cache[key] = torch.tensor(np.stack([background_distance(x) for x in m]), device=self.device)
        return self._cache[key]

    def gaussian(self, res: int, min_res: int = 16) -> list[torch.Tensor]:
        key = ("gauss", res, min_res)
        if key not in self._cache:
            self._cache[key] = gaussian_pyramid(self.mask(res), min_res)
        return self._cache[key]

    def save_pngs(self, folder, res: int | None = None, n: int | None = None) -> list[Path]:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        m = self.mask(res or self.base_resolution).cpu().numpy()[:n]
        for i, x in enumerate(m):
            p = folder / f"target_{i}.png"
            Image.fromarray(((1.0 - x) * 255).astype(np.uint8)).save(p)  # black silhouette on white
            out.append(p)
        return out
