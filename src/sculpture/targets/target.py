"""Target silhouette loading and the per-resolution target cache."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from .distance_transform import background_distance, foreground_distance
from .pyramids import downsample_area, gaussian_pyramid
from .color import quantize_lab


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


def load_target_rgb(path, resolution: int, pad_fraction: float = 0.0) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    h, w = rgb.shape[:2]
    side = int(round(max(h, w) * (1.0 + 2.0 * pad_fraction)))
    canvas = np.zeros((side, side, 3), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgb
    return np.asarray(Image.fromarray(canvas).resize((resolution, resolution), Image.Resampling.BOX if side >= resolution else Image.Resampling.BILINEAR), np.float32) / 255.0


def load_target_extent(path, resolution: int, pad_fraction: float = 0.0) -> np.ndarray:
    """Mask of actual image pixels after aspect-preserving square letterboxing."""
    img = ImageOps.exif_transpose(Image.open(path))
    w, h = img.size
    side = int(round(max(h, w) * (1.0 + 2.0 * pad_fraction)))
    canvas = np.zeros((side, side), np.float32)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = 1.0
    resized = Image.fromarray(canvas, mode="F").resize(
        (resolution, resolution), Image.Resampling.BOX if side >= resolution else Image.Resampling.BILINEAR,
    )
    return np.asarray(resized) >= 0.5


class TargetSet:
    """Binary target masks for V views plus cached derived quantities.

    All masks are square; ``base_resolution`` is the finest resolution used
    anywhere (evaluation / hull carving). Coarser binary masks are derived by
    area downsampling and thresholding at 0.5; soft (area) pyramids are used by
    the multiscale loss.
    """

    def __init__(self, masks: Sequence[np.ndarray], device="cuda", paths: Sequence[str] | None = None,
                 colors: Sequence[np.ndarray] | None = None, color_cfg=None, mask_background: bool = True):
        if len(masks) < 1:
            raise ValueError("at least one target image is required")
        self.device = torch.device(device)
        self.paths = list(paths) if paths else [None] * len(masks)
        self.mask_background = bool(mask_background)
        self.base = torch.tensor(np.stack(masks), dtype=torch.float32, device=self.device)  # [V,S,S]
        self.base_resolution = int(self.base.shape[-1])
        for i, m in enumerate(masks):
            if m.sum() == 0:
                raise ValueError(f"target {i} has an empty foreground; check threshold / invert settings")
        self._cache: dict = {}
        self.color_base = None if colors is None else torch.tensor(np.stack(colors), dtype=torch.float32, device=self.device)
        self.color_labels = None
        self.color_palettes = []
        if self.color_base is not None and (color_cfg is None or color_cfg.get("enabled", True)):
            labels = []
            for rgb, mask in zip(colors, masks):
                lab, palette = quantize_lab(rgb, np.asarray(mask, bool), int((color_cfg or {}).get("palette_size", 8)),
                                            float((color_cfg or {}).get("min_region_fraction", 0.015)),
                                            float((color_cfg or {}).get("min_segment_fraction", 0.003)))
                labels.append(lab)
                self.color_palettes.append(palette)
            self.color_labels = torch.tensor(np.stack(labels), dtype=torch.long, device=self.device)

    @classmethod
    def from_files(cls, paths, base_resolution: int, cfg, device="cuda") -> "TargetSet":
        masks = [
            load_target_image(p, base_resolution, cfg.get("threshold", 0.5), cfg.get("invert", "auto"), cfg.get("pad_fraction", 0.0))
            for p in paths
        ]
        mask_background = bool(cfg.get("mask_background", True))
        if not mask_background:
            masks = [load_target_extent(p, base_resolution, cfg.get("pad_fraction", 0.0)) for p in paths]
        colors = [load_target_rgb(p, base_resolution, cfg.get("pad_fraction", 0.0)) for p in paths]
        return cls(masks, device, [str(p) for p in paths], colors, cfg.get("color", {}), mask_background)

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

    def rgb(self, res: int) -> torch.Tensor | None:
        if self.color_base is None:
            return None
        key = ("rgb", res)
        if key not in self._cache:
            x = self.color_base.permute(0, 3, 1, 2)
            self._cache[key] = torch.nn.functional.interpolate(x, (res, res), mode="area").permute(0, 2, 3, 1)
        return self._cache[key]

    def regions(self, res: int) -> torch.Tensor | None:
        if self.color_labels is None:
            return None
        key = ("regions", res)
        if key not in self._cache:
            x = self.color_labels.float()[:, None]
            self._cache[key] = torch.nn.functional.interpolate(x, (res, res), mode="nearest")[:, 0].long()
        return self._cache[key]

    def color_region_distance(self, res: int) -> torch.Tensor | None:
        """Distance to the nearest palette-region boundary, in pixels."""
        regions = self.regions(res)
        if regions is None:
            return None
        key = ("color_region_distance", res)
        if key not in self._cache:
            from scipy.ndimage import distance_transform_edt
            out = []
            for v in range(len(regions)):
                r = regions[v].cpu().numpy()
                d = np.zeros_like(r, np.float32)
                for label in np.unique(r[r >= 0]):
                    d = np.maximum(d, distance_transform_edt(r == label).astype(np.float32))
                out.append(d)
            self._cache[key] = torch.tensor(np.stack(out), device=self.device)
        return self._cache[key]

    def save_pngs(self, folder, res: int | None = None, n: int | None = None) -> list[Path]:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        count = n or self.num_views
        m = self.mask(res or self.base_resolution).cpu().numpy()[:count]
        rgb = self.rgb(res or self.base_resolution)
        for i, x in enumerate(m):
            p = folder / f"target_{i}.png"
            if self.color_labels is not None and rgb is not None:
                Image.fromarray((rgb[i].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)).save(p)
            else:
                Image.fromarray(((1.0 - x) * 255).astype(np.uint8)).save(p)  # black silhouette on white
            out.append(p)
        return out
