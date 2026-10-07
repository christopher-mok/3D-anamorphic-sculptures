"""Foreground / background Euclidean distance transforms (in pixels)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def foreground_distance(mask: np.ndarray) -> np.ndarray:
    """For foreground pixels: distance to the nearest background pixel (0 on background)."""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros(m.shape, np.float32)
    padded = np.pad(m, 1, constant_values=False)  # image border counts as background
    return ndimage.distance_transform_edt(padded)[1:-1, 1:-1].astype(np.float32)


def background_distance(mask: np.ndarray) -> np.ndarray:
    """For background pixels: distance to the nearest foreground pixel (0 on foreground)."""
    m = mask.astype(bool)
    if not m.any():
        return np.full(m.shape, float(max(m.shape)), np.float32)
    return ndimage.distance_transform_edt(~m).astype(np.float32)
