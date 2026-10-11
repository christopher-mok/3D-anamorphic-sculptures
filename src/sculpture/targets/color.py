"""Perceptual target-color quantization and conversion helpers."""

from __future__ import annotations

import numpy as np


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, np.float32).clip(0, 1)
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]], np.float32).T
    xyz /= np.array([0.95047, 1.0, 1.08883], np.float32)
    d = 6 / 29
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3 * d**2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    x = np.asarray(lab, np.float32)
    fy = (x[..., 0] + 16) / 116
    fx, fz = fy + x[..., 1] / 500, fy - x[..., 2] / 200
    d = 6 / 29
    inv = lambda q: np.where(q > d, q**3, 3 * d**2 * (q - 4 / 29))
    xyz = np.stack([inv(fx), inv(fy), inv(fz)], -1) * np.array([0.95047, 1.0, 1.08883], np.float32)
    lin = xyz @ np.array([[3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660, 1.8760108, 0.0415560],
                          [0.0556434, -0.2040259, 1.0572252]], np.float32).T
    rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.maximum(lin, 0) ** (1 / 2.4) - 0.055)
    return rgb.clip(0, 1)


def quantize_lab(rgb: np.ndarray, foreground: np.ndarray, k: int = 8, min_fraction: float = 0.015,
                 min_segment_fraction: float = 0.003):
    """Return SLIC spatial segments merged by adjacent Lab similarity, plus a palette.

    Palette reduction never deletes a spatial segment: a small contrasting feature is
    retained even if it is too rare to be a major palette color.
    """
    lab = srgb_to_lab(rgb)
    pixels = lab[foreground]
    if len(pixels) == 0:
        return np.zeros(foreground.shape, np.int64), np.array([[50, 0, 0]], np.float32)
    if len(pixels) > 50000:
        pixels = pixels[np.linspace(0, len(pixels) - 1, 50000).astype(int)]
    centers = [pixels.mean(0)]
    for _ in range(1, min(k, len(pixels))):
        d2 = ((pixels[:, None] - np.asarray(centers)[None]) ** 2).sum(-1).min(1)
        centers.append(pixels[int(d2.argmax())])
    centers = np.asarray(centers, np.float32)
    for _ in range(20):
        labels = ((pixels[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
        new = np.stack([pixels[labels == i].mean(0) if np.any(labels == i) else centers[i] for i in range(len(centers))])
        if np.max(np.abs(new - centers)) < 0.05:
            centers = new
            break
        centers = new
    counts = np.bincount(labels, minlength=len(centers))
    major = np.flatnonzero(counts >= max(1, int(min_fraction * counts.sum())))
    if len(major) == 0:
        major = np.array([int(counts.argmax())])
    centers = centers[major]
    full = ((lab[..., None, :] - centers[None, None]) ** 2).sum(-1).argmin(-1).astype(np.int64)
    full[~foreground] = -1
    from skimage.segmentation import slic

    # Spatial superpixels preserve boundaries that global palette labels cannot express.
    # Merge adjacent, perceptually similar superpixels; never merge distant islands or
    # discard a contrasting feature just because it occupies a small image fraction.
    segments = slic(np.asarray(rgb, np.float32), n_segments=max(16, min(400, int(foreground.sum()) // 256)),
                    compactness=5, sigma=0.5, mask=foreground, start_label=1, channel_axis=-1) - 1
    ids = np.unique(segments[foreground])
    parent = {int(i): int(i) for i in ids}
    means = {int(i): lab[segments == i].mean(0) for i in ids}
    sizes = {int(i): int((segments == i).sum()) for i in ids}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = set()
    for a, b in ((segments[:-1], segments[1:]), (segments[:, :-1], segments[:, 1:])):
        valid = (a >= 0) & (b >= 0) & (a != b)
        pairs.update(tuple(sorted(p)) for p in np.stack([a[valid], b[valid]], -1).tolist())
    for a, b in sorted(pairs, key=lambda p: float(np.linalg.norm(means[p[0]] - means[p[1]]))):
        a, b = root(a), root(b)
        if a != b and np.linalg.norm(means[a] - means[b]) < 12:
            total = sizes[a] + sizes[b]
            means[a] = (means[a] * sizes[a] + means[b] * sizes[b]) / total
            sizes[a] = total
            parent[b] = a
    lut = np.arange(int(segments.max()) + 1)
    for i in ids:
        lut[i] = root(int(i))
    segments[foreground] = lut[segments[foreground]]
    segments[~foreground] = -1
    return segments, centers.astype(np.float32)
