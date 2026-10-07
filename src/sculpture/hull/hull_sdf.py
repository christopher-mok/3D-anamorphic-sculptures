"""Signed distance field of a voxelized region (phi <= 0 inside)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def occupancy_sdf(occ: np.ndarray, voxel_size) -> np.ndarray:
    """Signed distance (world units) at voxel centers for occupancy [Z, Y, X].

    The exterior of the grid counts as outside (Omega is a subset of B).
    The surface is placed half a voxel from voxel centers.
    """
    occ = occ.astype(bool)
    vs = np.asarray(voxel_size, dtype=np.float64)[::-1]  # (z, y, x) sampling
    if not occ.any():
        # no inside: distance to "nothing" -> large positive constant
        return np.full(occ.shape, float(np.max(vs) * max(occ.shape)), np.float32)
    padded = np.pad(occ, 1, constant_values=False)
    d_out = ndimage.distance_transform_edt(~padded, sampling=vs)[1:-1, 1:-1, 1:-1]
    d_in = ndimage.distance_transform_edt(padded, sampling=vs)[1:-1, 1:-1, 1:-1]
    half = 0.5 * float(vs.min())
    sdf = np.where(occ, -(d_in - half), d_out - half)
    return sdf.astype(np.float32)
