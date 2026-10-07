"""A single placed instance O_i = (mesh_id, translation, rotation, log_scale)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..geometry.rotation import matrix_to_rot6d, rot6d_to_matrix


@dataclass
class ObjectInstance:
    mesh_id: int                      # DISCRETE identity, never relaxed
    translation: np.ndarray           # [3]
    rotation6d: np.ndarray            # [6] continuous 6D rotation
    log_scale: float                  # isotropic scale = exp(log_scale)

    @property
    def scale(self) -> float:
        return float(np.exp(self.log_scale))

    @property
    def rotation_matrix(self) -> np.ndarray:
        return rot6d_to_matrix(torch.as_tensor(self.rotation6d, dtype=torch.float32)).numpy()

    @classmethod
    def from_matrix(cls, mesh_id: int, translation, R, scale: float) -> "ObjectInstance":
        r6 = matrix_to_rot6d(torch.as_tensor(np.asarray(R), dtype=torch.float32)).numpy()
        return cls(int(mesh_id), np.asarray(translation, dtype=np.float32), r6, float(np.log(scale)))

    def world_matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.rotation_matrix * self.scale
        M[:3, 3] = self.translation
        return M
