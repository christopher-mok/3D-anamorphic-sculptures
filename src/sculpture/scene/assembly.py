"""Assembly X = {O_1..O_N} stored as batched tensors (differentiable)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch

from ..geometry.rotation import matrix_to_rot6d, rot6d_to_matrix
from .object_instance import ObjectInstance


class Assembly:
    """Structure-of-arrays representation.

    mesh_ids  LongTensor [N]       (discrete)
    translation   [N, 3]          (continuous)
    rot6d         [N, 6]          (continuous)
    log_scale     [N]             (continuous)

    Tensors may require grad; ``params()`` turns an assembly into leaf
    tensors for joint optimization.
    """

    def __init__(self, mesh_ids, translation, rot6d, log_scale):
        self.mesh_ids = torch.as_tensor(mesh_ids, dtype=torch.long)
        self.translation = translation
        self.rot6d = rot6d
        self.log_scale = log_scale
        n = self.mesh_ids.shape[0]
        assert translation.shape == (n, 3) and rot6d.shape == (n, 6) and log_scale.shape == (n,), (
            translation.shape, rot6d.shape, log_scale.shape)
        self.mesh_ids = self.mesh_ids.to(translation.device)

    # ---------------------------------------------------------------- creation
    @classmethod
    def empty(cls, device="cuda") -> "Assembly":
        return cls(
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 6, device=device),
            torch.zeros(0, device=device),
        )

    @classmethod
    def from_matrices(cls, mesh_ids, translation, R, scale) -> "Assembly":
        translation = torch.as_tensor(translation, dtype=torch.float32)
        R = torch.as_tensor(R, dtype=torch.float32, device=translation.device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=translation.device)
        return cls(torch.as_tensor(mesh_ids, device=translation.device), translation, matrix_to_rot6d(R), torch.log(scale))

    @classmethod
    def from_instances(cls, instances: Sequence[ObjectInstance], device="cuda") -> "Assembly":
        if len(instances) == 0:
            return cls.empty(device)
        return cls(
            torch.tensor([o.mesh_id for o in instances], dtype=torch.long, device=device),
            torch.tensor(np.stack([o.translation for o in instances]), dtype=torch.float32, device=device),
            torch.tensor(np.stack([o.rotation6d for o in instances]), dtype=torch.float32, device=device),
            torch.tensor([o.log_scale for o in instances], dtype=torch.float32, device=device),
        )

    @staticmethod
    def concat(assemblies: Iterable["Assembly"]) -> "Assembly":
        a = list(assemblies)
        return Assembly(
            torch.cat([x.mesh_ids for x in a]),
            torch.cat([x.translation for x in a]),
            torch.cat([x.rot6d for x in a]),
            torch.cat([x.log_scale for x in a]),
        )

    # ---------------------------------------------------------------- views
    def __len__(self) -> int:
        return int(self.mesh_ids.shape[0])

    @property
    def device(self):
        return self.translation.device

    def rotation_matrices(self) -> torch.Tensor:
        return rot6d_to_matrix(self.rot6d)

    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scale)

    def to_instances(self) -> list[ObjectInstance]:
        ids = self.mesh_ids.cpu().numpy()
        t = self.translation.detach().cpu().numpy()
        r = self.rot6d.detach().cpu().numpy()
        s = self.log_scale.detach().cpu().numpy()
        return [ObjectInstance(int(ids[i]), t[i].copy(), r[i].copy(), float(s[i])) for i in range(len(ids))]

    def __getitem__(self, idx) -> "Assembly":
        if isinstance(idx, int):
            idx = [idx]
        if isinstance(idx, slice):
            idx = torch.arange(len(self), device=self.device)[idx]
        idx = torch.as_tensor(idx, device=self.device)
        if idx.dtype == torch.bool:
            idx = idx.nonzero(as_tuple=True)[0]
        return Assembly(self.mesh_ids[idx], self.translation[idx], self.rot6d[idx], self.log_scale[idx])

    def without(self, i: int) -> "Assembly":
        keep = torch.ones(len(self), dtype=torch.bool, device=self.device)
        keep[i] = False
        return self[keep]

    def detach(self) -> "Assembly":
        return Assembly(self.mesh_ids.clone(), self.translation.detach().clone(), self.rot6d.detach().clone(), self.log_scale.detach().clone())

    clone = detach

    def to(self, device) -> "Assembly":
        return Assembly(self.mesh_ids.to(device), self.translation.to(device), self.rot6d.to(device), self.log_scale.to(device))

    def params(self) -> "Assembly":
        """Detached leaf copy with requires_grad=True on the continuous parameters."""
        a = self.detach()
        a.translation.requires_grad_(True)
        a.rot6d.requires_grad_(True)
        a.log_scale.requires_grad_(True)
        return a

    def with_params(self, translation=None, rot6d=None, log_scale=None) -> "Assembly":
        return Assembly(
            self.mesh_ids,
            self.translation if translation is None else translation,
            self.rot6d if rot6d is None else rot6d,
            self.log_scale if log_scale is None else log_scale,
        )

    def normalized_rotations(self) -> "Assembly":
        """Re-orthonormalize the 6D parameters (keeps values well conditioned)."""
        return self.with_params(rot6d=matrix_to_rot6d(self.rotation_matrices().detach()))

    def __repr__(self) -> str:
        return f"Assembly(n={len(self)})"
