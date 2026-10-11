"""Renderer interface. Everything outside ``rendering/`` depends only on this."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, Union, runtime_checkable

import torch

from ..camera.perspective_camera import PerspectiveCamera
from ..scene.assembly import Assembly
from ..scene.object_instance import ObjectInstance

InstancesLike = Union[Assembly, Sequence[ObjectInstance]]


@dataclass
class RenderStats:
    calls: int = 0          # number of render_* invocations
    images: int = 0         # number of (instance or assembly) x view images produced
    pixels: int = 0
    by_kind: dict = field(default_factory=dict)

    def record(self, kind: str, images: int, pixels: int) -> None:
        self.calls += 1
        self.images += images
        self.pixels += pixels
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1

    def snapshot(self) -> dict:
        return {"calls": self.calls, "images": self.images, "pixels": self.pixels, "by_kind": dict(self.by_kind)}


@runtime_checkable
class Renderer(Protocol):
    """Differentiable silhouette renderer.

    Output masks are float tensors in [0, 1] with image convention row 0 = top.
    Gradients flow to the assembly's translation / rot6d / log_scale tensors.
    ``lod`` selects the geometry level of detail: "proxy" (default),
    "coarse" (cheap screening) or "original".
    """

    stats: RenderStats

    def render_silhouettes(
        self,
        assembly: Assembly,
        cameras: Sequence[PerspectiveCamera],
        resolution: tuple[int, int],
        lod: str = "proxy",
    ) -> torch.Tensor:
        """Union silhouette of the whole assembly: [V, H, W]."""
        ...

    def render_instance_silhouettes(
        self,
        instances: InstancesLike,
        cameras: Sequence[PerspectiveCamera],
        resolution: tuple[int, int],
        lod: str = "proxy",
    ) -> torch.Tensor:
        """Independent silhouette per instance: [N, V, H, W]."""
        ...

    def render_instance_ids(self, assembly: Assembly, cameras: Sequence[PerspectiveCamera], resolution: tuple[int, int],
                            lod: str = "proxy") -> torch.Tensor:
        """Z-buffer-visible instance index per pixel [V,H,W], background = -1."""
        ...
