"""Request / response models (see docs/api.md)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

MethodName = Literal["beam", "column_generation", "sdf_ray", "chained"]


class CameraModel(BaseModel):
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 5.0])
    look_at: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    up: list[float] = Field(default_factory=lambda: [0.0, 1.0, 0.0])
    fov_y_deg: float = 25.0
    near: float = 0.1
    far: float = 20.0


class InitialAssembly(BaseModel):
    """Lock-and-rerun: an edited assembly, the indices that must stay fixed, and whether the
    unlocked pieces are kept as a warm start (True) or discarded (False)."""

    assembly: dict[str, Any]
    locked: list[int] = Field(default_factory=list)
    keep_unlocked: bool = True


class JobRequest(BaseModel):
    models_dir: str = "assets/models"
    targets: list[str]
    cameras: list[CameraModel] | None = None
    methods: list[MethodName] = Field(default_factory=lambda: ["beam", "column_generation", "sdf_ray"])
    preset: Literal["fast", "default", "high_quality"] = "default"
    overrides: dict[str, Any] | None = None
    initial: InitialAssembly | None = None
