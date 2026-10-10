"""Method registry. The three solvers are independent implementations."""

from __future__ import annotations

METHOD_NAMES = ("beam", "column_generation", "sdf_ray", "chained")

DISPLAY_NAMES = {
    "beam": "Beam Constructive",
    "column_generation": "Column Generation",
    "sdf_ray": "SDF Ray Packing",
    "chained": "Chained (SDF -> CG -> Beam)",
}


def get_method(name: str):
    if name == "beam":
        from .beam.optimizer import BeamSearchOptimizer

        return BeamSearchOptimizer
    if name == "column_generation":
        from .column_generation.optimizer import ColumnGenerationOptimizer

        return ColumnGenerationOptimizer
    if name == "sdf_ray":
        from .sdf_ray.optimizer import SDFRayOptimizer

        return SDFRayOptimizer
    if name == "chained":
        from .chained.optimizer import ChainedOptimizer

        return ChainedOptimizer
    raise KeyError(f"unknown method {name!r}; choose from {METHOD_NAMES}")
