"""ProblemContext: shared preprocessing consumed by all three methods."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .camera.perspective_camera import PerspectiveCamera, cameras_from_config
from .config import Config
from .constraints import ConstraintEvaluator
from .geometry.mesh_library import MeshLibrary
from .hull.compatibility import CompatibilityReport, RayIntervals, compatibility_report
from .hull.visual_hull import VisualHull
from .loss.multiscale import MultiScaleSilhouetteLoss
from .rendering.base import Renderer
from .targets.target import TargetSet

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _round8(x: int) -> int:
    return max(8, int(round(x / 8)) * 8)


def load_initial_assembly(ctx: "ProblemContext", assembly, locked=None, keep_unlocked: bool = True):
    """Lock-and-rerun warm start. ``assembly``: path to a result/assembly JSON or an assembly dict.
    ``locked``: indices to lock, "all", or None (keep the per-object "locked" flags of the file).
    ``keep_unlocked=False`` keeps only the locked pieces."""
    from .scene.serialization import assembly_from_dict

    d = json.loads(Path(assembly).read_text()) if isinstance(assembly, (str, Path)) else assembly
    if "assembly" in d:
        d = d["assembly"]
    a = assembly_from_dict(d, ctx.library, ctx.device)
    if locked == "all":
        a.locked[:] = True
    elif locked is not None:
        idx = torch.tensor([int(i) for i in locked if 0 <= int(i) < len(a)], dtype=torch.long, device=ctx.device)
        a.locked[:] = False
        if len(idx):
            a.locked[idx] = True
    if not keep_unlocked:
        a = a[a.locked]
    return a


def viewing_zone_cameras(cams, zone_cfg) -> list[tuple[PerspectiveCamera, int]]:
    """[(camera, primary view index)] on a circle of radius r around each camera position,
    perpendicular to its viewing direction, looking at the same point."""
    r = float(zone_cfg.get("radius", 0.0)) if zone_cfg else 0.0
    n = int(zone_cfg.get("samples", 4)) if zone_cfg else 0
    if r <= 0 or n <= 0:
        return []
    out = []
    for v, cam in enumerate(cams):
        Rwc = cam.rotation()
        right, up = Rwc[0], Rwc[1]
        for k in range(n):
            th = 2 * math.pi * k / n
            eye = cam.eye + r * (math.cos(th) * right + math.sin(th) * up)
            out.append((PerspectiveCamera(position=tuple(eye), look_at=tuple(cam.look_at), up=tuple(cam.up),
                                          fov_y_deg=cam.fov_y_deg, near=cam.near, far=cam.far), v))
    return out


def view_axis_volume(camera: PerspectiveCamera, bv) -> tuple[list[float], list[float]]:
    """Axis-aligned box of the camera frustum between distances view_axis_near and
    view_axis_far (the target cone is carved out of it later). Used when a single view
    does not bound how far / close objects may be along the viewing direction."""
    near = float(bv.get("view_axis_near", 1.5))
    far = float(bv.get("view_axis_far", 12.0))
    t = math.tan(math.radians(camera.fov_y_deg) / 2.0)
    Rwc = camera.rotation()  # rows: right, up, back
    pts = []
    for d in (near, far):
        for sx in (-1, 1):
            for sy in (-1, 1):
                pc = np.array([sx * t * d, sy * t * d, -d])
                pts.append(camera.eye + Rwc.T @ pc)
    pts = np.array(pts)
    return pts.min(0).round(4).tolist(), pts.max(0).round(4).tolist()


@dataclass
class ProblemContext:
    cfg: Config
    device: torch.device
    library: MeshLibrary
    renderer: Renderer
    targets: TargetSet
    cameras: list[PerspectiveCamera]
    hull: VisualHull
    compat: CompatibilityReport
    ray_intervals: list[RayIntervals]
    constraints: ConstraintEvaluator
    loss: MultiScaleSilhouetteLoss
    timings: dict = field(default_factory=dict)
    num_primary_views: int = 0  # user cameras; views beyond these are viewing-zone samples
    initial_assembly: object = None  # lock-and-rerun / chained warm start (Assembly with locked flags)
    _bank: object = None
    _generator: object = None
    _near_depth: float | None = None
    _hull_points: object = None
    _reveal: object = None
    _voxel: object = None
    _coarse_err: float = 0.0

    @property
    def num_views(self) -> int:
        return len(self.cameras)

    @property
    def working_resolution(self) -> int:
        return int(self.cfg.targets.working_resolution)

    @property
    def strict(self) -> bool:
        return self.hull.mode == "strict"

    @property
    def bank(self):
        """Silhouette bank (built lazily, cached on disk)."""
        if self._bank is None:
            from .proposals.silhouette_bank import SilhouetteBank

            t0 = time.time()
            self._bank = SilhouetteBank.build(self.library, self.renderer, self.cfg.silhouette_bank, cache_dir=self.library.cache_dir)
            self.timings["silhouette_bank"] = time.time() - t0
        return self._bank

    @property
    def generator(self):
        if self._generator is None:
            from .proposals.candidate_generator import CandidateGenerator

            self._generator = CandidateGenerator(self)
        return self._generator

    def pick_lod(self, assembly, resolution: int) -> str:
        """Resolution-aware level of detail: the coarsest LOD whose worst-case projected
        deviation (LOD error x largest instance scale x focal length / nearest depth) stays below
        cfg.meshes.lod_pixel_tolerance pixels. Low-resolution screening / lookahead renders then
        use the coarse mesh, high-resolution losses the precise proxy."""
        if len(assembly) == 0:
            return "coarse"
        if self._near_depth is None:
            bmin = torch.tensor(self.hull.bmin, dtype=torch.float32)
            bmax = torch.tensor(self.hull.bmax, dtype=torch.float32)
            near = []
            for cam in self.cameras:
                eye = torch.tensor(cam.eye, dtype=torch.float32)
                near.append(float((eye - eye.clamp(bmin, bmax)).norm()))
            self._near_depth = max(min(near), 0.25)
            self._coarse_err = max(float(e.data.get("coarse_error", 0.0)) for e in self.library.entries)
        tol = float(self.cfg.meshes.get("lod_pixel_tolerance", 0.5))
        with torch.no_grad():
            s_max = float(torch.exp(assembly.log_scale.max()))
        f = max(c.focal_px(resolution) for c in self.cameras)
        return "coarse" if self._coarse_err * s_max * f / self._near_depth <= tol else "proxy"

    @property
    def reveal(self):
        """Off-axis "reveal" cameras (RevealTerm), built lazily; None if there are none."""
        if self._reveal is None:
            from .loss.reveal import RevealTerm

            self._reveal = RevealTerm(self, self.cfg.get("reveal", {}))
        return self._reveal if self._reveal.cameras else None

    def hull_samples(self, n: int, gen: torch.Generator | None = None) -> torch.Tensor:
        """n random points inside the visual hull (voxel centers + sub-voxel jitter)."""
        if self._hull_points is None:
            self._hull_points = self.hull.inside_centers(strict=self.strict, max_points=400_000)
            self._voxel = torch.tensor(self.hull.voxel_size, dtype=torch.float32, device=self.device)
        P = self._hull_points
        idx = torch.randint(0, len(P), (n,), device=self.device, generator=gen)
        jitter = (torch.rand(n, 3, device=self.device, generator=gen) - 0.5) * self._voxel
        return P[idx] + jitter

    def render(self, assembly, resolution: int | None = None, lod: str = "auto") -> torch.Tensor:
        r = resolution or self.working_resolution
        if lod == "auto":
            lod = self.pick_lod(assembly, r)
        return self.renderer.render_silhouettes(assembly, self.cameras, (r, r), lod=lod)

    def render_instances(self, assembly, resolution: int, lod: str = "auto") -> torch.Tensor:
        if lod == "auto":
            lod = self.pick_lod(assembly, resolution)
        return self.renderer.render_instance_silhouettes(assembly, self.cameras, (resolution, resolution), lod=lod)

    def generator_torch(self, seed: int) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(int(seed))

    # ------------------------------------------------------------ outputs
    def save_preprocessing(self, folder) -> dict:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.targets.save_pngs(folder, self.working_resolution, n=self.num_primary_views or None)
        self.hull.save(folder / "hull.npz")
        faces = self.hull.export_preview_obj(folder / "hull_preview.obj")
        V0 = self.num_primary_views or len(self.cameras)
        diag = {
            "cameras": [c.to_dict() for c in self.cameras[:V0]],
            "viewing_zone_cameras": [c.to_dict() for c in self.cameras[V0:]],
            "bounding_volume": {"min": list(map(float, self.hull.bmin)), "max": list(map(float, self.hull.bmax))},
            "hull_mode": self.hull.mode,
            "hull_resolution": list(map(int, self.hull.resolution)),
            "hull_voxels": self.hull.num_voxels,
            "hull_preview_faces": faces,
            "compatibility": self.compat.to_dict(),
            "models": [e.info() for e in self.library.entries],
            "targets": self.targets.paths,
            "timings": self.timings,
        }
        (folder / "diagnostics.json").write_text(json.dumps(diag, indent=2))
        return diag


def build_context(
    cfg: Config,
    models_dir,
    target_paths: Sequence,
    cameras: Sequence | None = None,
    library: MeshLibrary | None = None,
    renderer: Renderer | None = None,
) -> ProblemContext:
    """Run all shared preprocessing: models, targets, hull, SDF, diagnostics."""
    timings = {}
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required (nvdiffrast backend).")
    set_seed(int(cfg.seed))

    t0 = time.time()
    if library is None:
        library = MeshLibrary.from_folder(models_dir, cfg.meshes, device=device, cache_dir=cfg.cache_dir)
    timings["meshes"] = time.time() - t0

    if renderer is None:
        from .rendering.nvdiffrast_renderer import NvdiffrastRenderer

        renderer = NvdiffrastRenderer(library, device)

    n_views = len(target_paths)
    if cameras is None:
        cams = cameras_from_config(cfg.cameras, n_views)
    else:
        cams = [c if isinstance(c, PerspectiveCamera) else PerspectiveCamera.from_dict(c) for c in cameras]
    if len(cams) != n_views:
        raise ValueError(f"{n_views} targets but {len(cams)} cameras")

    # resolutions must be multiples of 8 for the CUDA rasterizer
    cfg.targets.working_resolution = _round8(cfg.targets.working_resolution)
    cfg.evaluation.resolution = _round8(cfg.evaluation.resolution)
    base_res = max(cfg.evaluation.resolution, cfg.targets.working_resolution, cfg.hull.compatibility_resolution, max(cfg.loss.levels))

    t0 = time.time()
    targets = TargetSet.from_files(target_paths, base_res, cfg.targets, device=device)
    timings["targets"] = time.time() - t0
    n_primary = len(cams)
    zone = viewing_zone_cameras(cams, cfg.get("viewing_zone", {}))
    if zone:
        # VIEWING ZONE: extra cameras on a circle around each user camera (perpendicular to its view
        # direction, same look-at), all sharing that camera's target: the illusion must hold for
        # every eye position in the zone. They are ordinary views for every downstream component.
        cams = cams + [c for c, _ in zone]
        base = targets.base.cpu().numpy()
        colors = targets.color_base.cpu().numpy() if targets.color_base is not None else None
        expanded_colors = None if colors is None else [colors[v] for v in range(n_primary)] + [colors[v] for _, v in zone]
        targets = TargetSet([base[v] for v in range(n_primary)] + [base[v] for _, v in zone], device,
                            list(targets.paths) + [targets.paths[v] for _, v in zone], expanded_colors,
                            cfg.targets.get("color", {}), targets.mask_background)

    isotropic = True
    if cfg.bounding_volume.get("unbounded_view_axis", False):
        if n_views == 1:
            bmin, bmax = view_axis_volume(cams[0], cfg.bounding_volume)
            cfg.bounding_volume.original = {"min": list(cfg.bounding_volume.min), "max": list(cfg.bounding_volume.max)}
            cfg.bounding_volume.min, cfg.bounding_volume.max = bmin, bmax
            far_needed = float(cfg.bounding_volume.view_axis_far) * 1.1 + 2.0
            if cams[0].far < far_needed:
                cams[0].far = far_needed
            isotropic = False
            log.info("unbounded view axis: bounding volume extended to %s .. %s", bmin, bmax)
        else:
            log.warning("bounding_volume.unbounded_view_axis is only supported for a single view; ignored")

    t0 = time.time()
    hull = VisualHull.carve(
        targets, cams, cfg.bounding_volume.min, cfg.bounding_volume.max,
        resolution=int(cfg.hull.resolution), mode=cfg.hull.mode,
        dilation_voxels=int(cfg.hull.relaxed_dilation_voxels), device=device, isotropic=isotropic,
    )
    timings["hull"] = time.time() - t0

    t0 = time.time()
    compat, intervals = compatibility_report(hull, targets, cams, int(cfg.hull.compatibility_resolution), float(cfg.hull.warn_coverage_bound))
    timings["compatibility"] = time.time() - t0
    for w in compat.warnings:
        log.warning(w)

    constraints = ConstraintEvaluator(library, hull, cfg)
    constraints.camera_eye = cams[0].torch_eye(device)
    t0 = time.time()
    constraints.sdf  # precompute model SDFs (shared by collisions and the SDF-ray method)
    timings["model_sdf"] = time.time() - t0

    loss = MultiScaleSilhouetteLoss.from_config(targets, cfg.loss, max_resolution=cfg.targets.working_resolution)
    return ProblemContext(cfg, device, library, renderer, targets, cams, hull, compat, intervals, constraints, loss, timings, n_primary)
