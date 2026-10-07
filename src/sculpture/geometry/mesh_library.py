"""Model pool: scans a folder, preprocesses every mesh (cached on disk)."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import trimesh

from .mesh_preprocess import canonicalize, load_mesh
from .sdf import MeshSDFLibrary, compute_mesh_sdf

log = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = (".obj", ".ply", ".stl", ".glb", ".gltf", ".off")
_PREPROCESS_VERSION = 4


def _file_hash(path: Path, extra: str) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    h.update(extra.encode())
    return h.hexdigest()[:16]


def _collision_points(data: dict, n: int) -> np.ndarray:
    """n canonical points: proxy vertices and edge midpoints first (corners/edges are where
    penetrations start), then surface samples. Deterministic shuffle so that any prefix
    [:k] is a uniform subset (used by the penalties)."""
    v = data["proxy_vertices"].astype(np.float32)
    f = data["proxy_faces"].astype(np.int64)
    edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    mids = v[edges].mean(1)
    rng = np.random.default_rng(0)
    feature = np.concatenate([v, mids])
    if len(feature) > n // 2:
        feature = feature[rng.choice(len(feature), n // 2, replace=False)]
    samples = data["samples"].astype(np.float32)
    rest = n - len(feature)
    samples = samples[rng.choice(len(samples), rest, replace=rest > len(samples))]
    pts = np.concatenate([feature, samples]).astype(np.float32)
    return pts[rng.permutation(len(pts))]


@dataclass
class MeshEntry:
    mesh_id: int
    name: str
    filename: str
    path: Path
    data: dict  # numpy arrays from canonicalize()
    torch_cache: dict = field(default_factory=dict, repr=False)

    @property
    def triangles(self) -> int:
        return int(len(self.data["faces"]))

    @property
    def proxy_triangles(self) -> int:
        return int(len(self.data["proxy_faces"]))

    @property
    def dims(self) -> list[float]:
        return (self.data["bbox_max"] - self.data["bbox_min"]).astype(float).tolist()

    def canonical_trimesh(self, lod: str = "original") -> trimesh.Trimesh:
        if lod == "original":
            return trimesh.Trimesh(self.data["vertices"], self.data["faces"], process=False)
        return trimesh.Trimesh(self.data[f"{lod}_vertices"], self.data[f"{lod}_faces"], process=False)

    def info(self) -> dict:
        return {
            "id": self.mesh_id,
            "name": self.name,
            "filename": self.filename,
            "triangles": self.triangles,
            "proxy_triangles": self.proxy_triangles,
            "dims": self.dims,
            "original_dims": self.data["original_extent"].astype(float).tolist(),
            "watertight": bool(self.data["watertight"]),
            "proxy_error": float(self.data.get("proxy_error", 0.0)),
            "coarse_triangles": int(len(self.data["coarse_faces"])),
        }


class MeshLibrary:
    """All models of the pool in canonical (centered, unit-radius) form.

    Everything a solver needs is exposed as torch tensors on ``device``:
    proxies per LOD, surface samples [M, S, 3], bounding boxes, radii.
    """

    def __init__(self, entries: list[MeshEntry], device, cfg, cache_dir: Path | None = None):
        if not entries:
            raise ValueError("model pool is empty")
        self.entries = entries
        self.device = torch.device(device)
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.samples = torch.stack([torch.from_numpy(e.data["samples"]) for e in entries]).to(self.device)
        self.sample_normals = torch.stack([torch.from_numpy(e.data["sample_normals"]) for e in entries]).to(self.device)
        self.radii = torch.ones(len(entries), device=self.device)
        self.bbox_min = torch.stack([torch.from_numpy(e.data["bbox_min"]) for e in entries]).to(self.device)
        self.bbox_max = torch.stack([torch.from_numpy(e.data["bbox_max"]) for e in entries]).to(self.device)
        # LOD approximation errors (canonical units): widen clearance margins so optimizing on
        # proxies cannot hide an intersection of the original meshes
        self.proxy_error = torch.tensor([float(e.data.get("proxy_error", 0.0)) for e in entries], device=self.device)
        n_coll = int(cfg.get("collision_points", 1024))
        self.collision_points = torch.stack([torch.from_numpy(_collision_points(e.data, n_coll)) for e in entries]).to(self.device)
        self._reference_points: dict[int, torch.Tensor] = {}
        self._sdf: MeshSDFLibrary | None = None

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_folder(cls, folder, cfg, device="cuda", cache_dir="cache") -> "MeshLibrary":
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"model folder {folder} does not exist")
        exts = tuple(e.lower() for e in cfg.get("extensions", DEFAULT_EXTENSIONS))
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)
        if not files:
            raise FileNotFoundError(f"no meshes with extensions {exts} found in {folder}")
        cache = Path(cache_dir) / "meshes" if cache_dir else None
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
        params = json.dumps(
            {k: cfg.get(k) for k in ("proxy_max_faces", "coarse_max_faces", "proxy_tolerance", "coarse_tolerance", "surface_samples")}
            | {"v": _PREPROCESS_VERSION},
            sort_keys=True,
        )
        entries = []
        names_seen: set[str] = set()
        for path in files:
            name = path.stem
            if name in names_seen:  # e.g. chair.obj and chair.ply
                name = f"{path.stem}_{path.suffix[1:]}"
            names_seen.add(name)
            data = None
            cache_file = None
            if cache:
                cache_file = cache / f"{path.stem}_{_file_hash(path, params)}.npz"
                if cache_file.exists():
                    with np.load(cache_file) as z:
                        data = {k: z[k] for k in z.files}
            if data is None:
                try:
                    mesh = load_mesh(path)
                    data = canonicalize(mesh, cfg)
                except Exception as exc:
                    log.warning("skipping %s: %s", path.name, exc)
                    continue
                if cache_file is not None:
                    np.savez_compressed(cache_file, **data)
            entries.append(MeshEntry(len(entries), name, path.name, path, data))
            log.info("loaded %s (%d tris, proxy %d)", path.name, entries[-1].triangles, entries[-1].proxy_triangles)
        lib = cls(entries, device, cfg, cache)
        lib.folder = folder
        return lib

    # ---------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    def by_name(self, name: str) -> MeshEntry:
        for e in self.entries:
            if e.name == name or e.filename == name:
                return e
        raise KeyError(name)

    def proxy(self, mesh_id: int, lod: str = "proxy") -> tuple[torch.Tensor, torch.Tensor]:
        """(vertices float32 [V,3], faces int32 [F,3]) on device for lod in {proxy, coarse, original}."""
        e = self.entries[mesh_id]
        if lod not in e.torch_cache:
            vk, fk = ("vertices", "faces") if lod == "original" else (f"{lod}_vertices", f"{lod}_faces")
            e.torch_cache[lod] = (
                torch.from_numpy(np.ascontiguousarray(e.data[vk], dtype=np.float32)).to(self.device),
                torch.from_numpy(np.ascontiguousarray(e.data[fk], dtype=np.int32)).to(self.device),
            )
        return e.torch_cache[lod]

    def content_hash(self) -> str:
        h = hashlib.sha1()
        for e in self.entries:
            h.update(e.name.encode())
            h.update(e.data["proxy_vertices"].tobytes()[:65536])
            h.update(str(e.data["proxy_vertices"].shape).encode())
        return h.hexdigest()[:16]

    def reference_points(self, mesh_id: int, n_samples: int = 12000, cap: int = 60000) -> torch.Tensor:
        """Dense canonical points of the ORIGINAL mesh (surface samples + vertices + edge
        midpoints) used by the reference intersection check."""
        if mesh_id not in self._reference_points:
            m = self.entries[mesh_id].canonical_trimesh("original")
            pts, _ = trimesh.sample.sample_surface(m, n_samples, seed=0)
            extra = np.concatenate([m.vertices, m.vertices[m.edges_unique].mean(1)])
            if len(extra) > cap:
                extra = extra[np.random.default_rng(0).choice(len(extra), cap, replace=False)]
            self._reference_points[mesh_id] = torch.tensor(np.concatenate([pts, extra]), dtype=torch.float32, device=self.device)
        return self._reference_points[mesh_id]

    # ---------------------------------------------------------------- SDFs
    def sdf(self, resolution: int | None = None) -> MeshSDFLibrary:
        """Canonical SDF grids for all models (computed lazily, cached on disk)."""
        res = int(resolution or self.cfg.get("sdf_resolution", 64))
        if self._sdf is not None and self._sdf.resolution == res:
            return self._sdf
        pad = float(self.cfg.get("sdf_padding", 0.1))
        grids = []
        lo = hi = None
        for e in self.entries:
            cache_file = None
            if self.cache_dir:
                key = hashlib.sha1(e.data["vertices"].tobytes() + e.data["faces"].tobytes()).hexdigest()[:16]
                cache_file = self.cache_dir / f"{e.name}_sdf{res}_p{pad}_{key}.npz"
            if cache_file is not None and cache_file.exists():
                with np.load(cache_file) as z:
                    d = {k: z[k] for k in z.files}
            else:
                d = compute_mesh_sdf(
                    e.data["vertices"], e.data["faces"], e.data["coarse_vertices"], e.data["coarse_faces"],
                    resolution=res, padding=pad, device=str(self.device),
                )
                log.info("SDF %s: %s (ambiguous %.3f)", e.name, d["method"], float(d["ambiguous_fraction"]))
                if cache_file is not None:
                    np.savez_compressed(cache_file, **d)
            grids.append(d["sdf"])
            lo, hi = float(d["lo"]), float(d["hi"])
        self._sdf = MeshSDFLibrary(grids, lo, hi, self.device)
        return self._sdf
