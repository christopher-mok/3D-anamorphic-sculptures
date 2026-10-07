"""Low-resolution orientation silhouette bank + silhouette-similarity index.

For every model m and every orientation R_k of an (approximately) uniform
SO(3) sample, the bank stores the silhouette seen by a canonical, nearly
orthographic camera looking down -Z with +Y up. Orientations are therefore
*camera-frame* orientations: to make an object look like entry k in a target
camera with world->camera rotation R_view, use R_world = R_view^T R_k.

The unit bounding sphere spans `FRAME` radii from the image center to the
border, so a bank image covers [-FRAME, FRAME]^2 in units of object radius.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..camera.perspective_camera import PerspectiveCamera
from ..geometry.rotation import matrix_to_rot6d, super_fibonacci_rotations
from ..scene.assembly import Assembly
from ..targets.pyramids import downsample_area

log = logging.getLogger(__name__)

FRAME = 1.05
_CANON_DISTANCE = 40.0


def canonical_camera() -> PerspectiveCamera:
    fov = 2.0 * math.degrees(math.atan(FRAME / _CANON_DISTANCE))
    return PerspectiveCamera(position=(0, 0, _CANON_DISTANCE), look_at=(0, 0, 0), up=(0, 1, 0), fov_y_deg=fov, near=1.0, far=100.0)


class SilhouetteBank:
    def __init__(self, rotations: torch.Tensor, masks: torch.Tensor, descriptor_resolution: int, n_neighbors: int = 8):
        self.device = masks.device
        self.rotations = rotations.to(self.device)          # [K,3,3] camera-frame orientations
        self.masks = masks                                  # [M,K,r,r] float
        self.M, self.K = masks.shape[:2]
        self.resolution = masks.shape[-1]
        self.d = descriptor_resolution
        flat = masks.reshape(self.M * self.K, self.resolution, self.resolution)
        self.desc = downsample_area(flat, self.d).reshape(self.M * self.K, -1)  # coverage fractions
        self.desc_unit = F.normalize(self.desc, dim=-1)
        self.area = self.desc.sum(-1)                        # in descriptor pixels
        self._build_index(n_neighbors)

    # ------------------------------------------------------------ construction
    @classmethod
    def build(cls, library, renderer, cfg, cache_dir: Path | None = None) -> "SilhouetteBank":
        K = int(cfg.num_orientations)
        res = int(cfg.resolution)
        d = int(cfg.descriptor_resolution)
        cache_file = None
        if cache_dir is not None:
            key = hashlib.sha1(f"{library.content_hash()}_{K}_{res}".encode()).hexdigest()[:16]
            cache_file = Path(cache_dir) / f"bank_{key}.npz"
            if cache_file.exists():
                with np.load(cache_file) as z:
                    return cls(torch.tensor(z["rotations"]), torch.tensor(z["masks"], device=library.device).float(), d, int(cfg.get("neighbors", 8)))
        rots = super_fibonacci_rotations(K).to(library.device)
        cam = canonical_camera()
        masks = []
        with torch.no_grad():
            for m in range(len(library)):
                a = Assembly(
                    torch.full((K,), m, dtype=torch.long, device=library.device),
                    torch.zeros(K, 3, device=library.device),
                    matrix_to_rot6d(rots),
                    torch.zeros(K, device=library.device),
                )
                masks.append(renderer.render_instance_silhouettes(a, [cam], (res, res), lod="proxy")[:, 0])
        masks = torch.stack(masks)  # [M,K,r,r]
        if cache_file is not None:
            np.savez_compressed(cache_file, rotations=rots.cpu().numpy(), masks=(masks > 0.5).cpu().numpy())
        return cls(rots, (masks > 0.5).float(), d, int(cfg.get("neighbors", 8)))

    def _build_index(self, n_neighbors: int) -> None:
        """Silhouette IoU nearest neighbours across *different* meshes."""
        b = (F.avg_pool2d(self.masks.reshape(-1, 1, self.resolution, self.resolution), 2) > 0.5).float().flatten(1)  # 32x32
        area = b.sum(-1)
        n = b.shape[0]
        mesh_of = torch.arange(self.M, device=self.device).repeat_interleave(self.K)
        k = min(n_neighbors, max(1, (self.M - 1) * self.K))
        nb_iou, nb_idx, per = [], [], []
        for s in range(0, n, 2048):  # chunked: memory O(chunk * M * K)
            inter = b[s : s + 2048] @ b.T
            iou = inter / (area[s : s + 2048, None] + area[None, :] - inter).clamp_min(1)
            per.append(iou.reshape(-1, self.M, self.K).amax(-1))  # best IoU per other mesh
            if self.M > 1:
                iou_x = iou.masked_fill(mesh_of[s : s + 2048, None] == mesh_of[None, :], -1.0)
                v, i = iou_x.topk(k, dim=-1)
                nb_iou.append(v)
                nb_idx.append(i)
        if self.M > 1:
            self.neighbor_iou, self.neighbors = torch.cat(nb_iou), torch.cat(nb_idx)
        else:
            self.neighbor_iou = torch.zeros(n, 0, device=self.device)
            self.neighbors = torch.zeros(n, 0, dtype=torch.long, device=self.device)
        # mesh-level compatibility: mean over orientations of the best IoU to the other mesh
        self.mesh_compatibility = torch.cat(per).reshape(self.M, self.K, self.M).mean(1)  # [M, M]

    # ------------------------------------------------------------ queries
    def entry(self, flat_idx: torch.Tensor):
        return flat_idx // self.K, flat_idx % self.K

    def flat(self, mesh_id: torch.Tensor, rot_idx: torch.Tensor) -> torch.Tensor:
        return mesh_id * self.K + rot_idx

    def score_patches(self, patches: torch.Tensor, allowed_meshes: torch.Tensor | None = None) -> torch.Tensor:
        """Utility of every bank entry for each patch.

        patches: [P, d, d] utility sampled on the bank frame -> [P, M*K]
        """
        s = patches.reshape(patches.shape[0], -1) @ self.desc.T
        if allowed_meshes is not None:
            mesh_of = torch.arange(self.M, device=self.device).repeat_interleave(self.K)
            s = s.masked_fill(~allowed_meshes[mesh_of][None], float("-inf"))
        return s

    def nearest_rotation(self, R_cam: torch.Tensor) -> torch.Tensor:
        """Index of the closest bank orientation for camera-frame rotations [N,3,3] -> [N]."""
        tr = torch.einsum("nij,kij->nk", R_cam, self.rotations)  # trace(R_k^T R)
        return tr.argmax(-1)

    def similar_entries(self, flat_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Silhouette-compatible entries of *other* meshes: ([N,k] flat indices, [N,k] IoU)."""
        return self.neighbors[flat_idx], self.neighbor_iou[flat_idx]

    def similar_in_mesh(self, flat_idx: int, mesh_id: int, k: int) -> torch.Tensor:
        """The k orientations of ``mesh_id`` whose silhouettes best match entry ``flat_idx`` (IoU)."""
        b = (F.avg_pool2d(self.masks.reshape(-1, 1, self.resolution, self.resolution), 2) > 0.5).float().flatten(1)
        ref = b[flat_idx]
        cand = b[mesh_id * self.K : (mesh_id + 1) * self.K]
        inter = cand @ ref
        iou = inter / (cand.sum(-1) + ref.sum() - inter).clamp_min(1)
        return iou.topk(min(k, self.K)).indices

    def world_rotation(self, rot_idx: torch.Tensor, camera: PerspectiveCamera) -> torch.Tensor:
        Rv = camera.torch_rotation(self.device)
        return Rv.T[None] @ self.rotations[rot_idx]
