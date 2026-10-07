"""Column pool and the coverage master problem (Method 2).

Variables:  x_c in {0,1} (placement c selected)
            y_p in [0,1]  (target pixel p covered), p in foreground pixels touched by some column
            z_p in [0,1]  (background pixel p spilled), p in background pixels touched by some column
maximize    sum_p a_p y_p  -  sum_p b_p z_p
            a_p = w_cov / (V |I_v|),   b_p = w_neg / (V |B_v|)
s.t.        y_p <= sum_{c covers p} x_c           (union coverage)
            z_p >= x_c   for every c spilling on p (union spill)
            sum_c x_c <= max_objects             (safety bound only; NOT in the objective)
            x_i + x_j <= 1                        (lazy collision conflicts)

For binary masks the objective equals  w_cov - L_image  exactly, so the MILP
optimizes the same silhouette loss as every other method (at its resolution).
Piece count never appears in the objective.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from ...scene.assembly import Assembly
from ...targets.pyramids import downsample_area
from .solver_base import LinearProgram, LPResult


class ColumnPool:
    """Discovered placements with their binary silhouette masks at the MILP resolution."""

    def __init__(self, ctx, resolution: int, render_resolution: int, threshold: float = 0.5):
        self.ctx = ctx
        self.res = int(resolution)
        self.rres = int(render_resolution)
        self.threshold = float(threshold)
        self.V = ctx.num_views
        self.P = self.V * self.res * self.res
        self.mesh_ids = torch.zeros(0, dtype=torch.long)
        self.translation = torch.zeros(0, 3)
        self.rot6d = torch.zeros(0, 6)
        self.log_scale = torch.zeros(0)
        self.rows: list[np.ndarray] = []   # flat pixel indices covered by each column
        self.source: list[str] = []
        self._hashes: set[str] = set()
        fg = ctx.targets.mask(self.res).reshape(-1).cpu().numpy() > 0.5
        self.fg = fg

    def __len__(self) -> int:
        return len(self.rows)

    @torch.no_grad()
    def masks(self, a: Assembly) -> torch.Tensor:
        """Binary masks [N, V*res*res] at the MILP resolution (rendered at rres, area-pooled)."""
        out = []
        for s in range(0, len(a), 512):
            S = self.ctx.render_instances(a[list(range(s, min(s + 512, len(a))))], self.rres)
            out.append((downsample_area(S, self.res) >= self.threshold).reshape(S.shape[0], -1))
        return torch.cat(out) if out else torch.zeros(0, self.P, dtype=torch.bool, device=self.ctx.device)

    def add(self, a: Assembly, source: str, masks: torch.Tensor | None = None) -> int:
        """Add placements (deduplicated by mask, zero-coverage dropped). Returns #added."""
        if len(a) == 0:
            return 0
        if masks is None:
            masks = self.masks(a)
        m_np = masks.cpu().numpy()
        keep = []
        for i in range(len(a)):
            idx = np.flatnonzero(m_np[i])
            if idx.size == 0 or not self.fg[idx].any():
                continue
            h = hashlib.blake2b(idx.astype(np.int32).tobytes(), digest_size=12).hexdigest()
            if h in self._hashes:
                continue
            self._hashes.add(h)
            keep.append(i)
            self.rows.append(idx)
            self.source.append(source)
        if keep:
            k = torch.tensor(keep, device=a.device)
            sub = a[k].detach().to("cpu")
            self.mesh_ids = torch.cat([self.mesh_ids, sub.mesh_ids])
            self.translation = torch.cat([self.translation, sub.translation])
            self.rot6d = torch.cat([self.rot6d, sub.rot6d])
            self.log_scale = torch.cat([self.log_scale, sub.log_scale])
        return len(keep)

    def assembly(self, idx) -> Assembly:
        idx = torch.as_tensor(np.asarray(idx, dtype=np.int64))
        a = Assembly(self.mesh_ids[idx], self.translation[idx], self.rot6d[idx], self.log_scale[idx])
        return a.to(self.ctx.device)

    def incidence(self) -> sp.csr_matrix:
        """[C, P] binary incidence matrix."""
        indptr = np.zeros(len(self.rows) + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(r) for r in self.rows])
        indices = np.concatenate(self.rows) if self.rows else np.zeros(0, np.int64)
        return sp.csr_matrix((np.ones(len(indices)), indices, indptr), shape=(len(self.rows), self.P))

    def state(self) -> dict:
        return {
            "mesh_ids": self.mesh_ids.numpy(), "translation": self.translation.numpy(),
            "rot6d": self.rot6d.numpy(), "log_scale": self.log_scale.numpy(), "source": np.array(self.source),
        }


@dataclass
class MasterLayout:
    C: int
    fg_pix: np.ndarray      # pixel ids with a y variable
    bg_pix: np.ndarray      # pixel ids with a z variable
    spill_pairs: np.ndarray # [S, 2] (column, bg pixel id) for each spill row


class MasterProblem:
    def __init__(self, pool: ColumnPool, w_cov: float, w_neg: float, max_objects: int):
        self.pool = pool
        ctx = pool.ctx
        I = ctx.targets.mask(pool.res).reshape(pool.V, -1).cpu().numpy() > 0.5
        nI = I.sum(1)
        nB = (~I).sum(1)
        V = pool.V
        a = np.where(I, w_cov / (V * np.maximum(nI, 1))[:, None], 0.0)
        b = np.where(~I, w_neg / (V * np.maximum(nB, 1))[:, None], 0.0)
        self.a = a.reshape(-1)   # fg value per pixel
        self.b = b.reshape(-1)   # bg cost per pixel
        self.w_cov = float(w_cov)
        self.max_objects = int(max_objects)
        self.conflicts: set[tuple[int, int]] = set()

    # ------------------------------------------------------------ build
    def build(self) -> tuple[LinearProgram, MasterLayout]:
        pool = self.pool
        A = pool.incidence()  # [C, P]
        C = A.shape[0]
        fg_mask = pool.fg
        touched = np.asarray(A.sum(0)).ravel() > 0
        fg_pix = np.flatnonzero(touched & fg_mask)
        bg_pix = np.flatnonzero(touched & ~fg_mask)
        nF, nZ = len(fg_pix), len(bg_pix)
        n = C + nF + nZ
        c = np.concatenate([np.zeros(C), -self.a[fg_pix], self.b[bg_pix]])

        blocks, b_ub, groups, r0 = [], [], {}, 0
        # coverage rows: y_p - sum_c A[c,p] x_c <= 0
        Acov = A[:, fg_pix].T.tocsr()  # [nF, C]
        cov = sp.hstack([-Acov, sp.identity(nF, format="csr"), sp.csr_matrix((nF, nZ))], format="csr")
        blocks.append(cov)
        b_ub.append(np.zeros(nF))
        groups["coverage"] = slice(r0, r0 + nF)
        r0 += nF
        # spill rows: x_c - z_p <= 0
        Abg = A[:, bg_pix].tocoo()
        S = Abg.nnz
        spill_pairs = np.stack([Abg.row, Abg.col], 1) if S else np.zeros((0, 2), np.int64)
        if S:
            rows = np.arange(S)
            data = np.concatenate([np.ones(S), -np.ones(S)])
            cols = np.concatenate([Abg.row, C + nF + Abg.col])
            blocks.append(sp.csr_matrix((data, (np.concatenate([rows, rows]), cols)), shape=(S, n)))
            b_ub.append(np.zeros(S))
        groups["spill"] = slice(r0, r0 + S)
        r0 += S
        # cardinality safety bound
        blocks.append(sp.csr_matrix((np.ones(C), (np.zeros(C, int), np.arange(C))), shape=(1, n)))
        b_ub.append(np.array([float(self.max_objects)]))
        groups["cardinality"] = slice(r0, r0 + 1)
        r0 += 1
        # conflicts
        conf = [p for p in self.conflicts if p[0] < C and p[1] < C]
        if conf:
            K = len(conf)
            ij = np.array(conf)
            rows = np.repeat(np.arange(K), 2)
            blocks.append(sp.csr_matrix((np.ones(2 * K), (rows, ij.reshape(-1))), shape=(K, n)))
            b_ub.append(np.ones(K))
        groups["conflicts"] = slice(r0, r0 + len(conf))
        A_ub = sp.vstack(blocks, format="csr")
        lp = LinearProgram(c, A_ub, np.concatenate(b_ub), C, groups)
        layout = MasterLayout(C, fg_pix, bg_pix, spill_pairs[:, :] if S else spill_pairs)
        if S:
            layout.spill_pairs = np.stack([Abg.row, bg_pix[Abg.col]], 1)
        return lp, layout

    # ------------------------------------------------------------ objectives
    def objective_value(self, x_cols: np.ndarray) -> float:
        """Exact maximization objective for a 0/1 column selection."""
        sel = np.flatnonzero(x_cols > 0.5)
        if sel.size == 0:
            return 0.0
        cov = np.zeros(self.pool.P, bool)
        for c in sel:
            cov[self.pool.rows[c]] = True
        return float(self.a[cov].sum() - self.b[cov].sum())

    def loss_from_objective(self, obj_max: float) -> float:
        """L_image (binary, MILP resolution) = w_cov - objective."""
        return self.w_cov - obj_max

    # ------------------------------------------------------------ duals -> pricing map
    def pricing_map(self, lp: LinearProgram, layout: MasterLayout, res: LPResult) -> tuple[np.ndarray, float]:
        """Image-space dual weights W[p] and cardinality dual sigma.

        A new column with binary mask S has reduced profit  sum_p W_p S_p - sigma:
          fg pixel with a coverage row: W = u_p (dual of y_p <= sum x),
          fg pixel never covered:       W = a_p (covering it is worth its full value),
          bg pixel already spilled:     W = -(b_p - sum_c w_cp)  (remaining spill price),
          bg pixel never spilled:       W = -b_p.
        """
        P = self.pool.P
        W = self.a - self.b  # defaults: untouched pixels
        d = res.duals
        if d is None:
            return W, 0.0
        g = lp.row_groups
        u = -d[g["coverage"]]
        W[layout.fg_pix] = np.maximum(u, 0.0)
        spill_d = -d[g["spill"]]
        if len(spill_d):
            paid = np.zeros(P)
            np.add.at(paid, layout.spill_pairs[:, 1], np.maximum(spill_d, 0.0))
            W[layout.bg_pix] = -np.maximum(self.b[layout.bg_pix] - paid[layout.bg_pix], 0.0)
        sigma = float(max(-d[g["cardinality"]][0], 0.0))
        return W, sigma


def greedy_selection(master: MasterProblem, x_lp: np.ndarray, max_objects: int) -> np.ndarray:
    """Feasible 0/1 incumbent: add columns in decreasing LP value if they do not
    conflict with already chosen ones and strictly increase the objective."""
    pool = master.pool
    C = len(pool)
    order = np.argsort(-x_lp[:C], kind="stable")
    order = order[x_lp[order] > 1e-6]
    conflicts: dict[int, set[int]] = {}
    for i, j in master.conflicts:
        conflicts.setdefault(i, set()).add(j)
        conflicts.setdefault(j, set()).add(i)
    covered = np.zeros(pool.P, bool)
    chosen: list[int] = []
    for c in order:
        if len(chosen) >= max_objects:
            break
        if conflicts.get(int(c), set()) & set(chosen):
            continue
        idx = pool.rows[c]
        new = idx[~covered[idx]]
        gain = master.a[new].sum() - master.b[new].sum()
        if gain > 0:
            chosen.append(int(c))
            covered[idx] = True
    x = np.zeros(C)
    x[chosen] = 1.0
    return x
