"""Piece-type diversity ("randomness") of an assembly.

p_m = fraction of instances that use mesh m, q_m = desired fraction (uniform by
default, or user weights per mesh name).

  randomness  = H(p) / log M            (1 = every type used equally, 0 = one type only)
  deficit     = KL(p || q) / log M      (0 = the counts match the desired distribution)

``deficit`` is what optimizers add to their objective (weight * deficit); it is
a function of the discrete mesh identities only, so it is optimized with
discrete moves (beam child selection, type swaps), never with gradients.
"""

from __future__ import annotations

import math

import torch


def target_distribution(cfg, library) -> torch.Tensor:
    """Desired type distribution q [M] from cfg.diversity.target."""
    M = len(library)
    target = cfg.get("diversity", {}).get("target", "uniform")
    if target in (None, "uniform"):
        q = torch.ones(M)
    else:
        q = torch.tensor([float(target.get(name, 0.0)) for name in library.names])
        if q.sum() <= 0:
            q = torch.ones(M)
    return (q / q.sum()).to(library.device)


def type_counts(mesh_ids: torch.Tensor, M: int) -> torch.Tensor:
    return torch.bincount(mesh_ids.reshape(-1).long(), minlength=M).float()


def diversity_stats(counts: torch.Tensor, q: torch.Tensor) -> dict:
    """counts [..., M] -> randomness and deficit [...] (tensors)."""
    M_eff = int((q > 0).sum().item())
    norm = math.log(max(M_eff, 2))
    n = counts.sum(-1, keepdim=True)
    p = counts / n.clamp_min(1)
    plogp = torch.where(p > 0, p * torch.log(p.clamp_min(1e-12)), torch.zeros_like(p))
    H = -plogp.sum(-1)
    kl = (plogp - torch.where(p > 0, p * torch.log(q.clamp_min(1e-12)), torch.zeros_like(p))).sum(-1)
    empty = n[..., 0] == 0
    return {
        "randomness": torch.where(empty, torch.zeros_like(H), H / norm),
        "deficit": torch.where(empty, torch.zeros_like(kl), kl / norm),
    }


def assembly_diversity(mesh_ids: torch.Tensor, q: torch.Tensor) -> dict:
    s = diversity_stats(type_counts(mesh_ids, len(q)), q)
    return {k: float(v) for k, v in s.items()}
