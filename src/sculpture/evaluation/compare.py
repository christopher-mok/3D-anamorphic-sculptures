"""Experiment runner: shared preprocessing, run methods, identical evaluation, comparison."""

from __future__ import annotations

import csv
import json
import logging
import re
import time
import traceback
from pathlib import Path
from typing import Callable, Sequence

import torch

from ..config import Config
from ..context import ProblemContext, build_context, set_seed
from ..io.result_writer import write_result
from ..methods import get_method
from ..methods.base import MethodResult, OptimizationCancelled, ProgressReporter
from ..scene.assembly import Assembly
from .metrics import evaluate_assembly, selection_score

log = logging.getLogger(__name__)

COMPARISON_FIELDS = [
    "method", "status", "score", "mean_iou", "min_view_iou", "view_iou", "spill", "coverage", "collisions",
    "containment_violations", "clearance_violations", "randomness", "object_count", "runtime_s", "renderer_calls", "loss",
]


def new_run_dir(root) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for p in root.iterdir() if (m := re.fullmatch(r"run_(\d+)", p.name))]
    d = root / f"run_{(max(nums) + 1 if nums else 1):03d}"
    d.mkdir()
    return d


def comparison_rows(results: dict, cfg: Config) -> tuple[list[dict], str | None]:
    rows = []
    for name, r in results.items():
        m = r.get("metrics")
        if not m:
            rows.append({"method": name, "status": r.get("status", "failed")})
            continue
        rows.append(
            {
                "method": name,
                "status": r["status"],
                "score": selection_score(m, cfg),
                "mean_iou": m["mean_iou"],
                "min_view_iou": m["min_view_iou"],
                "view_iou": m["view_iou"],
                "spill": m["spill"],
                "coverage": m["recall"],
                "collisions": m["collisions"],
                "containment_violations": m["containment_violations"],
                "clearance_violations": m.get("clearance_violations", 0),
                "randomness": m.get("randomness", 0.0),
                "type_counts": m.get("type_counts", {}),
                "object_count": m["object_count"],
                "runtime_s": m["runtime_s"],
                "renderer_calls": m["renderer_calls"],
                "loss": m["loss"],
            }
        )
    scored = [r for r in rows if "score" in r]
    best = max(scored, key=lambda r: r["score"])["method"] if scored else None
    return rows, best


def write_comparison(folder: Path, rows: list[dict], best: str | None, cfg: Config) -> dict:
    payload = {"selection_metric": cfg.evaluation.selection_metric, "best_method": best, "rows": rows}
    (folder / "comparison.json").write_text(json.dumps(payload, indent=2))
    with open(folder / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COMPARISON_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    return payload


def run_method(ctx: ProblemContext, name: str, folder: Path, reporter: ProgressReporter | None = None, resume: bool = False) -> dict:
    """Run one method and evaluate it with the shared final renderer + metrics."""
    folder = Path(folder)
    set_seed(int(ctx.cfg.seed))
    cls = get_method(name)
    try:
        result: MethodResult = cls(ctx, folder, reporter, resume=resume).run()
    except Exception as exc:  # keep the experiment going; record the failure
        log.error("method %s failed: %s", name, exc)
        (folder / "error.txt").write_text(traceback.format_exc())
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    metrics = evaluate_assembly(ctx, result.assembly, result.runtime_s, result.renderer_calls)
    write_result(ctx, result, metrics, folder)
    return {"status": result.status, "metrics": metrics, "info": result.info, "result": result}


def run_experiment(
    cfg: Config,
    models_dir,
    target_paths: Sequence,
    methods: Sequence[str],
    output_dir=None,
    cameras=None,
    reporter: ProgressReporter | None = None,
    resume: bool = False,
    ctx: ProblemContext | None = None,
    on_preprocessed: Callable[[ProblemContext, dict, Path], None] | None = None,
    on_method_done: Callable[[str, dict], None] | None = None,
    on_method_start: Callable[[str, Path], None] | None = None,
    initial: dict | None = None,
) -> dict:
    out = Path(output_dir) if output_dir else new_run_dir(cfg.output_root)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if ctx is None:
        ctx = build_context(cfg, models_dir, target_paths, cameras)
    if initial is not None:
        from ..context import load_initial_assembly

        ctx.initial_assembly = load_initial_assembly(ctx, **initial)
    diag = ctx.save_preprocessing(out / "preprocessing")
    if on_preprocessed:
        on_preprocessed(ctx, diag, out)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    results = {}
    for name in methods:
        if on_method_start:
            on_method_start(name, out / name)
        results[name] = run_method(ctx, name, out / name, reporter, resume)
        if on_method_done:
            on_method_done(name, results[name])
        if results[name]["status"] == "cancelled":
            break
    rows, best = comparison_rows(results, cfg)
    comparison = write_comparison(out, rows, best, cfg)
    log.info("comparison written to %s (best: %s, %.1fs total)", out / "comparison.json", best, time.time() - t0)
    return {"output_dir": out, "comparison": comparison, "results": results, "diagnostics": diag, "ctx": ctx}
