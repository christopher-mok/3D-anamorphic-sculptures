"""Background job system: one GPU worker thread, in-memory job registry."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import PROJECT_ROOT, load_config
from ..methods import METHOD_NAMES
from ..methods.base import ProgressReporter
from .schemas import JobRequest
from .websocket import Broadcaster

log = logging.getLogger(__name__)


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def outputs_root() -> Path:
    root = resolve_path(load_config().output_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def output_url(path: Path) -> str:
    rel = Path(path).resolve().relative_to(outputs_root().resolve())
    return "/outputs/" + rel.as_posix()


def _empty_progress(status="pending") -> dict:
    return {
        "status": status, "iteration": 0, "phase": "", "runtime_s": 0.0, "loss": None, "view_iou": [],
        "object_count": 0, "best_metric": None, "error": None, "preview_urls": [], "assembly": None,
        "metrics": None, "result_dir_url": None, "info": {},
    }


class Job:
    def __init__(self, request: JobRequest):
        self.id = uuid.uuid4().hex[:12]
        self.request = request
        self.status = "queued"
        self.error: str | None = None
        self.created_at = time.time()
        self.output_dir: Path | None = None
        self.preprocessing: dict | None = None
        self.methods = {m: _empty_progress() for m in request.methods}
        self.comparison: dict | None = None
        self.cancel_event = threading.Event()
        self.events = Broadcaster()
        self.lock = threading.RLock()

    def to_dict(self) -> dict:
        with self.lock:
            return {
                "job_id": self.id,
                "status": self.status,
                "error": self.error,
                "created_at": self.created_at,
                "output_dir_url": output_url(self.output_dir) if self.output_dir else None,
                "request": self.request.model_dump(),
                "preprocessing": self.preprocessing,
                "methods": {k: dict(v) for k, v in self.methods.items()},
                "comparison": self.comparison,
            }

    def publish_status(self) -> None:
        self.events.publish({"type": "status", "data": self.to_dict()})


class JobReporter(ProgressReporter):
    """Bridges optimizer progress events to job state + WebSocket stream."""

    def __init__(self, job: Job):
        self.job = job
        self.method_dirs: dict[str, Path] = {}

    def cancelled(self) -> bool:
        return self.job.cancel_event.is_set()

    def update(self, method: str, event: dict) -> None:
        job = self.job
        urls = []
        preview = event.get("preview")
        folder = self.method_dirs.get(method)
        if preview is not None and folder is not None:
            live = folder / "live"
            live.mkdir(parents=True, exist_ok=True)
            for v, m in enumerate(preview):
                p = live / f"view_{v}.png"
                Image.fromarray(((1.0 - np.clip(m, 0, 1)) * 255).astype(np.uint8)).save(p)
                urls.append(output_url(p))
        with job.lock:
            pr = job.methods.setdefault(method, _empty_progress())
            pr.update(
                status="running", iteration=event.get("iteration", 0), phase=event.get("phase", ""),
                runtime_s=float(event.get("runtime_s", 0.0)), loss=event.get("loss"),
                view_iou=list(event.get("view_iou") or []), object_count=int(event.get("object_count") or 0),
                best_metric=event.get("best_metric"), info=_jsonable(event.get("info", {})),
            )
            if urls:
                pr["preview_urls"] = urls
            if event.get("assembly") is not None:
                pr["assembly"] = event["assembly"]
            data = dict(pr)
        job.events.publish({"type": "progress", "method": method, "data": data})


def _jsonable(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (int, float, str, bool)) else str(x) for x in v]
        else:
            out[k] = str(v)
    return out


def _rel(p: str) -> str:
    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def job_from_run_dir(d: Path) -> Job | None:
    """Reconstruct a finished Job from a run folder written by run_experiment."""
    diag_file = d / "preprocessing" / "diagnostics.json"
    if not diag_file.exists():
        return None
    diag = json.loads(diag_file.read_text())
    results = {m: json.loads((d / m / "result.json").read_text()) for m in METHOD_NAMES if (d / m / "result.json").exists()}
    if not results:
        return None
    n_views = len(diag["cameras"])
    request = JobRequest(
        models_dir="assets/models",
        targets=[_rel(t) for t in diag.get("targets") or []] or ["assets/targets/view_0.png"],
        cameras=diag["cameras"],
        methods=list(results.keys()),
        preset="default",
    )
    job = Job(request)
    job.id = d.name
    job.status = "done"
    job.created_at = d.stat().st_mtime
    job.output_dir = d
    pre = d / "preprocessing"
    job.preprocessing = {
        "target_urls": [output_url(pre / f"target_{i}.png") for i in range(n_views)],
        "cameras": diag["cameras"],
        "bounding_volume": diag["bounding_volume"],
        "hull_url": output_url(pre / "hull_preview.obj") if (pre / "hull_preview.obj").exists() else None,
        "hull_voxels": diag.get("hull_voxels", 0),
        "strict_coverage_upper_bound": diag["compatibility"]["strict_coverage_upper_bound"],
        "warnings": diag["compatibility"].get("warnings", []),
    }
    for m, r in results.items():
        met = r.get("metrics") or {}
        pr = _empty_progress(r.get("status", "done"))
        pr.update(
            phase="done", runtime_s=float(met.get("runtime_s", 0.0)), loss=met.get("loss"),
            view_iou=met.get("view_iou", []), object_count=int(met.get("object_count", 0)),
            best_metric=met.get("min_view_iou"), metrics=met or None, assembly=r.get("assembly"),
            result_dir_url=output_url(d / m), info=_jsonable(r.get("info", {})),
            preview_urls=[output_url(d / m / "renders" / f"view_{v}.png") for v in range(n_views)],
        )
        job.methods[m] = pr
    comp_file = d / "comparison.json"
    if comp_file.exists():
        job.comparison = json.loads(comp_file.read_text())
    return job


class JobManager:
    def __init__(self, library_provider):
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[Job] = queue.Queue()
        self.library_provider = library_provider  # (models_dir, mesh_cfg) -> (library, renderer)
        self.worker = threading.Thread(target=self._loop, daemon=True, name="sculpture-gpu-worker")
        self.worker.start()

    def submit(self, request: JobRequest) -> Job:
        if not 1 <= len(request.targets) <= 2:
            raise ValueError("one or two targets are required")
        if request.cameras is not None and len(request.cameras) != len(request.targets):
            raise ValueError("need exactly one camera per target")
        bad = [m for m in request.methods if m not in METHOD_NAMES]
        if bad or not request.methods:
            raise ValueError(f"invalid methods {bad}")
        job = Job(request)
        self.jobs[job.id] = job
        self.queue.put(job)
        return job

    def load_past_runs(self, root: Path, depth: int = 2) -> int:
        """Register finished runs found under ``root`` (e.g. produced by the CLI) as done jobs.

        A run folder is any directory with preprocessing/diagnostics.json and at least one
        <method>/result.json, up to ``depth`` levels below root (so outputs/experiments/<name>
        is found too). Its job id is the path relative to root with "/" replaced by "--".
        Folders already owned by an in-memory job are skipped. Returns the number added.
        """
        root = Path(root)
        if not root.is_dir():
            return 0
        known = {j.output_dir.resolve() for j in self.jobs.values() if j.output_dir}
        added = 0
        frontier = [root]
        for _ in range(depth):
            nxt = []
            for parent in frontier:
                for d in sorted(p for p in parent.iterdir() if p.is_dir()):
                    if (d / "preprocessing" / "diagnostics.json").exists():
                        job_id = d.relative_to(root).as_posix().replace("/", "--")
                        if d.resolve() in known or job_id in self.jobs:
                            continue
                        try:
                            job = job_from_run_dir(d)
                        except Exception as exc:  # incomplete / foreign folders are ignored
                            log.debug("skipping %s: %s", d, exc)
                            continue
                        if job is not None:
                            job.id = job_id
                            self.jobs[job.id] = job
                            added += 1
                    elif d.name not in METHOD_NAMES and d.name != "preprocessing":
                        nxt.append(d)
            frontier = nxt
        return added

    def cancel(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.cancel_event.set()
        with job.lock:
            if job.status == "queued":
                job.status = "cancelled"
        job.publish_status()

    def _loop(self) -> None:
        while True:
            job = self.queue.get()
            if job.cancel_event.is_set():
                continue
            try:
                self._run(job)
            except Exception as exc:  # pragma: no cover - surfaced to the UI
                log.exception("job %s failed", job.id)
                with job.lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    for pr in job.methods.values():
                        if pr["status"] in ("pending", "running"):
                            pr["status"] = "failed"
                job.events.publish({"type": "error", "data": {"message": job.error, "traceback": traceback.format_exc()}})
                job.events.publish({"type": "done", "data": job.to_dict()})

    def _run(self, job: Job) -> None:
        from ..context import build_context
        from ..evaluation.compare import new_run_dir, run_experiment
        from ..scene.serialization import assembly_to_dict

        req = job.request
        cfg = load_config(req.preset, overrides=req.overrides or None)
        cfg.output_root = str(outputs_root())
        cfg.cache_dir = str(resolve_path(cfg.cache_dir))
        models_dir = resolve_path(req.models_dir)
        targets = [resolve_path(t) for t in req.targets]
        cams = [c.model_dump() for c in req.cameras] if req.cameras else None
        with job.lock:
            job.status = "preprocessing"
            job.output_dir = new_run_dir(cfg.output_root)
        job.publish_status()

        library, renderer = self.library_provider(models_dir, cfg.meshes)
        ctx = build_context(cfg, models_dir, targets, cams, library=library, renderer=renderer)
        reporter = JobReporter(job)

        def on_pre(ctx_, diag, out):
            pre = out / "preprocessing"
            with job.lock:
                job.preprocessing = {
                    "target_urls": [output_url(pre / f"target_{i}.png") for i in range(ctx_.num_views)],
                    "cameras": diag["cameras"],
                    "bounding_volume": diag["bounding_volume"],
                    "hull_url": output_url(pre / "hull_preview.obj"),
                    "hull_voxels": diag["hull_voxels"],
                    "strict_coverage_upper_bound": diag["compatibility"]["strict_coverage_upper_bound"],
                    "warnings": diag["compatibility"]["warnings"],
                }
                job.status = "running"
            job.publish_status()

        def on_start(name, folder):
            reporter.method_dirs[name] = folder
            with job.lock:
                job.methods[name]["status"] = "running"
                job.methods[name]["phase"] = "starting"
            job.publish_status()

        def on_done(name, res):
            with job.lock:
                pr = job.methods[name]
                pr["status"] = res["status"]
                pr["error"] = res.get("error")
                if res.get("metrics"):
                    m = res["metrics"]
                    pr["metrics"] = m
                    pr["view_iou"] = m["view_iou"]
                    pr["loss"] = m["loss"]
                    pr["object_count"] = m["object_count"]
                    pr["runtime_s"] = m["runtime_s"]
                    pr["best_metric"] = m["min_view_iou"]
                    pr["result_dir_url"] = output_url(job.output_dir / name)
                    pr["preview_urls"] = [output_url(job.output_dir / name / "renders" / f"view_{v}.png") for v in range(ctx.num_views)]
                    pr["assembly"] = assembly_to_dict(res["result"].assembly, ctx.library)
                    pr["info"] = _jsonable(res.get("info", {}))
                    pr["phase"] = "done"
            job.events.publish({"type": "progress", "method": name, "data": dict(job.methods[name])})

        out = run_experiment(
            cfg, models_dir, targets, list(req.methods), output_dir=job.output_dir, cameras=cams, reporter=reporter,
            ctx=ctx, on_preprocessed=on_pre, on_method_done=on_done, on_method_start=on_start,
        )
        with job.lock:
            job.comparison = out["comparison"]
            cancelled = job.cancel_event.is_set()
            job.status = "cancelled" if cancelled else "done"
            for pr in job.methods.values():
                if pr["status"] == "pending":
                    pr["status"] = "cancelled" if cancelled else "skipped"
        job.events.publish({"type": "comparison", "data": job.comparison})
        job.events.publish({"type": "done", "data": job.to_dict()})
