"""FastAPI application. Run with:  uvicorn sculpture.server.app:app --reload

Optimization runs in a background worker thread; HTTP requests never block on
it. Progress is streamed over /api/jobs/{id}/ws (see docs/api.md).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import shutil
import threading
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ..config import PRESETS, PROJECT_ROOT, load_config
from ..geometry.mesh_library import MeshLibrary
from ..methods import METHOD_NAMES
from .jobs import JobManager, outputs_root, resolve_path
from .schemas import JobRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
UPLOADS = PROJECT_ROOT / "uploads"
TARGETS = PROJECT_ROOT / "assets" / "targets"

# ------------------------------------------------------------------ libraries
_lib_lock = threading.Lock()
_libraries: dict[str, tuple[MeshLibrary, object]] = {}
_preview_renderers: dict[str, object] = {}
_preview_lock = threading.Lock()


def _lib_key(models_dir: Path, mesh_cfg, model_names=None) -> str:
    keys = {k: mesh_cfg.get(k) for k in ("proxy_max_faces", "coarse_max_faces", "surface_samples", "extensions")}
    files = sorted((p.name, p.stat().st_mtime, p.stat().st_size) for p in models_dir.iterdir() if p.is_file())
    selected = sorted(model_names) if model_names is not None else None
    return hashlib.sha1(json.dumps([str(models_dir.resolve()), keys, files, selected]).encode()).hexdigest()


def get_library(models_dir: Path, mesh_cfg, model_names=None):
    """Shared, cached (library, renderer) per model folder."""
    from ..rendering.nvdiffrast_renderer import NvdiffrastRenderer

    models_dir = resolve_path(models_dir)
    key = _lib_key(models_dir, mesh_cfg, model_names)
    with _lib_lock:
        if key not in _libraries:
            cfg = load_config()
            lib = MeshLibrary.from_folder(
                models_dir, mesh_cfg, device="cuda", cache_dir=resolve_path(cfg.cache_dir), model_names=model_names,
            )
            _libraries[key] = (lib, NvdiffrastRenderer(lib, "cuda"))
        return _libraries[key]


def _preview_renderer(lib: MeshLibrary):
    """A separate rasterizer context for thumbnails so HTTP threads never share the job's context."""
    from ..rendering.nvdiffrast_renderer import NvdiffrastRenderer

    key = str(id(lib))
    if key not in _preview_renderers:
        _preview_renderers[key] = NvdiffrastRenderer(lib, "cuda")
    return _preview_renderers[key]


# ------------------------------------------------------------------ app
app = FastAPI(title="Anamorphic Sculpture Designer", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
jobs = JobManager(get_library)
jobs.load_past_runs(outputs_root())  # finished CLI / earlier-server runs appear as done jobs
app.mount("/outputs", StaticFiles(directory=str(outputs_root())), name="outputs")


def _gurobi_available() -> bool:
    try:
        import gurobipy  # noqa: F401

        return True
    except Exception:
        return False


@app.get("/api/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available(), "gurobi": _gurobi_available()}


@app.get("/api/defaults")
def defaults():
    cfg = load_config()
    targets = [str(p.relative_to(PROJECT_ROOT).as_posix()) for p in (TARGETS / "view_0.png", TARGETS / "view_1.png") if p.exists()]
    return {
        "models_dir": "assets/models",
        "cameras": cfg.cameras,
        "bounding_volume": cfg.bounding_volume,
        "presets": list(PRESETS),
        "methods": list(METHOD_NAMES),
        "targets": targets,
    }


# ------------------------------------------------------------------ models
def _models_dir(dir: str) -> Path:
    d = resolve_path(dir)
    if not d.is_dir():
        raise HTTPException(404, f"model folder {dir!r} not found")
    return d


@app.get("/api/models")
def list_models(dir: str = Query("assets/models")):
    d = _models_dir(dir)
    try:
        lib, _ = get_library(d, load_config().meshes)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    out = []
    for e in lib.entries:
        info = e.info()
        q = f"?dir={dir}"
        info["thumbnail_url"] = f"/api/models/{e.name}/thumbnail.png{q}"
        info["mesh_url"] = f"/api/models/{e.name}/mesh.glb{q}"
        out.append(info)
    return out


def _entry(name: str, dir: str):
    lib, _ = get_library(_models_dir(dir), load_config().meshes)
    try:
        return lib, lib.by_name(name)
    except KeyError:
        raise HTTPException(404, f"model {name!r} not found")


@app.get("/api/models/{name}/mesh.glb")
def model_mesh(name: str, dir: str = Query("assets/models"), lod: str = Query("proxy")):
    lib, e = _entry(name, dir)
    if lod not in ("proxy", "original", "coarse"):
        raise HTTPException(400, "lod must be proxy | original | coarse")
    data = e.canonical_trimesh(lod).export(file_type="glb")
    return Response(content=data, media_type="model/gltf-binary", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/models/{name}/thumbnail.png")
def model_thumbnail(name: str, dir: str = Query("assets/models"), size: int = Query(128)):
    lib, e = _entry(name, dir)
    size = max(32, min(512, int(size) // 8 * 8))
    with _preview_lock:
        img = _preview_renderer(lib).render_mesh_preview(e.mesh_id, size)
    buf = io.BytesIO()
    Image.fromarray(img, "RGBA").save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


# ------------------------------------------------------------------ targets / files
def _target_record(p: Path) -> dict:
    rel = p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    return {"path": rel, "url": f"/api/files?path={rel}", "name": p.name}


@app.get("/api/targets")
def list_targets():
    out = []
    for folder in (TARGETS, UPLOADS):
        if folder.is_dir():
            out += [_target_record(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return out


@app.post("/api/targets/upload")
async def upload_target(file: UploadFile = File(...)):
    suffix = Path(file.filename or "target.png").suffix.lower()
    if suffix not in IMAGE_EXTS:
        raise HTTPException(400, f"unsupported image type {suffix}")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in Path(file.filename or "target").stem if c.isalnum() or c in "-_")[:40] or "target"
    dest = UPLOADS / f"{uuid.uuid4().hex[:8]}_{safe}{suffix}"
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        Image.open(dest).verify()
    except Exception:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "file is not a readable image")
    return _target_record(dest)


@app.get("/api/files")
def get_file(path: str):
    p = resolve_path(path).resolve()
    allowed = [PROJECT_ROOT / "assets", UPLOADS, outputs_root()]
    if not any(p.is_relative_to(a.resolve()) for a in allowed) or not p.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(p)


# ------------------------------------------------------------------ jobs
@app.post("/api/jobs")
def create_job(req: JobRequest):
    for t in req.targets:
        if not resolve_path(t).is_file():
            raise HTTPException(400, f"target {t!r} not found")
    if not resolve_path(req.models_dir).is_dir():
        raise HTTPException(400, f"model folder {req.models_dir!r} not found")
    try:
        job = jobs.submit(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"job_id": job.id}


@app.get("/api/jobs")
def list_jobs():
    jobs.load_past_runs(outputs_root())  # pick up runs written since startup
    return [j.to_dict() for j in sorted(jobs.jobs.values(), key=lambda j: -j.created_at)]


def _job(job_id: str):
    if job_id not in jobs.jobs:
        raise HTTPException(404, "job not found")
    return jobs.jobs[job_id]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job(job_id).to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    _job(job_id)
    jobs.cancel(job_id)
    return {"ok": True}


@app.websocket("/api/jobs/{job_id}/ws")
async def job_ws(ws: WebSocket, job_id: str):
    if job_id not in jobs.jobs:
        await ws.close(code=4404)
        return
    job = jobs.jobs[job_id]
    await ws.accept()
    q = job.events.subscribe()
    try:
        await ws.send_text(json.dumps({"type": "status", "data": job.to_dict()}, default=str))
        if job.status in ("done", "failed", "cancelled"):
            await ws.send_text(json.dumps({"type": "done", "data": job.to_dict()}, default=str))
        while True:
            try:
                text = await asyncio.wait_for(q.get(), timeout=15.0)
                await ws.send_text(text)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "status", "data": job.to_dict()}, default=str))  # keep-alive
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        job.events.unsubscribe(q)
