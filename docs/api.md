# Backend HTTP / WebSocket API

The backend is a FastAPI app (`uvicorn sculpture.server.app:app --port 8000`).
The Vite dev server proxies `/api` (including WebSockets) and `/outputs` to
`http://localhost:8000`.

All paths below are relative to the server root. All JSON uses snake_case.

## Geometric conventions (important for the 3D viewer)

* World space is right-handed, **+Y up**. Units are arbitrary; the default
  sculpture bounding volume is the cube `[-1, 1]^3`.
* A camera is
  ```json
  {"position": [0, 0, 5], "look_at": [0, 0, 0], "up": [0, 1, 0],
   "fov_y_deg": 25.0, "near": 0.1, "far": 20.0}
  ```
  It follows the OpenGL / three.js convention: it looks from `position`
  toward `look_at`, `up` is the approximate up vector, `fov_y_deg` is the
  **vertical** field of view in degrees. Optimization images are always
  **square** (aspect = 1).
  To reproduce the exact optimization view in three.js: set
  `camera.position`, `camera.up`, `camera.lookAt(look_at)`, `camera.fov = fov_y_deg`,
  `near`, `far`, `camera.aspect = canvasWidth / canvasHeight`.
  The square optimization image then corresponds to the **centered square of
  side `min(canvasWidth, canvasHeight)`**. If the canvas is landscape
  (aspect >= 1) use `camera.fov = fov_y_deg`; if it is portrait (aspect < 1) use
  `camera.fov = 2*atan(tan(fov_y_deg/2) / aspect)` (degrees) so the square still
  spans exactly `fov_y_deg`. Draw the translucent target overlay in that square.
* Every model served by the backend is already **canonically normalized**:
  centered at its bounding-box center and scaled so its bounding-sphere radius
  is 1. An object instance places a model in the world with
  `x_world = scale * (R @ x_local) + translation`.
  In three.js: `object.position.set(...translation)`,
  `object.setRotationFromMatrix(new Matrix4().set(R[0][0],R[0][1],R[0][2],0, R[1][0],...,0, 0,0,0,1))`,
  `object.scale.setScalar(scale)`.
  `rotation_matrix` is row-major (`R[row][col]`).

## Types

```ts
type Vec3 = [number, number, number];

interface Camera {
  position: Vec3; look_at: Vec3; up: Vec3;
  fov_y_deg: number; near: number; far: number;
}

interface ModelInfo {
  id: number;               // index in the (sorted) model folder
  name: string;             // stem, unique key used in URLs
  filename: string;         // e.g. "bunny.obj"
  triangles: number;        // original triangle count
  proxy_triangles: number;  // rendering proxy triangle count
  dims: Vec3;               // normalized bounding-box extents (radius-1 sphere)
  original_dims: Vec3;      // bounding-box extents in source units
  thumbnail_url: string;    // PNG
  mesh_url: string;         // GLB of the canonical normalized mesh
}

interface ObjectInstance {
  mesh_id: number;
  mesh_name: string;
  source_file: string;
  translation: Vec3;
  rotation_matrix: [Vec3, Vec3, Vec3];  // row-major
  rotation6d: number[];                 // 6 numbers (first two columns of R)
  log_scale: number;
  scale: number;
}

interface Assembly {
  objects: ObjectInstance[];
}

interface Metrics {
  view_iou: number[];
  mean_iou: number;
  min_view_iou: number;
  view_recall: number[];   // foreground coverage per view
  recall: number;          // mean foreground coverage
  view_spill: number[];    // false-positive pixels / background pixels per view
  spill: number;
  view_precision: number[];
  loss: number;
  collisions: number;            // truly interpenetrating pairs (dense winding-number reference)
  clearance_violations: number;  // pairs closer than constraints.collision_margin (conservative)
  containment_violations: number;
  randomness: number;            // normalized entropy of piece-type counts (1 = all types equally used)
  diversity_deficit: number;     // KL(type counts || desired distribution) / log M
  type_counts: Record<string, number>;
  object_count: number;
  runtime_s: number;
  renderer_calls: number;
}

type MethodName = "chained";

interface MethodProgress {
  status: "pending" | "running" | "done" | "failed" | "cancelled" | "skipped";
  iteration: number;
  phase: string;             // free text, e.g. "growth", "pricing", "raster_polish"
  runtime_s: number;
  loss: number | null;
  view_iou: number[];
  object_count: number;
  best_metric: number | null;  // best min-view IoU so far
  error: string | null;
  preview_urls: string[];      // latest rendered silhouette per view (PNG, cache-bust with ?t=)
  assembly: Assembly | null;   // latest assembly snapshot (may lag)
  metrics: Metrics | null;     // final evaluation, when done
  result_dir_url: string | null; // e.g. /outputs/run_003/beam
  info: Record<string, unknown>; // method-specific extras (LP objective, columns, ...)
}

interface ComparisonRow {
  method: MethodName;
  mean_iou: number; min_view_iou: number; view_iou: number[];
  spill: number; coverage: number; collisions: number;
  containment_violations: number; object_count: number;
  runtime_s: number; loss: number; score: number;
  randomness: number; clearance_violations: number; type_counts: Record<string, number>;
}

interface Comparison {
  selection_metric: string;
  best_method: MethodName | null;
  rows: ComparisonRow[];
}

interface Preprocessing {
  target_urls: string[];          // binary target masks (PNG, square)
  cameras: Camera[];
  bounding_volume: { min: Vec3; max: Vec3 };
  hull_url: string | null;        // OBJ preview mesh of the visual hull
  hull_voxels: number;
  strict_coverage_upper_bound: number[]; // per view, in [0,1]
  warnings: string[];
}

interface JobStatus {
  job_id: string;
  status: "queued" | "preprocessing" | "running" | "done" | "failed" | "cancelled";
  error: string | null;
  created_at: number;          // unix seconds
  output_dir_url: string | null;  // e.g. /outputs/run_003
  request: JobRequest;
  preprocessing: Preprocessing | null;
  methods: Partial<Record<MethodName, MethodProgress>>;
  comparison: Comparison | null;
}

interface JobRequest {
  models_dir: string;            // default "assets/models"
  model_names?: string[];        // selected model stems; omitted means every model in models_dir
  targets: string[];             // 1 or 2 server-side image paths (from upload or list)
  cameras: Camera[];             // same length as targets
  methods: MethodName[];
  preset: "fast" | "default" | "high_quality";
  overrides?: Record<string, unknown>;  // nested config overrides ("Custom" JSON merged with the design options:
                                        //   scale.mode/fixed, bounding_volume.unbounded_view_axis, diversity.weight)
}
```

Per-source-model scale multipliers may be supplied in `overrides.scale.model_factors`, keyed by
the model stem (or filename), for example `{"scale":{"model_factors":{"bunny":1.5}}}`. The
multiplier scales that model's optimized min/max range and its fixed/native size.
Fixed model colors may be supplied as `overrides.targets.color.fixed_models`, keyed by model stem
or filename, with CSS-style hex values such as `{"Chair":"#b87333"}`. Models absent from this map
receive independently matched colors for each placed instance.

## Endpoints

| Method | Path | Body / Query | Returns |
|---|---|---|---|
| GET  | `/api/health` | | `{ok: true, cuda: bool, gurobi: bool}` |
| GET  | `/api/defaults` | | `{models_dir, cameras: Camera[2], bounding_volume, presets: string[], methods: MethodName[], targets: string[] /* default target paths */}` |
| GET  | `/api/models?dir=assets/models` | | `ModelInfo[]` (triggers preprocessing/caching; can take several seconds the first time) |
| GET  | `/api/models/{name}/mesh.glb?dir=...` | | GLB binary |
| GET  | `/api/models/{name}/thumbnail.png?dir=...` | | PNG |
| GET  | `/api/targets` | | `{path: string, url: string, name: string}[]` — images in `assets/targets` and uploads |
| POST | `/api/targets/upload` | multipart `file` | `{path, url, name}` |
| GET  | `/api/files?path=...` | | serves an image under assets/ or uploads/ |
| POST | `/api/jobs` | `JobRequest` | `{job_id}` (returns immediately) |
| GET  | `/api/jobs` | | `JobStatus[]` (most recent first) |
| GET  | `/api/jobs/{job_id}` | | `JobStatus` |
| POST | `/api/jobs/{job_id}/cancel` | | `{ok: true}` |
| WS   | `/api/jobs/{job_id}/ws` | | stream of events (below) |
| GET  | `/outputs/...` | | static files of all runs |

Finished run folders found in `outputs/` (e.g. written by the CLI) are listed by `GET /api/jobs` as
`done` jobs whose `job_id` is the folder name. The UI opens one directly with `/?job=<job_id>`.

### WebSocket events

Each message is a JSON object:

```ts
{ type: "status", data: JobStatus }                 // sent on connect and on every status change
{ type: "progress", method: MethodName, data: MethodProgress }  // throttled (~2/s per method)
{ type: "comparison", data: Comparison }
{ type: "done", data: JobStatus }
{ type: "error", data: { message: string } }
```

The client can always fall back to polling `GET /api/jobs/{job_id}`.

### Result files (under `output_dir_url`)

```
preprocessing/target_0.png, target_1.png, hull_preview.obj, hull.npz, diagnostics.json
<method>/result.json      // {method, assembly: Assembly, metrics: Metrics, info, config}
<method>/assembly.glb
<method>/renders/view_0.png, view_1.png, overlay_0.png, color_view_0.png ...
<method>/metrics.json, optimization.csv
comparison.json, comparison.csv
```

## Frontend-facing additions (chained method, design options, lock-and-rerun)

* `MethodName` also includes `"chained"` (display name "Chained (SDF→CG→Beam)"); it appears in
  `/api/defaults.methods`, `JobStatus.methods` and comparison rows like the other methods.
  The UI lists it always and disables it when `/api/defaults.methods` does not offer it.
* Design-option overrides sent by the UI (only keys that differ from the defaults):
  ```ts
  scale?: { mode: "free" | "fixed" | "native"; fixed?: number; native_factor?: number /* default 0.3 */ };
  viewing_zone?: { radius: number /* 0 = off, UI range 0..0.6 */; samples?: number /* default 4, 2..8 */ };
  reveal?: { weight: number /* UI range 0.01..0.2, sent only when enabled */ };
  ```
  The viewer draws a ring of radius `viewing_zone.radius` around each camera, perpendicular to its view direction.
* `ObjectInstance.locked?: boolean` (default false): the piece was kept fixed by a lock-and-rerun job.
* `JobRequest.initial?: { assembly: Assembly; locked: number[]; keep_unlocked: boolean }`:
  `assembly.objects` is the full edited assembly (same object schema as results; edited pieces have a new
  `translation`, row-major `rotation_matrix` and `rotation6d` = `[R00, R10, R20, R01, R11, R21]`, scale unchanged).
  `locked` = indices into `assembly.objects` that must stay exactly fixed. `keep_unlocked`: true = the unlocked
  pieces are the warm start (they may move / be replaced); false = they are discarded. The UI's "Rerun with locked
  pieces" sends the viewed job's `models_dir`, `targets`, `cameras`, `methods` and `preset`, plus either the current
  design-option / bounds overrides or the job's own `overrides`. Edits are client-side only (never persisted).
