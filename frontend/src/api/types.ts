// Types copied from docs/api.md (the authoritative API contract).
// Additional helper types used only by the frontend are at the bottom.

export type Vec3 = [number, number, number];

export interface Camera {
  position: Vec3; look_at: Vec3; up: Vec3;
  fov_y_deg: number; near: number; far: number;
}

export interface ModelInfo {
  id: number;               // index in the (sorted) model folder
  name: string;             // stem, unique key used in URLs
  filename: string;         // e.g. "bunny.obj"
  triangles: number;        // original triangle count
  proxy_triangles: number;  // rendering proxy triangle count
  dims: Vec3;               // normalized bounding-box extents (radius-1 sphere)
  original_dims: Vec3;      // bounding-box extents in source units
  original_radius: number;  // source-space bounding-sphere radius
  thumbnail_url: string;    // PNG
  mesh_url: string;         // GLB of the canonical normalized mesh
}

export interface ObjectInstance {
  mesh_id: number;
  mesh_name: string;
  source_file: string;
  translation: Vec3;
  rotation_matrix: [Vec3, Vec3, Vec3];  // row-major
  rotation6d: number[];                 // 6 numbers (first two columns of R)
  log_scale: number;
  scale: number;
  color?: Vec3 | null;       // linear/sRGB display color in [0,1]
  /** Must stay exactly fixed (lock-and-rerun). Absent = false. */
  locked?: boolean;
}

export interface Assembly {
  objects: ObjectInstance[];
}

export interface Metrics {
  view_iou: number[];
  mean_iou: number;
  min_view_iou: number;
  view_recall: number[];   // foreground coverage per view
  recall: number;          // mean foreground coverage
  view_spill: number[];    // false-positive pixels / background pixels per view
  spill: number;
  view_precision: number[];
  loss: number;
  collisions: number;
  containment_violations: number;
  object_count: number;
  runtime_s: number;
  renderer_calls: number;
}

export type MethodName = "chained";

export interface MethodProgress {
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

export interface ComparisonRow {
  method: MethodName;
  mean_iou: number; min_view_iou: number; view_iou: number[];
  spill: number; coverage: number; collisions: number;
  containment_violations: number; object_count: number;
  runtime_s: number; loss: number; score: number;
  randomness?: number;            // normalized entropy of piece-type counts (1 = all types equally used)
  clearance_violations?: number;  // pairs closer than the clearance margin
  type_counts?: Record<string, number>;
}

export interface Comparison {
  selection_metric: string;
  best_method: MethodName | null;
  rows: ComparisonRow[];
}

export interface BoundingVolume { min: Vec3; max: Vec3 }

export interface Preprocessing {
  target_urls: string[];          // aspect-preserving target previews (PNG, square canvas)
  cameras: Camera[];
  bounding_volume: BoundingVolume;
  hull_url: string | null;        // OBJ preview mesh of the visual hull
  hull_voxels: number;
  strict_coverage_upper_bound: number[]; // per view, in [0,1]
  warnings: string[];
}

export type JobState = "queued" | "preprocessing" | "running" | "done" | "failed" | "cancelled";

export interface JobStatus {
  job_id: string;
  status: JobState;
  error: string | null;
  created_at: number;          // unix seconds
  output_dir_url: string | null;  // e.g. /outputs/run_003
  request: JobRequest;
  preprocessing: Preprocessing | null;
  methods: Partial<Record<MethodName, MethodProgress>>;
  comparison: Comparison | null;
}

export type Preset = "fast" | "default" | "high_quality";

export interface JobRequest {
  models_dir: string;            // default "assets/models"
  model_names?: string[];        // selected source models; omitted means all
  targets: string[];             // 1 or 2 server-side image paths (from upload or list)
  cameras: Camera[];             // same length as targets
  methods: MethodName[];
  preset: Preset;
  overrides?: Record<string, unknown>;  // nested config overrides ("Custom")
  /** Lock-and-rerun: optimize around the locked pieces of an (edited) assembly. */
  initial?: InitialAssembly;
}

export interface InitialAssembly {
  /** The full edited assembly (same object schema as results). */
  assembly: Assembly;
  /** Indices into `assembly.objects` that must stay exactly fixed. */
  locked: number[];
  /** true: unlocked pieces are the warm start (may move / be replaced); false: they are discarded. */
  keep_unlocked: boolean;
}

// ---------------------------------------------------------------------------
// Endpoint payloads (shapes from the endpoint table in docs/api.md)

export interface Health { ok: boolean; cuda: boolean; gurobi: boolean }

/** `bounding_volume` section of the config as returned by GET /api/defaults. */
export interface DefaultsBoundingVolume extends BoundingVolume {
  unbounded_view_axis?: boolean;  // single view only: B = box around camera 0's frustum segment
  view_axis_near?: number;        // distance from camera 0 (used when unbounded_view_axis)
  view_axis_far?: number;
}

export interface Defaults {
  models_dir: string;
  cameras: Camera[];
  bounding_volume: DefaultsBoundingVolume;
  presets: string[];
  methods: MethodName[];
  targets: string[];           // default target paths
}

export interface TargetInfo { path: string; url: string; name: string }

export type WsEvent =
  | { type: "status"; data: JobStatus }
  | { type: "progress"; method: MethodName; data: MethodProgress }
  | { type: "comparison"; data: Comparison }
  | { type: "done"; data: JobStatus }
  | { type: "error"; data: { message: string } };

/** Contents of `<result_dir_url>/result.json`. */
export interface MethodResultFile {
  method: MethodName;
  assembly: Assembly;
  metrics: Metrics;
  info: Record<string, unknown>;
  config: Record<string, unknown>;
}

export const ALL_METHODS: MethodName[] = ["chained"];

export const METHOD_LABELS: Record<MethodName, string> = {
  chained: "Chained (SDF→CG→Beam)",
};

export const METHOD_SHORT: Record<MethodName, string> = {
  chained: "Chained",
};
