// Top-level layout and global state.
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, errorMessage, modelMeshUrl } from "../api/client";
import { isTerminal, useJobStream } from "../api/ws";
import type {
  Camera, Defaults, Health, JobRequest, JobStatus, MethodName, ModelInfo, TargetInfo, Vec3,
} from "../api/types";
import { ALL_METHODS, METHOD_SHORT } from "../api/types";
import { CameraEditor } from "../components/CameraEditor";
import { ComparisonTable } from "../components/ComparisonTable";
import { JobControls } from "../components/JobControls";
import {
  DEFAULT_DESIGN_OPTIONS, deepMerge, designOverrides, MethodPanel, parseOverrides, type DesignOptions, type PresetChoice,
} from "../components/MethodPanel";
import { ModelPanel } from "../components/ModelPanel";
import { jobMethods, ProgressPanel } from "../components/ProgressPanel";
import { TargetPanel, targetUrlFor } from "../components/TargetPanel";
import { ViewerToolbar, type DisplayMethod, type SceneMode } from "../components/ViewerToolbar";
import {
  DEFAULT_VIEWER_SETTINGS, SculptureViewer, type SceneData, type ViewerSettings,
} from "../viewer/SculptureViewer";
import { useMethodAssembly } from "./useMethodAssembly";

// Used until / unless GET /api/defaults answers (mirrors configs/default.yaml).
const FALLBACK_CAMERAS: Camera[] = [
  { position: [0, 0, 5], look_at: [0, 0, 0], up: [0, 1, 0], fov_y_deg: 25, near: 0.1, far: 20 },
  { position: [5, 0, 0], look_at: [0, 0, 0], up: [0, 1, 0], fov_y_deg: 25, near: 0.1, far: 20 },
];
const FALLBACK_DEFAULTS: Defaults = {
  models_dir: "assets/models",
  cameras: FALLBACK_CAMERAS,
  bounding_volume: { min: [-1, -1, -1], max: [1, 1, 1] },
  presets: ["fast", "default", "high_quality"],
  methods: ALL_METHODS,
  targets: [],
};

/** Always two editable cameras (the second one is used when Target 2 is enabled). */
function twoCameras(cams: Camera[] | undefined): Camera[] {
  return [0, 1].map((i) => cams?.[i] ?? FALLBACK_CAMERAS[i]);
}

const round4 = (x: number) => Math.round(x * 1e4) / 1e4;

export function App() {
  // ---- backend defaults / health -------------------------------------------------
  const [defaults, setDefaults] = useState<Defaults>(FALLBACK_DEFAULTS);
  const [health, setHealth] = useState<Health | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);

  // ---- inputs ------------------------------------------------------------------------
  const [modelsDir, setModelsDir] = useState(FALLBACK_DEFAULTS.models_dir);
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [modelsLoadedDir, setModelsLoadedDir] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [targets, setTargets] = useState<TargetInfo[]>([]);
  const [target1, setTarget1] = useState<string | null>(null);
  const [target2Enabled, setTarget2Enabled] = useState(false);
  const [target2, setTarget2] = useState<string | null>(null);
  const [cameras, setCameras] = useState<Camera[]>(FALLBACK_CAMERAS);

  const [methods, setMethods] = useState<MethodName[]>(ALL_METHODS);
  const [preset, setPreset] = useState<PresetChoice>("default");
  const [overridesText, setOverridesText] = useState("{\n}");
  const [designOptions, setDesignOptions] = useState<DesignOptions>(DEFAULT_DESIGN_OPTIONS);

  // ---- jobs --------------------------------------------------------------------------
  const [jobId, setJobId] = useState<string | null>(null);
  const stream = useJobStream(jobId);
  const job = stream.job;
  // Active until a terminal status arrives (or the job cannot be fetched at all).
  const jobActive = job ? !isTerminal(job.status) : !!jobId && !stream.error;
  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  // ---- viewer --------------------------------------------------------------------------
  const [sceneMode, setSceneMode] = useState<SceneMode>("setup");
  const [displayMethod, setDisplayMethod] = useState<DisplayMethod>("best");
  const [viewIndex, setViewIndex] = useState<number | null>(null);
  const [selectedCamera, setSelectedCamera] = useState<number | null>(null);
  const [settings, setSettings] = useState<ViewerSettings>(DEFAULT_VIEWER_SETTINGS);

  // ---- initial load ------------------------------------------------------------------
  const refreshTargets = useCallback(() => {
    api.targets().then((t) => setTargets(Array.isArray(t) ? t : [])).catch(() => undefined);
  }, []);
  const refreshJobs = useCallback(() => {
    api.listJobs().then((j) => setJobs(Array.isArray(j) ? j : [])).catch(() => undefined);
  }, []);

  useEffect(() => {
    api
      .defaults()
      .then((d) => {
        const merged: Defaults = {
          ...FALLBACK_DEFAULTS,
          ...d,
          cameras: twoCameras(d.cameras),
          bounding_volume: d.bounding_volume ?? FALLBACK_DEFAULTS.bounding_volume,
          methods: d.methods?.length ? d.methods : ALL_METHODS,
          targets: d.targets ?? [],
        };
        setDefaults(merged);
        setModelsDir(merged.models_dir || FALLBACK_DEFAULTS.models_dir);
        setCameras(merged.cameras);
        setMethods(merged.methods);
        setTarget1((t) => t ?? merged.targets[0] ?? null);
        setTarget2((t) => t ?? merged.targets[1] ?? null);
        setBackendError(null);
      })
      .catch((e) => setBackendError(`Backend unavailable (${errorMessage(e)}). Using built-in defaults.`));
    api.health().then(setHealth).catch(() => setHealth(null));
    refreshTargets();
    refreshJobs();
  }, [refreshTargets, refreshJobs]);

  // Refresh the job list whenever the current job reaches a terminal state.
  useEffect(() => {
    if (job && isTerminal(job.status)) refreshJobs();
  }, [job?.status, refreshJobs]);

  // ---- models --------------------------------------------------------------------------
  const loadModels = useCallback(() => {
    const dir = modelsDir.trim();
    if (!dir) return;
    setModelsLoading(true);
    setModelsError(null);
    api
      .models(dir)
      .then((m) => {
        setModels(Array.isArray(m) ? m : []);
        setModelsLoadedDir(dir);
      })
      .catch((e) => setModelsError(errorMessage(e)))
      .finally(() => setModelsLoading(false));
  }, [modelsDir]);

  const modelsByName = useMemo(() => new Map((models ?? []).map((m) => [m.name, m])), [models]);
  const meshDir = sceneMode === "results" && job?.request?.models_dir ? job.request.models_dir : modelsDir.trim();
  const meshUrlFor = useCallback(
    (name: string) => {
      const info = modelsLoadedDir === meshDir ? modelsByName.get(name) : undefined;
      return info?.mesh_url || modelMeshUrl(name, meshDir);
    },
    [modelsByName, modelsLoadedDir, meshDir],
  );

  // ---- run -----------------------------------------------------------------------------
  const numSetupViews = target2Enabled ? 2 : 1;
  const overrides = preset === "custom" ? parseOverrides(overridesText) : null;
  const startBlockers: string[] = [];
  if (!modelsDir.trim()) startBlockers.push("Model folder is empty.");
  if (!target1) startBlockers.push("Select Target Image 1.");
  if (target2Enabled && !target2) startBlockers.push("Select Target Image 2 (or disable it).");
  if (methods.length === 0) startBlockers.push("Select at least one method.");
  if (overrides?.error) startBlockers.push("Fix the Custom overrides JSON.");

  const openJob = useCallback((id: string) => {
    setJobId(id);
    setSceneMode("results");
    setDisplayMethod("best");
    setSelectedCamera(null);
    setRunError(null);
  }, []);

  // Deep link: /?job=<id> opens that job's results directly (e.g. a run loaded from outputs/).
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("job");
    if (id) openJob(id);
  }, [openJob]);

  const start = async () => {
    if (startBlockers.length || !target1) return;
    const targetsReq = target2Enabled && target2 ? [target1, target2] : [target1];
    const req: JobRequest = {
      models_dir: modelsDir.trim(),
      targets: targetsReq,
      cameras: cameras.slice(0, targetsReq.length),
      methods: ALL_METHODS.filter((m) => methods.includes(m)),
      preset: preset === "custom" ? "default" : preset,
    };
    const merged = deepMerge(
      designOverrides(designOptions, targetsReq.length === 1),
      preset === "custom" && overrides?.value ? overrides.value : {},
    );
    if (Object.keys(merged).length) req.overrides = merged;
    setSubmitting(true);
    setRunError(null);
    try {
      const { job_id } = await api.createJob(req);
      openJob(job_id);
      setViewIndex(null);
      refreshJobs();
    } catch (e) {
      setRunError(errorMessage(e));
    } finally {
      setSubmitting(false);
    }
  };

  const cancel = async () => {
    if (!jobId) return;
    setCancelling(true);
    try {
      await api.cancelJob(jobId);
    } catch (e) {
      setRunError(errorMessage(e));
    } finally {
      setCancelling(false);
    }
  };

  // ---- results scene -----------------------------------------------------------------
  const methodAvailable = useCallback(
    (m: MethodName) => {
      const p = job?.methods?.[m];
      return !!p && p.status !== "skipped" && (!!p.assembly || !!p.result_dir_url);
    },
    [job],
  );
  const liveFallback = job ? jobMethods(job).find(methodAvailable) ?? null : null;
  const bestMethod = job?.comparison?.best_method ?? null;
  const resolvedMethod: MethodName | null = displayMethod === "best" ? bestMethod ?? liveFallback : displayMethod;
  const assembly = useMethodAssembly(job, sceneMode === "results" ? resolvedMethod : null);

  const jobTargetUrls = useMemo(() => {
    if (!job) return [];
    const reqTargets = job.request?.targets ?? [];
    const pre = job.preprocessing?.target_urls ?? [];
    const n = Math.max(reqTargets.length, pre.length);
    return Array.from({ length: n }, (_, i) => pre[i] ?? targetUrlFor(reqTargets[i] ?? null, targets));
  }, [job, targets]);

  const setupTargetUrls = useMemo(
    () => [targetUrlFor(target1, targets), target2Enabled ? targetUrlFor(target2, targets) : null],
    [target1, target2, target2Enabled, targets],
  );

  const scene: SceneData = useMemo(() => {
    if (sceneMode === "results" && job) {
      const pre = job.preprocessing;
      return {
        cameras: pre?.cameras?.length ? pre.cameras : job.request?.cameras ?? [],
        targetImageUrls: jobTargetUrls,
        boundingVolume: pre?.bounding_volume ?? defaults.bounding_volume,
        hullUrl: pre?.hull_url ?? null,
        assembly,
        meshUrlFor,
        editable: false,
      };
    }
    return {
      cameras: cameras.slice(0, numSetupViews),
      targetImageUrls: setupTargetUrls,
      boundingVolume: defaults.bounding_volume,
      hullUrl: null,
      assembly: null,
      meshUrlFor,
      editable: true,
    };
  }, [sceneMode, job, jobTargetUrls, defaults, assembly, meshUrlFor, cameras, numSetupViews, setupTargetUrls]);

  const numViews = scene.cameras.length;

  // Keep view / selection indices valid when the number of cameras changes.
  useEffect(() => {
    if (viewIndex !== null && viewIndex >= numViews) setViewIndex(null);
    if (selectedCamera !== null && (selectedCamera >= numViews || !scene.editable)) setSelectedCamera(null);
  }, [numViews, viewIndex, selectedCamera, scene.editable]);

  const updateCamera = (i: number, cam: Camera) => setCameras((cs) => cs.map((c, k) => (k === i ? cam : c)));
  const moveCamera = (i: number, p: Vec3) =>
    setCameras((cs) => cs.map((c, k) => (k === i ? { ...c, position: p.map(round4) as Vec3 } : c)));

  const pickMethod = (m: DisplayMethod) => {
    setDisplayMethod(m);
    if (job) setSceneMode("results");
  };

  // ---- HUD text ------------------------------------------------------------------------
  let hud: string;
  if (scene.editable) {
    hud = selectedCamera !== null
      ? `Setup · dragging Camera ${selectedCamera + 1} (gizmo)`
      : "Setup · click a camera body to drag it";
  } else if (!resolvedMethod) {
    hud = "Results · no assembly available yet";
  } else {
    const p = job?.methods?.[resolvedMethod];
    const live = p && p.status !== "done" ? " (live)" : "";
    const label = displayMethod === "best" && bestMethod ? `Best = ${METHOD_SHORT[resolvedMethod]}` : METHOD_SHORT[resolvedMethod];
    hud = `Results · ${label}${live} · ${assembly ? `${assembly.objects?.length ?? 0} objects` : "no assembly yet"}`;
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>Floating Anamorphic Sculptures</h1>
        <span className="spacer" />
        {health ? (
          <>
            <span className={`pill ${health.cuda ? "ok" : "warn"}`}>CUDA {health.cuda ? "✓" : "✗"}</span>
            <span className={`pill ${health.gurobi ? "ok" : "warn"}`}>Gurobi {health.gurobi ? "✓" : "✗"}</span>
          </>
        ) : (
          <span className="pill warn">backend offline</span>
        )}
      </header>
      {backendError && <div className="banner">{backendError}</div>}

      <div className="layout">
        <aside className="sidebar left">
          <ModelPanel
            dir={modelsDir}
            onDirChange={setModelsDir}
            onLoad={loadModels}
            models={models}
            loading={modelsLoading}
            error={modelsError}
          />
          <TargetPanel
            targets={targets}
            target1={target1}
            onTarget1={setTarget1}
            target2Enabled={target2Enabled}
            onTarget2Enabled={setTarget2Enabled}
            target2={target2}
            onTarget2={setTarget2}
            onUploaded={(t) => setTargets((ts) => (ts.some((x) => x.path === t.path) ? ts : [...ts, t]))}
          />
          <CameraEditor
            cameras={cameras}
            count={numSetupViews}
            defaults={defaults.cameras}
            onChange={updateCamera}
            selected={sceneMode === "setup" ? selectedCamera : null}
            onSelect={(i) => {
              setSceneMode("setup");
              setSelectedCamera(i);
            }}
          />
          <MethodPanel
            available={defaults.methods}
            selected={methods}
            onToggle={(m, on) => setMethods((ms) => (on ? [...new Set([...ms, m])] : ms.filter((x) => x !== m)))}
            preset={preset}
            onPreset={setPreset}
            overridesText={overridesText}
            onOverridesText={setOverridesText}
            options={designOptions}
            onOptions={setDesignOptions}
            singleView={!target2Enabled}
          />
          <JobControls
            canStart={startBlockers.length === 0}
            startBlockers={startBlockers}
            submitting={submitting}
            jobActive={jobActive}
            onStart={() => void start()}
            onCancel={() => void cancel()}
            cancelling={cancelling}
            error={runError}
            jobs={jobs}
            currentJobId={jobId}
            onOpenJob={openJob}
            onRefreshJobs={refreshJobs}
          />
        </aside>

        <main className="center">
          <ViewerToolbar
            sceneMode={sceneMode}
            onSceneMode={setSceneMode}
            hasJob={!!job}
            displayMethod={displayMethod}
            onDisplayMethod={pickMethod}
            methodAvailable={methodAvailable}
            bestAvailable={!!bestMethod || !!liveFallback}
            numViews={numViews}
            viewIndex={viewIndex}
            onViewIndex={setViewIndex}
            settings={settings}
            onSettings={(patch) => setSettings((s) => ({ ...s, ...patch }))}
            hasHull={!scene.editable && !!scene.hullUrl}
          />
          <SculptureViewer
            scene={scene}
            settings={settings}
            viewIndex={viewIndex}
            selectedCamera={selectedCamera}
            onSelectCamera={setSelectedCamera}
            onMoveCamera={moveCamera}
            hud={hud}
          />
          {job?.comparison && (
            <ComparisonTable
              job={job}
              comparison={job.comparison}
              selectedMethod={sceneMode === "results" ? resolvedMethod : null}
              onSelect={(m) => pickMethod(m)}
            />
          )}
        </main>

        <aside className="sidebar right">
          <ProgressPanel
            job={job}
            connection={stream.connection}
            lastUpdate={stream.lastUpdate}
            streamError={stream.error}
            targetUrls={jobTargetUrls}
            displayedMethod={sceneMode === "results" ? resolvedMethod : null}
            onShowMethod={(m) => pickMethod(m)}
          />
        </aside>
      </div>
    </div>
  );
}
