// Top-level layout and global state.
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, errorMessage, modelMeshUrl } from "../api/client";
import { isTerminal, useJobStream } from "../api/ws";
import type {
  BoundingVolume, Camera, Defaults, Health, JobRequest, JobStatus, MethodName, ModelInfo, ObjectInstance, TargetInfo, Vec3,
} from "../api/types";
import { ALL_METHODS, METHOD_LABELS, METHOD_SHORT } from "../api/types";
import {
  BoundsPanel, boundsErrors, boundsFromDefaults, boundsOverrides, FALLBACK_BOUNDS, type BoundsSettings,
} from "../components/BoundsPanel";
import { CameraEditor } from "../components/CameraEditor";
import { EditPanel, type GizmoChoice } from "../components/EditPanel";
import { ComparisonTable } from "../components/ComparisonTable";
import { JobControls } from "../components/JobControls";
import {
  DEFAULT_DESIGN_OPTIONS, deepMerge, designOverrides, MethodPanel, parseOverrides, viewingZoneRadiusOf,
  type DesignOptions, type PresetChoice,
} from "../components/MethodPanel";
import { ModelPanel } from "../components/ModelPanel";
import { jobMethods, ProgressPanel } from "../components/ProgressPanel";
import { TargetPanel, targetUrlFor } from "../components/TargetPanel";
import { ViewerToolbar, type DisplayMethod, type SceneMode } from "../components/ViewerToolbar";
import {
  DEFAULT_VIEWER_SETTINGS, SculptureViewer, type SceneData, type ViewerEditing, type ViewerSettings,
} from "../viewer/SculptureViewer";
import { useAssemblyEdits } from "./useAssemblyEdits";
import { useMethodAssembly } from "./useMethodAssembly";

// Used until / unless GET /api/defaults answers (mirrors configs/default.yaml).
const FALLBACK_CAMERAS: Camera[] = [
  { position: [0, 0, 5], look_at: [0, 0, 0], up: [0, 1, 0], fov_y_deg: 25, near: 0.1, far: 20 },
  { position: [5, 0, 0], look_at: [0, 0, 0], up: [0, 1, 0], fov_y_deg: 25, near: 0.1, far: 20 },
];
const FALLBACK_DEFAULTS: Defaults = {
  models_dir: "assets/models",
  cameras: FALLBACK_CAMERAS,
  bounding_volume: {
    min: FALLBACK_BOUNDS.min,
    max: FALLBACK_BOUNDS.max,
    unbounded_view_axis: false,
    view_axis_near: FALLBACK_BOUNDS.viewAxisNear,
    view_axis_far: FALLBACK_BOUNDS.viewAxisFar,
  },
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
  const [bounds, setBounds] = useState<BoundsSettings>(FALLBACK_BOUNDS);

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

  // ---- assembly editing (lock-and-rerun) ---------------------------------------------
  const [selection, setSelection] = useState<number[]>([]);
  const [gizmo, setGizmo] = useState<GizmoChoice>("off");
  const [keepUnlocked, setKeepUnlocked] = useState(true);
  const [applyCurrent, setApplyCurrent] = useState(true);
  const [rerunning, setRerunning] = useState(false);
  const [rerunError, setRerunError] = useState<string | null>(null);

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
        setBounds(boundsFromDefaults(merged.bounding_volume));
        if (merged.bounding_volume.unbounded_view_axis) setDesignOptions((o) => ({ ...o, unboundedAxis: true }));
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
  // Unbounded camera-axis bounds (single view only): B follows camera 0's frustum.
  const viewAxisMode = designOptions.unboundedAxis && numSetupViews === 1;
  const overrides = preset === "custom" ? parseOverrides(overridesText) : null;
  const startBlockers: string[] = [];
  if (!modelsDir.trim()) startBlockers.push("Model folder is empty.");
  if (!target1) startBlockers.push("Select Target Image 1.");
  if (target2Enabled && !target2) startBlockers.push("Select Target Image 2 (or disable it).");
  if (methods.length === 0) startBlockers.push("Select at least one method.");
  if (overrides?.error) startBlockers.push("Fix the Custom overrides JSON.");
  startBlockers.push(...boundsErrors(bounds, viewAxisMode));

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

  /** Config overrides from the current form. Later entries win: design options < bounds < Custom JSON. */
  const currentOverrides = (singleView: boolean): Record<string, unknown> =>
    deepMerge(
      deepMerge(designOverrides(designOptions, singleView), boundsOverrides(bounds, designOptions.unboundedAxis && singleView)),
      preset === "custom" && overrides?.value ? overrides.value : {},
    );

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
    const merged = currentOverrides(targetsReq.length === 1);
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
  const baseAssembly = useMethodAssembly(job, sceneMode === "results" ? resolvedMethod : null);

  // Local edits per (job, method); never sent anywhere except as `initial` of a rerun.
  const editKey = sceneMode === "results" && job && resolvedMethod ? `${job.job_id}/${resolvedMethod}` : null;
  const edits = useAssemblyEdits(editKey, baseAssembly);
  const updateEdits = edits.update;
  const assembly = edits.assembly;
  const editObjects = useMemo(() => assembly?.objects ?? [], [assembly]);
  const methodStatus = resolvedMethod ? job?.methods?.[resolvedMethod]?.status : undefined;
  const canEdit = !!editKey && !!assembly && (methodStatus === "done" || methodStatus === "cancelled" || methodStatus === "failed");
  const pickable = canEdit && viewIndex === null;

  // Selection / gizmo belong to one (job, method).
  useEffect(() => {
    setSelection([]);
    setGizmo("off");
    setRerunError(null);
  }, [editKey]);
  const validSelection = useMemo(() => selection.filter((i) => i < editObjects.length), [selection, editObjects.length]);
  const selectedSet = useMemo(() => new Set(validSelection), [validSelection]);

  const pick = useCallback((index: number, additive: boolean) => {
    setSelection((sel) => (additive ? (sel.includes(index) ? sel.filter((i) => i !== index) : [...sel, index]) : [index]));
  }, []);
  const clearSelection = useCallback(() => setSelection([]), []);
  const setLocked = (indices: ReadonlySet<number> | null, locked: boolean) =>
    updateEdits((objs) =>
      objs.map((o, i) => ((indices === null || indices.has(i)) && !!o.locked !== locked ? { ...o, locked } : o)),
    );
  const deleteSelected = () => {
    const del = new Set(validSelection);
    updateEdits((objs) => objs.filter((_, i) => !del.has(i)));
    setSelection([]);
    setGizmo("off");
  };
  const transformPiece = useCallback(
    (index: number, next: ObjectInstance) => updateEdits((objs) => objs.map((o, i) => (i === index && !o.locked ? next : o))),
    [updateEdits],
  );

  const gizmoIndex = validSelection.length === 1 ? validSelection[0] : null;
  const gizmoObject = gizmoIndex !== null ? editObjects[gizmoIndex] : undefined;
  const editing: ViewerEditing | null = canEdit
    ? {
        selected: selectedSet,
        pickable,
        onPick: pick,
        onClear: clearSelection,
        gizmo:
          gizmo !== "off" && gizmoIndex !== null && gizmoObject && !gizmoObject.locked
            ? { index: gizmoIndex, object: gizmoObject, mode: gizmo }
            : null,
        onTransform: transformPiece,
      }
    : null;

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

  // Stable object identity while only near / far change (keeps the box memo cheap).
  const setupVolume: BoundingVolume = useMemo(() => ({ min: bounds.min, max: bounds.max }), [bounds.min, bounds.max]);

  const scene: SceneData = useMemo(() => {
    if (sceneMode === "results" && job) {
      const pre = job.preprocessing;
      return {
        cameras: pre?.cameras?.length ? pre.cameras : job.request?.cameras ?? [],
        targetImageUrls: jobTargetUrls,
        boundingVolume: pre?.bounding_volume ?? defaults.bounding_volume,
        viewAxis: null,
        boundsEditable: false,
        hullUrl: pre?.hull_url ?? null,
        assembly,
        meshUrlFor,
        editable: false,
        zoneRadius: viewingZoneRadiusOf(job.request?.overrides),
      };
    }
    return {
      cameras: cameras.slice(0, numSetupViews),
      targetImageUrls: setupTargetUrls,
      boundingVolume: setupVolume,
      viewAxis: viewAxisMode ? { near: bounds.viewAxisNear, far: bounds.viewAxisFar } : null,
      boundsEditable: true,
      hullUrl: null,
      assembly: null,
      meshUrlFor,
      editable: true,
      zoneRadius: designOptions.viewingZoneRadius,
    };
  }, [
    sceneMode, job, jobTargetUrls, defaults, assembly, meshUrlFor, cameras, numSetupViews, setupTargetUrls,
    setupVolume, viewAxisMode, bounds.viewAxisNear, bounds.viewAxisFar, designOptions.viewingZoneRadius,
  ]);

  const numViews = scene.cameras.length;

  // Keep view / selection indices valid when the number of cameras changes.
  useEffect(() => {
    if (viewIndex !== null && viewIndex >= numViews) setViewIndex(null);
    if (selectedCamera !== null && (selectedCamera >= numViews || !scene.editable)) setSelectedCamera(null);
  }, [numViews, viewIndex, selectedCamera, scene.editable]);

  const updateCamera = (i: number, cam: Camera) => setCameras((cs) => cs.map((c, k) => (k === i ? cam : c)));
  const moveCamera = (i: number, p: Vec3) =>
    setCameras((cs) => cs.map((c, k) => (k === i ? { ...c, position: p.map(round4) as Vec3 } : c)));

  const changeBox = useCallback((v: BoundingVolume) => setBounds((b) => ({ ...b, min: v.min, max: v.max })), []);
  const changeViewAxis = useCallback(
    (viewAxisNear: number, viewAxisFar: number) => setBounds((b) => ({ ...b, viewAxisNear, viewAxisFar })),
    [],
  );

  const pickMethod = (m: DisplayMethod) => {
    setDisplayMethod(m);
    if (job) setSceneMode("results");
  };

  // ---- lock-and-rerun -------------------------------------------------------------------
  const jobReq = job?.request;
  const rerunSingleView = (jobReq?.targets?.length ?? 1) === 1;
  const lockedIndices = editObjects.flatMap((o, i) => (o.locked ? [i] : []));
  const rerunBlockers: string[] = [];
  if (!jobReq?.targets?.length || !jobReq.cameras?.length || !jobReq.methods?.length) {
    rerunBlockers.push("This job's request (targets / cameras / methods) is unavailable.");
  }
  if (editObjects.length === 0) rerunBlockers.push("The assembly is empty.");
  else if (lockedIndices.length === 0 && !keepUnlocked) rerunBlockers.push("Lock at least one piece (or keep the unlocked pieces).");
  if (applyCurrent) {
    if (overrides?.error) rerunBlockers.push("Fix the Custom overrides JSON.");
    rerunBlockers.push(...boundsErrors(bounds, designOptions.unboundedAxis && rerunSingleView));
  }

  const rerun = async () => {
    if (!jobReq || rerunBlockers.length) return;
    const n = jobReq.targets.length;
    const req: JobRequest = {
      models_dir: jobReq.models_dir,
      targets: [...jobReq.targets],
      cameras: jobReq.cameras.slice(0, n),
      methods: [...jobReq.methods],
      preset: jobReq.preset,
      initial: {
        assembly: { objects: editObjects.map((o) => ({ ...o, locked: !!o.locked })) },
        locked: lockedIndices,
        keep_unlocked: keepUnlocked,
      },
    };
    const ov = applyCurrent ? currentOverrides(rerunSingleView) : jobReq.overrides ?? {};
    if (Object.keys(ov).length) req.overrides = ov;
    setRerunning(true);
    setRerunError(null);
    try {
      const { job_id } = await api.createJob(req);
      openJob(job_id);
      setViewIndex(null);
      refreshJobs();
    } catch (e) {
      setRerunError(`Rerun failed: ${errorMessage(e)}`);
    } finally {
      setRerunning(false);
    }
  };

  // ---- HUD text ------------------------------------------------------------------------
  let hud: string;
  if (scene.editable) {
    hud = selectedCamera !== null
      ? `Setup · dragging Camera ${selectedCamera + 1} (gizmo)`
      : `Setup · click a camera body to drag it${
          settings.editBounds ? (viewAxisMode ? " · drag near / far handles on Camera 1's axis" : " · drag face handles to resize bounds") : ""
        }`;
  } else if (!resolvedMethod) {
    hud = "Results · no assembly available yet";
  } else {
    const p = job?.methods?.[resolvedMethod];
    const live = p && p.status !== "done" ? " (live)" : "";
    const label = displayMethod === "best" && bestMethod ? `Best = ${METHOD_SHORT[resolvedMethod]}` : METHOD_SHORT[resolvedMethod];
    hud = `Results · ${label}${live} · ${assembly ? `${assembly.objects?.length ?? 0} objects` : "no assembly yet"}${
      edits.edited ? " · edited" : ""
    }${validSelection.length ? ` · ${validSelection.length} selected` : ""}${
      editing?.gizmo ? ` · drag the gizmo to ${editing.gizmo.mode === "rotate" ? "rotate" : "move"}` : ""
    }`;
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
          <BoundsPanel
            bounds={bounds}
            onChange={setBounds}
            onReset={() => setBounds(boundsFromDefaults(defaults.bounding_volume))}
            singleView={numSetupViews === 1}
            unboundedAxis={designOptions.unboundedAxis}
            onUnboundedAxis={(unboundedAxis) => setDesignOptions((o) => ({ ...o, unboundedAxis }))}
            camera={cameras[0]}
            editInViewer={sceneMode === "setup" && settings.editBounds}
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
            onBoundsChange={changeBox}
            onViewAxisChange={changeViewAxis}
            hud={hud}
            editing={editing}
            sidePanel={
              sceneMode === "results" && resolvedMethod && assembly ? (
                <EditPanel
                  methodLabel={METHOD_LABELS[resolvedMethod] ?? resolvedMethod}
                  objects={editObjects}
                  selection={validSelection}
                  edited={edits.edited}
                  canEdit={canEdit}
                  pickHint={viewIndex !== null ? "Selection is disabled in Target View (switch to Free Orbit)." : null}
                  gizmo={gizmo}
                  onGizmo={setGizmo}
                  onLockSelected={(locked) => {
                    setLocked(selectedSet, locked);
                    if (locked) setGizmo("off");
                  }}
                  onDeleteSelected={deleteSelected}
                  onClearSelection={clearSelection}
                  onLockAll={(locked) => setLocked(null, locked)}
                  onReset={() => {
                    edits.reset();
                    setSelection([]);
                    setGizmo("off");
                  }}
                  keepUnlocked={keepUnlocked}
                  onKeepUnlocked={setKeepUnlocked}
                  applyCurrent={applyCurrent}
                  onApplyCurrent={setApplyCurrent}
                  rerunBlockers={rerunBlockers}
                  onRerun={() => void rerun()}
                  rerunning={rerunning}
                  rerunError={rerunError}
                />
              ) : null
            }
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
