// Controls above the 3D viewer: scene mode, method, view and display toggles.
import type { MethodName } from "../api/types";
import { ALL_METHODS, METHOD_SHORT } from "../api/types";
import type { ViewerSettings } from "../viewer/SculptureViewer";
import type { HullStyle } from "../viewer/HullMesh";
import type { OverlayBlend } from "../viewer/TargetOverlay";
import { Segmented } from "./common";

export type SceneMode = "setup" | "results";
export type DisplayMethod = MethodName | "best";

interface ViewerToolbarProps {
  sceneMode: SceneMode;
  onSceneMode: (m: SceneMode) => void;
  hasJob: boolean;
  displayMethod: DisplayMethod;
  onDisplayMethod: (m: DisplayMethod) => void;
  methodAvailable: (m: MethodName) => boolean;
  bestAvailable: boolean;
  numViews: number;
  viewIndex: number | null;
  onViewIndex: (i: number | null) => void;
  settings: ViewerSettings;
  onSettings: (patch: Partial<ViewerSettings>) => void;
  hasHull: boolean;
}

export function ViewerToolbar(p: ViewerToolbarProps) {
  const s = p.settings;
  const toggle = (key: keyof ViewerSettings, label: string, disabled = false, title?: string) => (
    <label className={`toggle ${disabled ? "disabled" : ""}`} title={title}>
      <input
        type="checkbox"
        checked={Boolean(s[key])}
        disabled={disabled}
        onChange={(e) => p.onSettings({ [key]: e.target.checked } as Partial<ViewerSettings>)}
      />
      {label}
    </label>
  );

  return (
    <div className="viewer-toolbar">
      <div className="toolbar-row">
        <Segmented<SceneMode>
          value={p.sceneMode}
          onChange={p.onSceneMode}
          options={[
            { value: "setup", label: "Setup" },
            { value: "results", label: "Results", disabled: !p.hasJob },
          ]}
        />
        {p.sceneMode === "results" && (
          <Segmented<DisplayMethod>
            value={p.displayMethod}
            onChange={p.onDisplayMethod}
            options={[
              ...ALL_METHODS.map((m) => ({
                value: m as DisplayMethod,
                label: METHOD_SHORT[m],
                disabled: !p.methodAvailable(m),
              })),
              {
                value: "best" as DisplayMethod,
                label: "Best",
                disabled: !p.bestAvailable,
                title: "comparison.best_method (live: first method with an assembly)",
              },
            ]}
          />
        )}
        <span className="spacer" />
        <Segmented<number>
          value={p.viewIndex ?? -1}
          onChange={(v) => p.onViewIndex(v < 0 ? null : v)}
          options={[
            { value: -1, label: "Free Orbit" },
            ...Array.from({ length: p.numViews }, (_, i) => ({ value: i, label: `Target View ${i + 1}` })),
          ]}
        />
      </div>
      <div className="toolbar-row toggles">
        {toggle("showBounds", "Bounding volume")}
        {toggle(
          "editBounds",
          "Edit bounds",
          p.sceneMode !== "setup",
          p.sceneMode === "setup" ? "Drag handles in the 3D view to resize / move the bounds" : "Setup mode only",
        )}
        {toggle("showHull", "Visual hull", !p.hasHull, p.hasHull ? undefined : "Available after preprocessing")}
        {s.showHull && p.hasHull && (
          <select
            className="compact"
            value={s.hullStyle}
            onChange={(e) => p.onSettings({ hullStyle: e.target.value as HullStyle })}
          >
            <option value="wireframe">wireframe</option>
            <option value="translucent">translucent</option>
          </select>
        )}
        {toggle("showFrusta", "Camera frusta")}
        {toggle("showBoxes", "Object boxes")}
        {toggle("showTargetPlanes", "Target images")}
        {toggle("silhouette", "Silhouette mode", false, "Flat black meshes on white for verifying the illusion")}
        {toggle("showOverlay", "Target overlay", p.viewIndex === null, "Visible in Target View")}
        {s.showOverlay && p.viewIndex !== null && (
          <>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={s.overlayOpacity}
              onChange={(e) => p.onSettings({ overlayOpacity: Number(e.target.value) })}
              title="Overlay opacity"
            />
            <select
              className="compact"
              value={s.overlayBlend}
              onChange={(e) => p.onSettings({ overlayBlend: e.target.value as OverlayBlend })}
              title="Overlay blend mode"
            >
              <option value="normal">normal</option>
              <option value="multiply">multiply</option>
              <option value="difference">difference</option>
            </select>
          </>
        )}
      </div>
    </div>
  );
}
