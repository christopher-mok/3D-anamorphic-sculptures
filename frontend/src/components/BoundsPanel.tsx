// Step 4: the sculpture bounding volume B (box, or camera-axis segment for a single view).
import type { Camera, DefaultsBoundingVolume, Vec3 } from "../api/types";
import { viewAxisVolume } from "../viewer/math";
import { ErrorText, NumberField, Section, Segmented, Vec3Field } from "./common";

/** Editable bounds; the single source of truth for both the inputs and the 3D gizmos. */
export interface BoundsSettings {
  min: Vec3;
  max: Vec3;
  /** Distances from camera 0 (bounding_volume.view_axis_near / view_axis_far). */
  viewAxisNear: number;
  viewAxisFar: number;
}

/** Gizmo limits (the numeric inputs only require min < max and 0 < near < far). */
export const MIN_BOX_SIZE = 0.05;
export const MIN_VIEW_AXIS_NEAR = 0.1;
export const MIN_VIEW_AXIS_GAP = 0.1;

export const FALLBACK_BOUNDS: BoundsSettings = {
  min: [-1, -1, -1],
  max: [1, 1, 1],
  viewAxisNear: 1.5,
  viewAxisFar: 12,
};

export function boundsFromDefaults(bv: DefaultsBoundingVolume | undefined): BoundsSettings {
  const vec = (v: unknown, fb: Vec3): Vec3 =>
    Array.isArray(v) && v.length === 3 && v.every(Number.isFinite) ? ([...v] as Vec3) : fb;
  const num = (v: unknown, fb: number) => (typeof v === "number" && Number.isFinite(v) ? v : fb);
  return {
    min: vec(bv?.min, FALLBACK_BOUNDS.min),
    max: vec(bv?.max, FALLBACK_BOUNDS.max),
    viewAxisNear: num(bv?.view_axis_near, FALLBACK_BOUNDS.viewAxisNear),
    viewAxisFar: num(bv?.view_axis_far, FALLBACK_BOUNDS.viewAxisFar),
  };
}

/** Validation messages (empty = valid). `viewAxis` = unbounded single-view mode is active. */
export function boundsErrors(b: BoundsSettings, viewAxis: boolean): string[] {
  const errs: string[] = [];
  const bad = (["x", "y", "z"] as const).filter((_, i) => !(b.min[i] < b.max[i]));
  if (bad.length) errs.push(`Bounds: min must be < max (${bad.join(", ")}).`);
  if (viewAxis && !(b.viewAxisNear > 0 && b.viewAxisNear < b.viewAxisFar)) {
    errs.push("Bounds: camera-axis distances need 0 < near < far.");
  }
  return errs;
}

/** Config overrides for the bounds (merged under the Custom JSON, which wins). */
export function boundsOverrides(b: BoundsSettings, viewAxis: boolean): Record<string, unknown> {
  const bv: Record<string, unknown> = { min: [...b.min], max: [...b.max] };
  if (viewAxis) {
    bv.view_axis_near = b.viewAxisNear;
    bv.view_axis_far = b.viewAxisFar;
  }
  return { bounding_volume: bv };
}

const fmtVec = (v: Vec3) => `(${v.map((x) => x.toFixed(2)).join(", ")})`;

interface BoundsPanelProps {
  bounds: BoundsSettings;
  onChange: (b: BoundsSettings) => void;
  onReset: () => void;
  /** Exactly one target enabled (the unbounded camera-axis mode is possible). */
  singleView: boolean;
  /** "Unbounded camera axis" design option (shared with the Methods panel). */
  unboundedAxis: boolean;
  onUnboundedAxis: (on: boolean) => void;
  /** Camera 0, for showing the effective box in camera-axis mode. */
  camera: Camera;
  editInViewer: boolean;
}

export function BoundsPanel(p: BoundsPanelProps) {
  const b = p.bounds;
  const viewAxis = p.singleView && p.unboundedAxis;
  const errors = boundsErrors(b, viewAxis);
  const set = (patch: Partial<BoundsSettings>) => p.onChange({ ...b, ...patch });
  const size = b.max.map((x, i) => x - b.min[i]);
  const effective = viewAxis && errors.length === 0 ? viewAxisVolume(p.camera, b.viewAxisNear, b.viewAxisFar) : null;

  return (
    <Section
      title="4 · Sculpture bounds"
      right={
        <button type="button" className="small-btn" onClick={p.onReset} title="Back to the backend defaults">
          Reset
        </button>
      }
    >
      {p.singleView && (
        <div className="field-row">
          <label>Mode</label>
          <Segmented<"box" | "axis">
            value={viewAxis ? "axis" : "box"}
            onChange={(v) => p.onUnboundedAxis(v === "axis")}
            options={[
              { value: "box", label: "Box" },
              {
                value: "axis",
                label: "Camera axis",
                title: "Unbounded camera axis: pieces may sit anywhere along Camera 1's view direction between near and far",
              },
            ]}
          />
        </div>
      )}

      {viewAxis && (
        <>
          <div className="field-row">
            <label>Near · far</label>
            <div className="vec2">
              <NumberField value={b.viewAxisNear} step={0.1} min={0} onChange={(viewAxisNear) => set({ viewAxisNear })} title="view_axis_near (distance from Camera 1)" />
              <NumberField value={b.viewAxisFar} step={0.5} min={0} onChange={(viewAxisFar) => set({ viewAxisFar })} title="view_axis_far (distance from Camera 1)" />
            </div>
          </div>
          <div className="hint small">
            B is the box around Camera 1's frustum between these distances (the target cone is carved out of it).
            The min / max box below is ignored in this mode.
          </div>
          {effective && (
            <div className="muted small mono">
              B ≈ {fmtVec(effective.min)} … {fmtVec(effective.max)}
            </div>
          )}
        </>
      )}

      <div className={viewAxis ? "dimmed" : undefined}>
        <Vec3Field label="Min" value={b.min} onChange={(min) => set({ min })} />
        <Vec3Field label="Max" value={b.max} onChange={(max) => set({ max })} />
        <div className="field-sub muted small">size {size.map((x) => x.toFixed(2)).join(" × ")}</div>
      </div>
      <ErrorText>{errors.join("\n")}</ErrorText>
      {p.editInViewer && (
        <div className="hint small">
          {viewAxis
            ? "Drag the near / far handles on Camera 1's axis in the 3D view."
            : "Drag the face handles in the 3D view; click the centre handle to move the whole box."}
        </div>
      )}
    </Section>
  );
}
