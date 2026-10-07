// Step 4: numeric editing of the optimization cameras (one per enabled target).
import type { Camera } from "../api/types";
import { cameraColor } from "../viewer/math";
import { NumberField, Section, Vec3Field } from "./common";

interface CameraEditorProps {
  cameras: Camera[];
  count: number;
  defaults: Camera[];
  onChange: (i: number, cam: Camera) => void;
  selected: number | null;
  onSelect: (i: number | null) => void;
}

export function CameraEditor({ cameras, count, defaults, onChange, selected, onSelect }: CameraEditorProps) {
  return (
    <Section title="3 · Cameras">
      {cameras.slice(0, count).map((cam, i) => {
        const set = (patch: Partial<Camera>) => onChange(i, { ...cam, ...patch });
        const dist = Math.hypot(...cam.position.map((x, k) => x - cam.look_at[k]));
        return (
          <div className={`camera-card ${selected === i ? "selected" : ""}`} key={i}>
            <div className="camera-header">
              <span className="swatch" style={{ background: cameraColor(i) }} />
              <strong>Camera {i + 1}</strong>
              <span className="muted small">dist {dist.toFixed(2)}</span>
              <span className="spacer" />
              <button
                type="button"
                className={`small-btn ${selected === i ? "active" : ""}`}
                onClick={() => onSelect(selected === i ? null : i)}
                title="Select in the 3D view and drag with the gizmo"
              >
                {selected === i ? "Deselect" : "Drag in 3D"}
              </button>
              {defaults[i] && (
                <button type="button" className="small-btn" onClick={() => onChange(i, defaults[i])}>
                  Reset
                </button>
              )}
            </div>
            <Vec3Field label="Position" value={cam.position} onChange={(position) => set({ position })} />
            <Vec3Field label="Look at" value={cam.look_at} onChange={(look_at) => set({ look_at })} />
            <Vec3Field label="Up" value={cam.up} onChange={(up) => set({ up })} />
            <div className="field-row">
              <label>FOV y°</label>
              <div className="vec3">
                <NumberField value={cam.fov_y_deg} step={1} min={1} max={170} onChange={(fov_y_deg) => set({ fov_y_deg })} title="vertical fov (deg)" />
                <NumberField value={cam.near} step={0.05} min={0} onChange={(near) => set({ near })} title="near" />
                <NumberField value={cam.far} step={1} min={0} onChange={(far) => set({ far })} title="far" />
              </div>
            </div>
            <div className="field-sub muted small">fov · near · far</div>
          </div>
        );
      })}
    </Section>
  );
}
