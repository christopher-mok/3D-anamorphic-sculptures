// Steps 5–6: method selection and quality preset (with Custom JSON overrides).
import type { MethodName } from "../api/types";
import { ErrorText, Section, Segmented } from "./common";

export type PresetChoice = "fast" | "default" | "high_quality" | "custom";

const PRESET_OPTIONS: { value: PresetChoice; label: string }[] = [
  { value: "fast", label: "Fast" },
  { value: "default", label: "Medium" },
  { value: "high_quality", label: "High" },
  { value: "custom", label: "Custom" },
];

/** Parses the Custom overrides textarea; returns an error string for invalid input. */
export function parseOverrides(text: string): { value: Record<string, unknown> | null; error: string | null } {
  if (!text.trim()) return { value: {}, error: null };
  try {
    const v: unknown = JSON.parse(text);
    if (!v || typeof v !== "object" || Array.isArray(v)) return { value: null, error: "Overrides must be a JSON object." };
    return { value: v as Record<string, unknown>, error: null };
  } catch (e) {
    return { value: null, error: `Invalid JSON: ${(e as Error).message}` };
  }
}

export type ScaleMode = "free" | "fixed" | "native";

/** Design options exposed as UI controls; turned into config overrides (see configs/default.yaml). */
export interface DesignOptions {
  scaleMode: ScaleMode;         // scale.mode: optimized per piece / uniform fixed / from the model files
  fixedScaleValue: number;      // scale.fixed (world bounding radius of every piece)
  nativeFactor: number;         // scale.native_factor (world radius = original radius x factor)
  matchWholeImage: boolean;     // targets.mask_background = false
  unboundedAxis: boolean;       // bounding_volume.unbounded_view_axis (single view only)
  diversityWeight: number;      // diversity.weight (0 = off)
  viewingZoneRadius: number;    // viewing_zone.radius (0 = off)
  viewingZoneSamples: number;   // viewing_zone.samples
  reveal: boolean;              // reveal.weight > 0: look unlike the target from other angles
  revealWeight: number;         // reveal.weight
}

export const DEFAULT_DESIGN_OPTIONS: DesignOptions = {
  scaleMode: "free",
  fixedScaleValue: 0.15,
  nativeFactor: 0.3,
  matchWholeImage: false,
  unboundedAxis: false,
  diversityWeight: 0,
  viewingZoneRadius: 0,
  viewingZoneSamples: 4,
  reveal: false,
  revealWeight: 0.05,
};

/** Config overrides for the design options; only keys that differ from the defaults. */
export function designOverrides(o: DesignOptions, singleView: boolean): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const d = DEFAULT_DESIGN_OPTIONS;
  const scale: Record<string, unknown> = {};
  if (o.scaleMode === "fixed") Object.assign(scale, { mode: "fixed", fixed: o.fixedScaleValue });
  if (o.scaleMode === "native") Object.assign(scale, { mode: "native", native_factor: o.nativeFactor });
  if (Object.keys(scale).length) out.scale = scale;
  if (o.matchWholeImage) out.targets = { mask_background: false };
  if (o.unboundedAxis && singleView) out.bounding_volume = { unbounded_view_axis: true };
  if (o.diversityWeight > 0) out.diversity = { weight: o.diversityWeight };
  if (o.viewingZoneRadius > 0) {
    const vz: Record<string, unknown> = { radius: o.viewingZoneRadius };
    if (o.viewingZoneSamples !== d.viewingZoneSamples) vz.samples = o.viewingZoneSamples;
    out.viewing_zone = vz;
  }
  if (o.reveal && o.revealWeight > 0) out.reveal = { weight: o.revealWeight };
  return out;
}

/** viewing_zone.radius of a job's overrides (0 when absent), for drawing the zone rings. */
export function viewingZoneRadiusOf(overrides: Record<string, unknown> | null | undefined): number {
  const vz = overrides?.viewing_zone;
  const r = vz && typeof vz === "object" ? (vz as Record<string, unknown>).radius : undefined;
  return typeof r === "number" && Number.isFinite(r) && r > 0 ? r : 0;
}

const SCALE_MODES: { value: ScaleMode; label: string; title: string }[] = [
  { value: "free", label: "Free", title: "Free (optimized): every piece gets its own optimized size" },
  { value: "fixed", label: "Uniform", title: "Uniform fixed: every piece has the same size; depth sets the apparent size" },
  { value: "native", label: "Native", title: "Native (from model files): pieces keep their original relative sizes" },
];

/** Recursive merge; values in `b` win. */
export function deepMerge(a: Record<string, unknown>, b: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...a };
  for (const [k, v] of Object.entries(b)) {
    const av = out[k];
    out[k] =
      v && typeof v === "object" && !Array.isArray(v) && av && typeof av === "object" && !Array.isArray(av)
        ? deepMerge(av as Record<string, unknown>, v as Record<string, unknown>)
        : v;
  }
  return out;
}

interface MethodPanelProps {
  available: MethodName[];
  selected: MethodName[];
  onToggle: (m: MethodName, on: boolean) => void;
  preset: PresetChoice;
  onPreset: (p: PresetChoice) => void;
  overridesText: string;
  onOverridesText: (t: string) => void;
  options: DesignOptions;
  onOptions: (o: DesignOptions) => void;
  singleView: boolean;
}

export function MethodPanel(p: MethodPanelProps) {
  const parsed = p.preset === "custom" ? parseOverrides(p.overridesText) : null;
  return (
    <Section title="5 · Methods & quality">
      <div className="field-row"><label>Optimizer</label><span>Chained</span></div>
      <div className="field-row">
        <label>Quality</label>
        <select value={p.preset} onChange={(e) => p.onPreset(e.target.value as PresetChoice)}>
          {PRESET_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>
      <div className="hint small" style={{ marginTop: 8 }}>Design options</div>
      <DesignOptionsControls options={p.options} onOptions={p.onOptions} singleView={p.singleView} />
      {p.preset === "custom" && (
        <>
          <div className="hint small">Nested config overrides applied on top of the Medium ("default") preset.</div>
          <textarea
            className="code"
            rows={8}
            spellCheck={false}
            value={p.overridesText}
            onChange={(e) => p.onOverridesText(e.target.value)}
            placeholder='{"beam": {"beam_width": 8}}'
          />
          <ErrorText>{parsed?.error}</ErrorText>
        </>
      )}
    </Section>
  );
}

function DesignOptionsControls({
  options: o, onOptions, singleView,
}: { options: DesignOptions; onOptions: (o: DesignOptions) => void; singleView: boolean }) {
  const set = (patch: Partial<DesignOptions>) => onOptions({ ...o, ...patch });
  return (
    <>
      <div className="field-row wide" title="How piece sizes are chosen (scale.mode)">
        <label>Piece size</label>
        <Segmented<ScaleMode> value={o.scaleMode} onChange={(scaleMode) => set({ scaleMode })} options={SCALE_MODES} />
      </div>
      <div className="hint small">{SCALE_MODES.find((m) => m.value === o.scaleMode)?.title}</div>
      <label className="checkbox" title="Treat every image pixel, including the background, as an area that pieces should cover and color-match">
        <input type="checkbox" checked={o.matchWholeImage} onChange={(e) => set({ matchWholeImage: e.target.checked })} />
        Match whole image (include background)
      </label>
      {o.scaleMode === "fixed" && (
        <div className="field-row wide" title="World bounding radius of every piece (scale.fixed)">
          <label>Piece radius</label>
          <input
            className="num"
            type="number"
            step={0.01}
            min={0.02}
            max={1}
            value={o.fixedScaleValue}
            onChange={(e) => set({ fixedScaleValue: Number(e.target.value) || 0.15 })}
          />
        </div>
      )}
      {o.scaleMode === "native" && (
        <div className="field-row wide" title="World bounding radius = the model's original bounding radius × factor (scale.native_factor)">
          <label>Native size factor</label>
          <input
            className="num"
            type="number"
            step={0.01}
            min={0.001}
            value={o.nativeFactor}
            onChange={(e) => set({ nativeFactor: Number(e.target.value) > 0 ? Number(e.target.value) : 0.3 })}
          />
        </div>
      )}
      <label
        className="checkbox"
        title={singleView ? "Pieces may go closer to / farther from the camera than the bounding box" : "Only available with a single target view"}
        style={singleView ? undefined : { opacity: 0.5 }}
      >
        <input
          type="checkbox"
          disabled={!singleView}
          checked={o.unboundedAxis && singleView}
          onChange={(e) => set({ unboundedAxis: e.target.checked })}
        />
        Unbounded camera axis (single view)
      </label>
      <div className="field-row wide" title="Rewards using all piece types evenly (0 = off)">
        <label>Type variety {o.diversityWeight > 0 ? o.diversityWeight.toFixed(2) : "off"}</label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={o.diversityWeight}
          onChange={(e) => set({ diversityWeight: Number(e.target.value) })}
        />
      </div>
      <div className="field-row wide" title="viewing_zone.radius (0 = off)">
        <label>Viewing zone {o.viewingZoneRadius > 0 ? o.viewingZoneRadius.toFixed(2) : "off"}</label>
        <input
          type="range"
          min={0}
          max={0.6}
          step={0.01}
          value={o.viewingZoneRadius}
          onChange={(e) => set({ viewingZoneRadius: Number(e.target.value) })}
          aria-label="Viewing zone radius"
        />
      </div>
      {o.viewingZoneRadius > 0 && (
        <>
          <div className="field-row wide" title="Eye positions sampled on the zone circle (viewing_zone.samples)">
            <label>Zone samples</label>
            <input
              className="num"
              type="number"
              step={1}
              min={2}
              max={8}
              value={o.viewingZoneSamples}
              onChange={(e) => {
                const n = Math.round(Number(e.target.value));
                set({ viewingZoneSamples: Number.isFinite(n) ? Math.min(8, Math.max(2, n)) : 4 });
              }}
            />
          </div>
          <div className="hint small">
            The illusion must hold for any eye position within this radius of the camera (perpendicular to the view
            direction). Costs extra views.
          </div>
        </>
      )}
      <label className="checkbox" title="Penalize resembling the target from other viewing angles (reveal.weight)">
        <input type="checkbox" checked={o.reveal} onChange={(e) => set({ reveal: e.target.checked })} />
        Reveal: look unlike the target from other angles
      </label>
      {o.reveal && (
        <div className="field-row wide" title="reveal.weight">
          <label>Strength {o.revealWeight.toFixed(2)}</label>
          <input
            type="range"
            min={0.01}
            max={0.2}
            step={0.01}
            value={o.revealWeight}
            onChange={(e) => set({ revealWeight: Number(e.target.value) })}
          />
        </div>
      )}
    </>
  );
}
