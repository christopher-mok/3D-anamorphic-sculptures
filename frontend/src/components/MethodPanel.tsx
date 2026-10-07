// Steps 5–6: method selection and quality preset (with Custom JSON overrides).
import type { MethodName } from "../api/types";
import { METHOD_LABELS } from "../api/types";
import { ErrorText, Section } from "./common";

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

/** Design options exposed as UI controls; turned into config overrides (see configs/default.yaml). */
export interface DesignOptions {
  fixedScale: boolean;          // scale.mode = "fixed": size comes from depth only
  fixedScaleValue: number;      // scale.fixed (world bounding radius of every piece)
  unboundedAxis: boolean;       // bounding_volume.unbounded_view_axis (single view only)
  diversityWeight: number;      // diversity.weight (0 = off)
}

export const DEFAULT_DESIGN_OPTIONS: DesignOptions = {
  fixedScale: false,
  fixedScaleValue: 0.15,
  unboundedAxis: false,
  diversityWeight: 0,
};

export function designOverrides(o: DesignOptions, singleView: boolean): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (o.fixedScale) out.scale = { mode: "fixed", fixed: o.fixedScaleValue };
  if (o.unboundedAxis && singleView) out.bounding_volume = { unbounded_view_axis: true };
  if (o.diversityWeight > 0) out.diversity = { weight: o.diversityWeight };
  return out;
}

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
      <div className="checkbox-list">
        {p.available.map((m) => (
          <label className="checkbox" key={m}>
            <input type="checkbox" checked={p.selected.includes(m)} onChange={(e) => p.onToggle(m, e.target.checked)} />
            {METHOD_LABELS[m] ?? m}
          </label>
        ))}
      </div>
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
      <div className="checkbox-list">
        <label className="checkbox" title="Every piece has the same size; pieces that must look bigger move toward the camera">
          <input type="checkbox" checked={p.options.fixedScale} onChange={(e) => p.onOptions({ ...p.options, fixedScale: e.target.checked })} />
          Fixed piece size (depth sets apparent size)
        </label>
        {p.options.fixedScale && (
          <div className="field-row">
            <label>Piece radius</label>
            <input
              type="number"
              step={0.01}
              min={0.02}
              max={1}
              value={p.options.fixedScaleValue}
              onChange={(e) => p.onOptions({ ...p.options, fixedScaleValue: Number(e.target.value) || 0.15 })}
            />
          </div>
        )}
        <label
          className="checkbox"
          title={p.singleView ? "Pieces may go closer to / farther from the camera than the bounding box" : "Only available with a single target view"}
          style={p.singleView ? undefined : { opacity: 0.5 }}
        >
          <input
            type="checkbox"
            disabled={!p.singleView}
            checked={p.options.unboundedAxis && p.singleView}
            onChange={(e) => p.onOptions({ ...p.options, unboundedAxis: e.target.checked })}
          />
          Unbounded camera axis (single view)
        </label>
      </div>
      <div className="field-row" title="Rewards using all piece types evenly (0 = off)">
        <label>Type variety {p.options.diversityWeight > 0 ? p.options.diversityWeight.toFixed(2) : "off"}</label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={p.options.diversityWeight}
          onChange={(e) => p.onOptions({ ...p.options, diversityWeight: Number(e.target.value) })}
        />
      </div>
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
