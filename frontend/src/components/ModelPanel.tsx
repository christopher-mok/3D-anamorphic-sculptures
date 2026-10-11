// Step 1: model folder + grid of normalized models.
import type { ModelInfo } from "../api/types";
import { fmtInt, fmtVec } from "../utils/format";
import { ErrorText, Section, Spinner } from "./common";

interface ModelPanelProps {
  dir: string;
  onDirChange: (dir: string) => void;
  onLoad: () => void;
  models: ModelInfo[] | null;
  selected: ReadonlySet<string>;
  previewed: ReadonlySet<string>;
  onSelected: (name: string, selected: boolean) => void;
  onPreviewed: (name: string, previewed: boolean) => void;
  globalScale: number;
  onGlobalScale: (factor: number) => void;
  scaleFactors: Record<string, number>;
  onScaleFactor: (name: string, factor: number) => void;
  fixedColors: Record<string, string | null>;
  onFixedColor: (name: string, color: string | null) => void;
  loading: boolean;
  error: string | null;
}

export function ModelPanel({
  dir, onDirChange, onLoad, models, selected, previewed, onSelected, onPreviewed,
  globalScale, onGlobalScale, scaleFactors, onScaleFactor, fixedColors, onFixedColor, loading, error,
}: ModelPanelProps) {
  return (
    <Section
      title="1 · Models"
      right={models && <span className="muted small">{models.length} model{models.length === 1 ? "" : "s"}</span>}
    >
      <form
        className="inline-form"
        onSubmit={(e) => {
          e.preventDefault();
          onLoad();
        }}
      >
        <input
          type="text"
          value={dir}
          onChange={(e) => onDirChange(e.target.value)}
          placeholder="assets/models"
          spellCheck={false}
        />
        <button type="submit" disabled={loading || !dir.trim()}>
          Load
        </button>
      </form>
      {loading && (
        <div className="hint">
          <Spinner label="Loading & preprocessing models (can take a while the first time)…" />
        </div>
      )}
      <ErrorText>{error}</ErrorText>
      <div className="field-row wide" title="Multiplier applied to every model, in addition to each model's individual scale">
        <label>Global scale</label>
        <input
          className="num"
          type="number"
          min={0.01}
          max={100}
          step={0.05}
          value={globalScale}
          onChange={(e) => onGlobalScale(Number(e.target.value))}
          aria-label="global model scale"
        />
      </div>
      {models && models.length === 0 && !loading && <div className="hint">No models found in this folder.</div>}
      {models && models.length > 0 && (
        <div className="model-grid">
          {models.map((m) => (
            <div className={`model-card ${selected.has(m.name) ? "model-selected" : "model-disabled"}`} key={m.name} title={`${m.filename}\nproxy: ${fmtInt(m.proxy_triangles)} tris`}>
              <div className="thumb">
                {m.thumbnail_url ? <img src={m.thumbnail_url} alt={m.name} loading="lazy" /> : <span>?</span>}
              </div>
              <div className="model-name">{m.filename || m.name}</div>
              <div className="model-meta">{fmtInt(m.triangles)} tris</div>
              <div className="model-meta">{fmtVec(m.dims, 2)}</div>
              <div className="model-toggles">
                <label><input type="checkbox" checked={selected.has(m.name)} onChange={(e) => onSelected(m.name, e.target.checked)} /> Use</label>
                <label><input type="checkbox" checked={previewed.has(m.name)} onChange={(e) => onPreviewed(m.name, e.target.checked)} /> Preview</label>
              </div>
              <label className="model-scale" title="Multiplier applied to this model's world-size range before optimization">
                <span>Scale</span>
                <input
                  type="number"
                  min={0.01}
                  max={100}
                  step={0.05}
                  value={scaleFactors[m.name] ?? 1}
                  onChange={(e) => onScaleFactor(m.name, Number(e.target.value))}
                  aria-label={`${m.filename || m.name} scale`}
                />
              </label>
              <div className="model-color" title="Automatic assigns a separate target-matched color to every placed instance">
                <select value={fixedColors[m.name] ? "fixed" : "auto"} onChange={(e) => onFixedColor(m.name, e.target.value === "fixed" ? (fixedColors[m.name] ?? "#808080") : null)}>
                  <option value="auto">Auto color</option>
                  <option value="fixed">Fixed color</option>
                </select>
                {fixedColors[m.name] && <input type="color" value={fixedColors[m.name] ?? "#808080"} onChange={(e) => onFixedColor(m.name, e.target.value)} aria-label={`${m.filename || m.name} fixed color`} />}
              </div>
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}
