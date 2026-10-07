// Step 1: model folder + grid of normalized models.
import type { ModelInfo } from "../api/types";
import { fmtInt, fmtVec } from "../utils/format";
import { ErrorText, Section, Spinner } from "./common";

interface ModelPanelProps {
  dir: string;
  onDirChange: (dir: string) => void;
  onLoad: () => void;
  models: ModelInfo[] | null;
  loading: boolean;
  error: string | null;
}

export function ModelPanel({ dir, onDirChange, onLoad, models, loading, error }: ModelPanelProps) {
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
      {models && models.length === 0 && !loading && <div className="hint">No models found in this folder.</div>}
      {models && models.length > 0 && (
        <div className="model-grid">
          {models.map((m) => (
            <div className="model-card" key={m.name} title={`${m.filename}\nproxy: ${fmtInt(m.proxy_triangles)} tris`}>
              <div className="thumb">
                {m.thumbnail_url ? <img src={m.thumbnail_url} alt={m.name} loading="lazy" /> : <span>?</span>}
              </div>
              <div className="model-name">{m.filename || m.name}</div>
              <div className="model-meta">{fmtInt(m.triangles)} tris</div>
              <div className="model-meta">{fmtVec(m.dims, 2)}</div>
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}
