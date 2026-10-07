// Steps 2–3: target image selection (from the server list or by upload).
import { useRef, useState } from "react";
import { api, errorMessage, fileUrl } from "../api/client";
import type { TargetInfo } from "../api/types";
import { cameraColor } from "../viewer/math";
import { ErrorText, Section, Spinner } from "./common";

export function targetUrlFor(path: string | null, targets: TargetInfo[]): string | null {
  if (!path) return null;
  return targets.find((t) => t.path === path)?.url ?? fileUrl(path);
}

interface TargetSelectorProps {
  index: number;
  targets: TargetInfo[];
  value: string | null;
  onChange: (path: string | null) => void;
  onUploaded: (t: TargetInfo) => void;
}

function TargetSelector({ index, targets, value, onChange, onUploaded }: TargetSelectorProps) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const url = targetUrlFor(value, targets);
  const known = value !== null && targets.some((t) => t.path === value);

  const upload = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const info = await api.uploadTarget(file);
      onUploaded(info);
      onChange(info.path);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  return (
    <div className="target-selector">
      <div className="target-label">
        <span className="swatch" style={{ background: cameraColor(index) }} />
        Target Image {index + 1}
      </div>
      <div className="inline-form">
        <select value={value ?? ""} onChange={(e) => onChange(e.target.value || null)}>
          <option value="">— select —</option>
          {!known && value && <option value={value}>{value}</option>}
          {targets.map((t) => (
            <option key={t.path} value={t.path}>
              {t.name}
            </option>
          ))}
        </select>
        <button type="button" onClick={() => fileInput.current?.click()} disabled={uploading}>
          Upload…
        </button>
        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void upload(f);
          }}
        />
      </div>
      {uploading && <Spinner label="Uploading…" />}
      <ErrorText>{error}</ErrorText>
      {url && (
        <div className="target-preview">
          <img src={url} alt={`target ${index + 1}`} />
        </div>
      )}
    </div>
  );
}

interface TargetPanelProps {
  targets: TargetInfo[];
  target1: string | null;
  onTarget1: (p: string | null) => void;
  target2Enabled: boolean;
  onTarget2Enabled: (b: boolean) => void;
  target2: string | null;
  onTarget2: (p: string | null) => void;
  onUploaded: (t: TargetInfo) => void;
}

export function TargetPanel(p: TargetPanelProps) {
  return (
    <Section title="2 · Targets">
      <TargetSelector index={0} targets={p.targets} value={p.target1} onChange={p.onTarget1} onUploaded={p.onUploaded} />
      <label className="checkbox">
        <input type="checkbox" checked={p.target2Enabled} onChange={(e) => p.onTarget2Enabled(e.target.checked)} />
        Enable Target Image 2
      </label>
      {p.target2Enabled && (
        <TargetSelector index={1} targets={p.targets} value={p.target2} onChange={p.onTarget2} onUploaded={p.onUploaded} />
      )}
    </Section>
  );
}
