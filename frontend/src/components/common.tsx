// Small shared UI primitives.
import { useEffect, useState, type ReactNode } from "react";
import type { Vec3 } from "../api/types";

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="spinner-wrap">
      <span className="spinner" aria-hidden />
      {label && <span>{label}</span>}
    </span>
  );
}

export function Section({ title, right, children }: { title: ReactNode; right?: ReactNode; children: ReactNode }) {
  return (
    <section className="section">
      <header className="section-header">
        <h2>{title}</h2>
        {right}
      </header>
      <div className="section-body">{children}</div>
    </section>
  );
}

/**
 * Numeric input that keeps the raw text while typing ("-", "1.", "") and only
 * commits finite numbers. Re-syncs when the value is changed externally.
 */
export function NumberField({
  value, onChange, step = 0.1, min, max, title, className,
}: {
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
  title?: string;
  className?: string;
}) {
  const [text, setText] = useState(() => String(value));
  useEffect(() => {
    if (Number(text) !== value) setText(String(value));
  }, [value]);
  return (
    <input
      className={`num ${className ?? ""}`}
      type="number"
      value={text}
      step={step}
      min={min}
      max={max}
      title={title}
      onChange={(e) => {
        setText(e.target.value);
        const v = parseFloat(e.target.value);
        if (Number.isFinite(v)) onChange(v);
      }}
      onBlur={() => setText(String(value))}
    />
  );
}

export function Vec3Field({ label, value, onChange, step = 0.1 }: {
  label: string;
  value: Vec3;
  onChange: (v: Vec3) => void;
  step?: number;
}) {
  const set = (i: number, x: number) => {
    const next = [...value] as Vec3;
    next[i] = x;
    onChange(next);
  };
  return (
    <div className="field-row">
      <label>{label}</label>
      <div className="vec3">
        {(["x", "y", "z"] as const).map((axis, i) => (
          <NumberField key={axis} value={value[i]} onChange={(x) => set(i, x)} step={step} title={axis} />
        ))}
      </div>
    </div>
  );
}

export function StatusBadge({ status }: { status: string }) {
  return <span className={`badge badge-${status}`}>{status}</span>;
}

export function ErrorText({ children }: { children: ReactNode }) {
  if (!children) return null;
  return <div className="error-text">{children}</div>;
}

export function Segmented<T extends string | number>({
  options, value, onChange,
}: {
  options: { value: T; label: ReactNode; disabled?: boolean; title?: string }[];
  value: T | null;
  onChange: (v: T) => void;
}) {
  return (
    <div className="segmented" role="tablist">
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="tab"
          aria-selected={value === o.value}
          className={value === o.value ? "active" : ""}
          disabled={o.disabled}
          title={o.title}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
