// Number / time formatting that tolerates missing values.

export const DASH = "—";

export function isNum(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

export function fmt(x: number | null | undefined, digits = 3): string {
  return isNum(x) ? x.toFixed(digits) : DASH;
}

export function fmtInt(x: number | null | undefined): string {
  return isNum(x) ? Math.round(x).toLocaleString() : DASH;
}

export function fmtPct(x: number | null | undefined, digits = 1): string {
  return isNum(x) ? `${(x * 100).toFixed(digits)}%` : DASH;
}

export function fmtTime(s: number | null | undefined): string {
  if (!isNum(s)) return DASH;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  if (m < 60) return `${m}m ${String(sec).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

export function fmtDate(unixSeconds: number | null | undefined): string {
  if (!isNum(unixSeconds)) return DASH;
  return new Date(unixSeconds * 1000).toLocaleString();
}

export function fmtVec(v: readonly number[] | null | undefined, digits = 2, sep = " × "): string {
  if (!v || v.length === 0) return DASH;
  return v.map((x) => fmt(x, digits)).join(sep);
}
