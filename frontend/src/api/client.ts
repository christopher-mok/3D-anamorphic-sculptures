// Thin fetch helpers for the REST endpoints in docs/api.md.
import type {
  Defaults, Health, JobRequest, JobStatus, MethodResultFile, ModelInfo, TargetInfo,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (e) {
    throw new ApiError(`Network error: ${(e as Error).message}`, 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body: unknown = await res.json();
      if (body && typeof body === "object" && "detail" in body) {
        const d = (body as { detail: unknown }).detail;
        detail = typeof d === "string" ? d : JSON.stringify(d);
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(`${res.status} ${detail}`.trim(), res.status);
  }
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

export const api = {
  health: () => request<Health>("/api/health"),
  defaults: () => request<Defaults>("/api/defaults"),
  models: (dir: string) => request<ModelInfo[]>(`/api/models?dir=${enc(dir)}`),
  targets: () => request<TargetInfo[]>("/api/targets"),
  uploadTarget: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<TargetInfo>("/api/targets/upload", { method: "POST", body: form });
  },
  createJob: (req: JobRequest) =>
    request<{ job_id: string }>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
  listJobs: () => request<JobStatus[]>("/api/jobs"),
  getJob: (id: string) => request<JobStatus>(`/api/jobs/${enc(id)}`),
  cancelJob: (id: string) =>
    request<{ ok: boolean }>(`/api/jobs/${enc(id)}/cancel`, { method: "POST" }),
  methodResult: (resultDirUrl: string) =>
    request<MethodResultFile>(`${trimSlash(resultDirUrl)}/result.json`),
};

export function trimSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

/** GLB of a canonical normalized model (fallback when no ModelInfo.mesh_url is known). */
export function modelMeshUrl(name: string, dir: string): string {
  return `/api/models/${enc(name)}/mesh.glb?dir=${enc(dir)}`;
}

/** Serves an image under assets/ or uploads/ by server-side path. */
export function fileUrl(path: string): string {
  return `/api/files?path=${enc(path)}`;
}

/** Append a `t=` cache-busting query parameter. */
export function withCacheBust(url: string, t: number): string {
  return `${url}${url.includes("?") ? "&" : "?"}t=${t}`;
}

export function errorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}
