// Job progress subscription: WebSocket first, polling GET /api/jobs/{id}
// every 2 s as a fallback when the socket cannot be opened or drops.
import { useEffect, useState } from "react";
import { api, errorMessage } from "./client";
import type { JobState, JobStatus, MethodProgress, WsEvent } from "./types";

export type ConnectionMode = "idle" | "connecting" | "websocket" | "polling" | "closed";

const TERMINAL: ReadonlySet<JobState> = new Set<JobState>(["done", "failed", "cancelled"]);
export const isTerminal = (s: JobState | undefined | null): boolean => !!s && TERMINAL.has(s);

const POLL_MS = 2000;
const WS_OPEN_TIMEOUT_MS = 5000;

export interface JobStream {
  job: JobStatus | null;
  connection: ConnectionMode;
  /** ms timestamp of the last received update (used for image cache-busting). */
  lastUpdate: number;
  error: string | null;
}

function jobSocketUrl(jobId: string): string {
  // Relative to the page host so the Vite dev proxy (ws: true) forwards it.
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/api/jobs/${encodeURIComponent(jobId)}/ws`;
}

/** Merge a throttled progress event; keep the previous assembly snapshot if the event omits it. */
function mergeProgress(prev: MethodProgress | undefined, next: MethodProgress): MethodProgress {
  if (!prev) return next;
  return { ...next, assembly: next.assembly ?? prev.assembly ?? null };
}

export function useJobStream(jobId: string | null): JobStream {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [connection, setConnection] = useState<ConnectionMode>("idle");
  const [lastUpdate, setLastUpdate] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setJob(null);
    setError(null);
    if (!jobId) {
      setConnection("idle");
      return;
    }

    let disposed = false;
    let finished = false;
    let polling = false;
    let pollTimer: number | undefined;
    let ws: WebSocket | null = null;

    const touch = () => setLastUpdate(Date.now());

    const applyStatus = (s: JobStatus) => {
      if (disposed) return;
      setJob(s);
      touch();
      if (isTerminal(s.status)) finished = true;
    };

    const poll = async () => {
      if (disposed) return;
      try {
        applyStatus(await api.getJob(jobId));
        setError(null);
      } catch (e) {
        if (!disposed) setError(errorMessage(e));
      }
      if (disposed) return;
      if (finished) setConnection("closed");
      else pollTimer = window.setTimeout(poll, POLL_MS);
    };

    const startPolling = () => {
      if (disposed || polling) return;
      polling = true;
      setConnection("polling");
      void poll();
    };

    const handle = (ev: WsEvent) => {
      if (disposed) return;
      switch (ev.type) {
        case "status":
          applyStatus(ev.data);
          break;
        case "done":
          applyStatus(ev.data);
          finished = true;
          break;
        case "progress":
          setJob((prev) =>
            prev
              ? {
                  ...prev,
                  methods: { ...prev.methods, [ev.method]: mergeProgress(prev.methods[ev.method], ev.data) },
                }
              : prev,
          );
          touch();
          break;
        case "comparison":
          setJob((prev) => (prev ? { ...prev, comparison: ev.data } : prev));
          touch();
          break;
        case "error":
          setError(ev.data?.message ?? "Unknown error");
          break;
      }
    };

    try {
      setConnection("connecting");
      ws = new WebSocket(jobSocketUrl(jobId));
      ws.onopen = () => {
        if (!disposed) setConnection("websocket");
      };
      ws.onmessage = (m: MessageEvent) => {
        try {
          handle(JSON.parse(String(m.data)) as WsEvent);
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onclose = () => {
        if (disposed) return;
        if (finished) setConnection("closed");
        else startPolling();
      };
    } catch {
      startPolling();
    }

    // If the socket never opens (e.g. a proxy without ws support), poll instead.
    const openTimeout = window.setTimeout(() => {
      if (ws && ws.readyState !== WebSocket.OPEN && !finished) {
        ws.onclose = null;
        ws.close();
        startPolling();
      }
    }, WS_OPEN_TIMEOUT_MS);

    return () => {
      disposed = true;
      window.clearTimeout(openTimeout);
      if (pollTimer !== undefined) window.clearTimeout(pollTimer);
      if (ws) {
        ws.onclose = null;
        ws.onmessage = null;
        ws.close();
      }
    };
  }, [jobId]);

  return { job, connection, lastUpdate, error };
}
