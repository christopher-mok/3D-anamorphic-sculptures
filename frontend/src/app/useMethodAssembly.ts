import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Assembly, JobStatus, MethodName } from "../api/types";
import { useThrottledValue } from "../utils/hooks";

/**
 * Assembly to display for a method: the latest live snapshot from the progress
 * stream, or (for finished methods without a snapshot) `<result_dir_url>/result.json`.
 * Live snapshots are throttled so the scene is not rebuilt on every event.
 */
export function useMethodAssembly(job: JobStatus | null, method: MethodName | null): Assembly | null {
  const progress = method ? job?.methods?.[method] : undefined;
  const live = progress?.assembly ?? null;
  const resultDir = progress?.status === "done" ? progress.result_dir_url : null;

  const [fetched, setFetched] = useState<{ dir: string; assembly: Assembly | null } | null>(null);

  useEffect(() => {
    if (live || !resultDir || fetched?.dir === resultDir) return;
    let cancelled = false;
    api
      .methodResult(resultDir)
      .then((r) => !cancelled && setFetched({ dir: resultDir, assembly: r?.assembly ?? null }))
      .catch(() => !cancelled && setFetched({ dir: resultDir, assembly: null }));
    return () => {
      cancelled = true;
    };
  }, [live, resultDir, fetched?.dir]);

  const fromFile = resultDir && fetched?.dir === resultDir ? fetched.assembly : null;
  const value = live ?? fromFile;
  return useThrottledValue(value, 1000, `${job?.job_id ?? ""}/${method ?? ""}`);
}
