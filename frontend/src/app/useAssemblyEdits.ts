import { useCallback, useMemo, useState } from "react";
import type { Assembly, ObjectInstance } from "../api/types";

/**
 * Client-only edits of result assemblies (lock-and-rerun), one edited copy per
 * `key` (= "<job_id>/<method>"). Nothing is persisted: a reload shows the
 * original results again. The base assembly is never mutated.
 */
export interface AssemblyEdits {
  /** The edited assembly for `key`, else the base assembly. */
  assembly: Assembly | null;
  /** The user changed something for this key (until "Reset edits"). */
  edited: boolean;
  /** Applies an immutable update to the objects (starting from the base on the first edit). */
  update: (fn: (objects: ObjectInstance[]) => ObjectInstance[]) => void;
  reset: () => void;
}

export function useAssemblyEdits(key: string | null, base: Assembly | null): AssemblyEdits {
  const [edits, setEdits] = useState<Record<string, ObjectInstance[]>>({});
  const current = key ? edits[key] : undefined;

  const assembly = useMemo<Assembly | null>(() => (current ? { objects: current } : base), [current, base]);

  const update = useCallback(
    (fn: (objects: ObjectInstance[]) => ObjectInstance[]) => {
      if (!key) return;
      setEdits((all) => {
        const from = all[key] ?? base?.objects ?? null;
        if (!from) return all;
        return { ...all, [key]: fn(from) };
      });
    },
    [key, base],
  );

  const reset = useCallback(() => {
    if (!key) return;
    setEdits((all) => {
      if (!(key in all)) return all;
      const next = { ...all };
      delete next[key];
      return next;
    });
  }, [key]);

  return { assembly, edited: !!current, update, reset };
}
