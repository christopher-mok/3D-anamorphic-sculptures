import { useEffect, useRef, useState, type RefObject } from "react";

/**
 * Rate-limits how often a fast-changing value (e.g. assembly snapshots from the
 * progress stream) propagates. When `resetKey` changes the value is applied
 * immediately (e.g. when the user switches the displayed method).
 */
export function useThrottledValue<T>(value: T, ms: number, resetKey?: unknown): T {
  const [out, setOut] = useState(value);
  const last = useRef(0);
  const lastKey = useRef(resetKey);

  useEffect(() => {
    const now = Date.now();
    const keyChanged = lastKey.current !== resetKey;
    lastKey.current = resetKey;
    const wait = ms - (now - last.current);
    if (keyChanged || wait <= 0) {
      last.current = now;
      setOut(() => value);
      return;
    }
    const t = window.setTimeout(() => {
      last.current = Date.now();
      setOut(() => value);
    }, wait);
    return () => window.clearTimeout(t);
  }, [value, ms, resetKey]);

  return out;
}

/** Tracks an element's content-box size with a ResizeObserver. */
export function useElementSize(ref: RefObject<HTMLElement>): { width: number; height: number } {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setSize({ width: el.clientWidth, height: el.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}
