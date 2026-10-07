import { useMemo } from "react";
import { Box3, Box3Helper, Color, Vector3 } from "three";
import type { BoundingVolume as BV } from "../api/types";

/** Axis-aligned sculpture bounding volume (default [-1, 1]^3). */
export function BoundingVolume({ volume, color = "#5c7cfa" }: { volume: BV; color?: string }) {
  const helper = useMemo(() => {
    const box = new Box3(new Vector3(...volume.min), new Vector3(...volume.max));
    return new Box3Helper(box, new Color(color));
  }, [volume, color]);
  return <primitive object={helper} />;
}
