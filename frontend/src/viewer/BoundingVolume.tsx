import { useMemo } from "react";
import { Line } from "@react-three/drei";
import { Vector3 } from "three";
import type { BoundingVolume as BV, Vec3 } from "../api/types";

/** The 12 edges of an axis-aligned box as 24 segment endpoints. */
export function boxEdges(min: Vec3, max: Vec3): Vector3[] {
  const c = (x: number, y: number, z: number) =>
    new Vector3(x ? max[0] : min[0], y ? max[1] : min[1], z ? max[2] : min[2]);
  const out: Vector3[] = [];
  for (const [a, b] of [
    [[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 0]], [[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 1]],
    [[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]], [[0, 0, 1], [0, 1, 1]], [[1, 0, 1], [1, 1, 1]],
    [[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 1]], [[0, 1, 0], [0, 1, 1]], [[1, 1, 0], [1, 1, 1]],
  ] as const) {
    out.push(c(a[0], a[1], a[2]), c(b[0], b[1], b[2]));
  }
  return out;
}

/** Axis-aligned sculpture bounding volume (default [-1, 1]^3). */
export function BoundingVolume({
  volume, color = "#5c7cfa", dashed = false, lineWidth = 1.25,
}: {
  volume: BV;
  color?: string;
  dashed?: boolean;
  lineWidth?: number;
}) {
  const points = useMemo(() => boxEdges(volume.min, volume.max), [volume.min, volume.max]);
  return (
    <Line
      points={points}
      segments
      color={color}
      lineWidth={lineWidth}
      dashed={dashed}
      dashSize={0.12}
      gapSize={0.08}
      transparent={dashed}
      opacity={dashed ? 0.6 : 1}
    />
  );
}
