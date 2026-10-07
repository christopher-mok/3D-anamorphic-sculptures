// Geometry helpers implementing the conventions in docs/api.md.
import { Color, MathUtils, Matrix4, PerspectiveCamera, Quaternion, Vector3 } from "three";
import type { Camera, ObjectInstance, Vec3 } from "../api/types";
import { isNum } from "../utils/format";

const v3 = (v: Vec3 | undefined | null, fallback: Vec3 = [0, 0, 0]) =>
  new Vector3(...(Array.isArray(v) && v.length === 3 ? v : fallback));

/** Row-major rotation matrix from an instance, falling back to rotation6d, then identity. */
function rotationRows(obj: ObjectInstance): [Vec3, Vec3, Vec3] {
  const R = obj.rotation_matrix;
  if (Array.isArray(R) && R.length === 3 && R.every((r) => Array.isArray(r) && r.length === 3)) return R;
  const r6 = obj.rotation6d;
  if (Array.isArray(r6) && r6.length === 6) {
    // rotation6d = first two columns of R: [c1x, c1y, c1z, c2x, c2y, c2z] (Gram-Schmidt).
    const a = new Vector3(r6[0], r6[1], r6[2]).normalize();
    const b2 = new Vector3(r6[3], r6[4], r6[5]);
    const b = b2.sub(a.clone().multiplyScalar(a.dot(b2))).normalize();
    const c = new Vector3().crossVectors(a, b);
    return [
      [a.x, b.x, c.x],
      [a.y, b.y, c.y],
      [a.z, b.z, c.z],
    ];
  }
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
}

/** World matrix for `x_world = scale * (R @ x_local) + translation`. */
export function instanceMatrix(obj: ObjectInstance): Matrix4 {
  const R = rotationRows(obj);
  const s = isNum(obj.scale) ? obj.scale : isNum(obj.log_scale) ? Math.exp(obj.log_scale) : 1;
  const t = obj.translation ?? [0, 0, 0];
  return new Matrix4().set(
    s * R[0][0], s * R[0][1], s * R[0][2], t[0],
    s * R[1][0], s * R[1][1], s * R[1][2], t[1],
    s * R[2][0], s * R[2][1], s * R[2][2], t[2],
    0, 0, 0, 1,
  );
}

/**
 * Orientation of an optimization camera (three.js convention: looks down -Z).
 * Uses a PerspectiveCamera because Object3D.lookAt points +Z for non-cameras.
 */
export function cameraPose(cam: Camera): { position: Vector3; quaternion: Quaternion; target: Vector3; up: Vector3 } {
  const helper = new PerspectiveCamera();
  const position = v3(cam.position, [0, 0, 5]);
  const target = v3(cam.look_at);
  const up = v3(cam.up, [0, 1, 0]);
  if (up.lengthSq() < 1e-12) up.set(0, 1, 0);
  up.normalize();
  helper.position.copy(position);
  helper.up.copy(up);
  helper.lookAt(target);
  return { position, quaternion: helper.quaternion.clone(), target, up };
}

/**
 * three.js vertical fov reproducing a square optimization image in the centered
 * square of side min(w, h) (docs/api.md "portrait-aspect fov rule").
 */
export function viewerFovDeg(fovYDeg: number, aspect: number): number {
  if (!(aspect > 0) || aspect >= 1) return fovYDeg;
  const half = MathUtils.degToRad(fovYDeg) / 2;
  return MathUtils.radToDeg(2 * Math.atan(Math.tan(half) / aspect));
}

/** Stable, subtle per-mesh-type colour. */
export function colorForMesh(name: string): Color {
  let h = 2166136261;
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const hue = ((h >>> 0) % 360) / 360;
  return new Color().setHSL(hue, 0.32, 0.64);
}

export const CAMERA_COLORS = ["#ff9f43", "#4dabf7"];
export const cameraColor = (i: number) => CAMERA_COLORS[i % CAMERA_COLORS.length];
