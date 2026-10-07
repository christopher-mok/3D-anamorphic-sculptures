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

/** Unit viewing direction (eye -> look_at) of an optimization camera. */
export function cameraForward(cam: Camera): Vector3 {
  const d = v3(cam.look_at).sub(v3(cam.position, [0, 0, 5]));
  return d.lengthSq() < 1e-12 ? new Vector3(0, 0, -1) : d.normalize();
}

/**
 * Corners of the square-aspect frustum segment of `cam` between distances `near`
 * and `far` (world space): [near tl, tr, br, bl, far tl, tr, br, bl].
 */
export function frustumSegmentCorners(cam: Camera, near: number, far: number): Vector3[] {
  const pose = cameraPose(cam);
  const t = Math.tan(MathUtils.degToRad(cam.fov_y_deg || 1) / 2);
  const out: Vector3[] = [];
  for (const d of [near, far]) {
    for (const [sx, sy] of [[-1, 1], [1, 1], [1, -1], [-1, -1]]) {
      out.push(new Vector3(sx * t * d, sy * t * d, -d).applyQuaternion(pose.quaternion).add(pose.position));
    }
  }
  return out;
}

/**
 * Axis-aligned box around the frustum segment: what the backend uses as B when
 * bounding_volume.unbounded_view_axis is on (mirrors context.view_axis_volume).
 */
export function viewAxisVolume(cam: Camera, near: number, far: number): { min: Vec3; max: Vec3 } {
  const pts = frustumSegmentCorners(cam, near, far);
  const r = (x: number) => Math.round(x * 1e4) / 1e4;
  const min = [0, 1, 2].map((i) => r(Math.min(...pts.map((p) => p.getComponent(i))))) as Vec3;
  const max = [0, 1, 2].map((i) => r(Math.max(...pts.map((p) => p.getComponent(i))))) as Vec3;
  return { min, max };
}

/**
 * Parameter t of the point on the line `origin + t * dir` (dir unit length) closest
 * to the ray `rayOrigin + s * rayDir` (unit). Null when the ray is (nearly) parallel
 * to the line or the closest point lies behind the ray origin.
 */
export function closestLineParam(origin: Vector3, dir: Vector3, rayOrigin: Vector3, rayDir: Vector3): number | null {
  const w0 = origin.clone().sub(rayOrigin);
  const b = dir.dot(rayDir);
  const denom = 1 - b * b;
  if (denom < 1e-4) return null;
  const d = dir.dot(w0);
  const e = rayDir.dot(w0);
  const t = (b * e - d) / denom;
  const s = e + t * b;
  return s > 0 ? t : null;
}
