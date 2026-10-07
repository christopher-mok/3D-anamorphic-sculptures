// Moves the viewer camera to exactly match an optimization camera ("Target View N"),
// with a short animation; the final pose is set from the camera values directly.
import { useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { PerspectiveCamera, Vector3 } from "three";
import type { Camera } from "../api/types";
import { viewerFovDeg } from "./math";

/** Minimal surface of drei's OrbitControls we rely on (avoids importing three-stdlib types). */
interface OrbitLike {
  enabled: boolean;
  target: Vector3;
  update: () => void;
}

export const FREE_CAMERA = { fov: 40, near: 0.01, far: 500 };
const ANIM_MS = 650;

interface Anim {
  start: number;
  pos: Vector3;
  target: Vector3;
  up: Vector3;
  fov: number;
}

const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

export function ViewerCameraController({ viewIndex, cameras }: { viewIndex: number | null; cameras: Camera[] }) {
  const camera = useThree((s) => s.camera) as PerspectiveCamera;
  const controls = useThree((s) => s.controls) as unknown as OrbitLike | null;
  const size = useThree((s) => s.size);
  const anim = useRef<Anim | null>(null);
  const wasTargetView = useRef(false);

  useEffect(() => {
    if (viewIndex === null) {
      if (controls) controls.enabled = true;
      if (wasTargetView.current) {
        // Back to free orbit: restore a generic projection and a +Y up vector
        // (OrbitControls assumes the up vector it was created with).
        camera.up.set(0, 1, 0);
        camera.fov = FREE_CAMERA.fov;
        camera.near = FREE_CAMERA.near;
        camera.far = FREE_CAMERA.far;
        camera.updateProjectionMatrix();
        controls?.update();
      }
      wasTargetView.current = false;
      anim.current = null;
      return;
    }
    if (controls) controls.enabled = false;
    const forward = new Vector3();
    camera.getWorldDirection(forward);
    anim.current = {
      start: performance.now(),
      pos: camera.position.clone(),
      target: controls ? controls.target.clone() : camera.position.clone().add(forward.multiplyScalar(5)),
      up: camera.up.clone(),
      fov: camera.fov,
    };
    wasTargetView.current = true;
  }, [viewIndex, controls, camera]);

  // Default priority (0) runs after drei's OrbitControls (-1), so this pose wins every frame.
  useFrame(() => {
    if (viewIndex === null) return;
    const c = cameras[viewIndex];
    if (!c) return;
    const aspect = size.width / Math.max(1, size.height);
    const exactFov = viewerFovDeg(c.fov_y_deg, aspect);
    const pos = new Vector3(...c.position);
    const target = new Vector3(...c.look_at);
    const up = new Vector3(...c.up);
    if (up.lengthSq() < 1e-12) up.set(0, 1, 0);
    up.normalize();
    let fov = exactFov;

    const a = anim.current;
    if (a) {
      const t = Math.min(1, (performance.now() - a.start) / ANIM_MS);
      if (t < 1) {
        const e = ease(t);
        pos.lerpVectors(a.pos, pos, e);
        target.lerpVectors(a.target, target, e);
        const u = new Vector3().lerpVectors(a.up, up, e);
        if (u.lengthSq() > 1e-8) up.copy(u.normalize());
        fov = a.fov + (exactFov - a.fov) * e;
      } else {
        anim.current = null; // fall through with the exact values
      }
    }

    camera.position.copy(pos);
    camera.up.copy(up);
    camera.lookAt(target);
    camera.fov = fov;
    camera.near = c.near;
    camera.far = c.far;
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
    if (controls) controls.target.copy(target);
  });

  return null;
}
