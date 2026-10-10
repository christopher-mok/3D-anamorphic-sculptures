// Draws an optimization camera: frustum lines (square, aspect = 1), an "up"
// marker, a clickable body and (optionally) the target image on the far end.
import { Suspense, useMemo, useState } from "react";
import { Html, Line, TransformControls, useTexture } from "@react-three/drei";
import { DoubleSide, MathUtils, SRGBColorSpace, Vector3, type Mesh, type Texture } from "three";
import type { Camera, Vec3 } from "../api/types";
import { ErrorBoundary } from "../utils/ErrorBoundary";
import { cameraPose } from "./math";

interface CameraFrustumProps {
  camera: Camera;
  index: number;
  color: string;
  showFrustum: boolean;
  /** Target image drawn on the far end of the (display-truncated) frustum. */
  imageUrl?: string | null;
  showImage: boolean;
  imageOpacity?: number;
  /** Editing support (setup mode). */
  selected?: boolean;
  editable?: boolean;
  onSelect?: () => void;
  onMove?: (position: Vec3) => void;
  /** Viewing-zone radius (0 = none): ring around the eye, perpendicular to the view direction. */
  zoneRadius?: number;
  showZone?: boolean;
}

const RING_SEGMENTS = 72;

export function CameraFrustum({
  camera, index, color, showFrustum, imageUrl, showImage, imageOpacity = 0.85,
  selected = false, editable = false, onSelect, onMove, zoneRadius = 0, showZone = true,
}: CameraFrustumProps) {
  const pose = useMemo(() => cameraPose(camera), [camera]);

  // The real far plane is usually far behind the sculpture; truncate the drawn
  // frustum a bit behind the look-at point so the scene stays readable.
  const dist = pose.position.distanceTo(pose.target);
  const depth = Math.max(0.05, Math.min(camera.far > 0 ? camera.far : Infinity, Math.max(dist, 0.5) * 1.6));
  const half = depth * Math.tan(MathUtils.degToRad(camera.fov_y_deg || 1) / 2);

  const { segments, upMarker } = useMemo(() => {
    const c = (x: number, y: number) => new Vector3(x * half, y * half, -depth);
    const tl = c(-1, 1), tr = c(1, 1), br = c(1, -1), bl = c(-1, -1);
    const o = new Vector3();
    const segments = [o, tl, o, tr, o, br, o, bl, tl, tr, tr, br, br, bl, bl, tl];
    const upMarker = [tl.clone().lerp(tr, 0.35), new Vector3(0, half * 1.25, -depth), tl.clone().lerp(tr, 0.65)];
    return { segments, upMarker };
  }, [half, depth]);

  // Camera-local XY plane = perpendicular to the viewing direction (camera looks down -Z).
  const ring = useMemo(
    () =>
      zoneRadius > 0
        ? Array.from({ length: RING_SEGMENTS + 1 }, (_, k) => {
            const a = (k / RING_SEGMENTS) * Math.PI * 2;
            return new Vector3(Math.cos(a) * zoneRadius, Math.sin(a) * zoneRadius, 0);
          })
        : null,
    [zoneRadius],
  );

  const [body, setBody] = useState<Mesh | null>(null);
  const bodySize = Math.max(0.06, dist * 0.03);

  return (
    <>
      <group position={pose.position} quaternion={pose.quaternion}>
        {ring && showZone && <Line points={ring} color={color} lineWidth={2} />}
        {showFrustum && (
          <>
            <Line points={segments} segments color={color} lineWidth={selected ? 2.5 : 1.5} />
            <Line points={upMarker} color={color} lineWidth={1.5} />
            <Html position={[0, half * 1.35, -depth]} center style={{ pointerEvents: "none" }}>
              <div className="frustum-label" style={{ borderColor: color, color }}>
                Cam {index + 1}
              </div>
            </Html>
          </>
        )}
        {showImage && imageUrl && (
          <ErrorBoundary resetKey={imageUrl} label="target image">
            <Suspense fallback={null}>
              <TargetPlane url={imageUrl} size={2 * half} depth={depth * 0.999} opacity={imageOpacity} />
            </Suspense>
          </ErrorBoundary>
        )}
      </group>

      {/* Camera body at the true position (clickable / draggable in setup mode). */}
      {showFrustum && (
        <mesh
          ref={setBody}
          position={pose.position}
          quaternion={pose.quaternion}
          onClick={(e) => {
            if (!editable) return;
            e.stopPropagation();
            onSelect?.();
          }}
        >
          <boxGeometry args={[bodySize, bodySize, bodySize * 1.4]} />
          <meshBasicMaterial color={color} wireframe={!selected} />
        </mesh>
      )}
      {editable && selected && body && (
        <TransformControls
          object={body}
          mode="translate"
          size={0.8}
          onObjectChange={() => onMove?.(body.position.toArray() as Vec3)}
        />
      )}
    </>
  );
}

function TargetPlane({ url, size, depth, opacity }: { url: string; size: number; depth: number; opacity: number }) {
  const texture = useTexture(url) as Texture;
  texture.colorSpace = SRGBColorSpace;
  // PlaneGeometry faces +Z, i.e. back toward the camera apex; its +X/+Y are camera right/up.
  return (
    <mesh position={[0, 0, -depth]}>
      <planeGeometry args={[size, size]} />
      <meshBasicMaterial map={texture} transparent opacity={opacity} side={DoubleSide} toneMapped={false} depthWrite={false} />
    </mesh>
  );
}
