// Main 3D viewer: R3F canvas + DOM overlays.
import { Suspense, useRef, type ReactNode } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { Assembly, BoundingVolume as BV, Camera, Vec3 } from "../api/types";
import { ErrorBoundary } from "../utils/ErrorBoundary";
import { useElementSize } from "../utils/hooks";
import { AssemblyMeshes } from "./AssemblyMeshes";
import { BoundingVolume } from "./BoundingVolume";
import { CameraFrustum } from "./CameraFrustum";
import { HullMesh, type HullStyle } from "./HullMesh";
import { cameraColor } from "./math";
import { TargetOverlay, type OverlayBlend } from "./TargetOverlay";
import { FREE_CAMERA, ViewerCameraController } from "./ViewerCameraController";

/** Everything the scene shows; built by App from either the setup form or a job. */
export interface SceneData {
  cameras: Camera[];
  /** Target image per camera (for frustum planes and the overlay). */
  targetImageUrls: (string | null)[];
  boundingVolume: BV;
  hullUrl: string | null;
  assembly: Assembly | null;
  meshUrlFor: (name: string) => string;
  /** Cameras can be selected / dragged (setup mode). */
  editable: boolean;
}

export interface ViewerSettings {
  showBounds: boolean;
  showHull: boolean;
  hullStyle: HullStyle;
  showFrusta: boolean;
  showBoxes: boolean;
  showTargetPlanes: boolean;
  showOverlay: boolean;
  overlayOpacity: number;
  overlayBlend: OverlayBlend;
  silhouette: boolean;
}

export const DEFAULT_VIEWER_SETTINGS: ViewerSettings = {
  showBounds: true,
  showHull: false,
  hullStyle: "wireframe",
  showFrusta: true,
  showBoxes: false,
  showTargetPlanes: true,
  showOverlay: true,
  overlayOpacity: 0.4,
  overlayBlend: "normal",
  silhouette: false,
};

interface SculptureViewerProps {
  scene: SceneData;
  settings: ViewerSettings;
  /** null = free orbit, otherwise index of the optimization camera to match. */
  viewIndex: number | null;
  selectedCamera: number | null;
  onSelectCamera: (i: number | null) => void;
  onMoveCamera: (i: number, position: Vec3) => void;
  /** Extra DOM content drawn on top (status text etc.). */
  hud?: ReactNode;
}

export function SculptureViewer({
  scene, settings, viewIndex, selectedCamera, onSelectCamera, onMoveCamera, hud,
}: SculptureViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const size = useElementSize(containerRef);
  const s = settings;
  const bv = scene.boundingVolume;
  const floorY = Math.min(bv.min[1], bv.max[1]) - 0.001;
  const span = Math.max(...bv.max.map((x, i) => Math.abs(x - bv.min[i])), 1);

  const overlayUrl = viewIndex !== null ? scene.targetImageUrls[viewIndex] ?? null : null;

  return (
    <div className={`viewer ${s.silhouette ? "viewer-light" : ""}`} ref={containerRef}>
      <Canvas
        camera={{ position: [span * 3.5, span * 2.5, span * 4.5], ...FREE_CAMERA }}
        gl={{ antialias: true, preserveDrawingBuffer: true }}
        dpr={[1, 2]}
      >
        <color attach="background" args={[s.silhouette ? "#ffffff" : "#14171c"]} />
        <hemisphereLight args={["#e8eeff", "#3b3530", 1.4]} />
        <ambientLight intensity={0.15} />
        <directionalLight position={[5, 8, 6]} intensity={2.2} />
        <directionalLight position={[-6, -2, -4]} intensity={0.6} />

        <OrbitControls makeDefault enableDamping dampingFactor={0.12} />
        <ViewerCameraController viewIndex={viewIndex} cameras={scene.cameras} />

        {!s.silhouette && <gridHelper args={[span * 6, 24, "#3a3f47", "#262a31"]} position={[0, floorY, 0]} />}
        {s.showBounds && <BoundingVolume volume={bv} color={s.silhouette ? "#adb5bd" : "#5c7cfa"} />}

        {scene.cameras.map((cam, i) => (
          <CameraFrustum
            key={i}
            camera={cam}
            index={i}
            color={cameraColor(i)}
            // The active target-view camera's own frustum would surround the eye; hide it.
            showFrustum={s.showFrusta && i !== viewIndex}
            imageUrl={scene.targetImageUrls[i] ?? null}
            showImage={s.showTargetPlanes && !s.silhouette}
            editable={scene.editable}
            selected={scene.editable && selectedCamera === i}
            onSelect={() => onSelectCamera(selectedCamera === i ? null : i)}
            onMove={(p) => onMoveCamera(i, p)}
          />
        ))}

        {s.showHull && scene.hullUrl && !s.silhouette && (
          <ErrorBoundary resetKey={scene.hullUrl} label="hull">
            <Suspense fallback={null}>
              <HullMesh url={scene.hullUrl} style={s.hullStyle} />
            </Suspense>
          </ErrorBoundary>
        )}

        {scene.assembly && (
          <AssemblyMeshes
            assembly={scene.assembly}
            meshUrlFor={scene.meshUrlFor}
            silhouette={s.silhouette}
            showBoxes={s.showBoxes}
          />
        )}
      </Canvas>

      {viewIndex !== null && (
        <TargetOverlay
          width={size.width}
          height={size.height}
          imageUrl={overlayUrl}
          showImage={s.showOverlay}
          opacity={s.overlayOpacity}
          blend={s.overlayBlend}
        />
      )}
      {hud && <div className="viewer-hud">{hud}</div>}
    </div>
  );
}
