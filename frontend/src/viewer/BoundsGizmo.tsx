// Draggable handles for editing the sculpture bounds in setup mode.
//
// Face / distance handles use a custom 1-D drag: on pointer down the pointer is
// captured on the canvas and OrbitControls is disabled; every pointer move casts a
// ray from the viewer camera and projects it onto the handle's axis line (closest
// point between the ray and the line). The whole box can also be translated with
// drei's TransformControls on a centre handle.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { Html, Line, TransformControls } from "@react-three/drei";
import { MathUtils, PerspectiveCamera, Raycaster, Vector2, Vector3, type Mesh } from "three";
import type { BoundingVolume as BV, Camera, Vec3 } from "../api/types";
import { MIN_BOX_SIZE, MIN_VIEW_AXIS_GAP, MIN_VIEW_AXIS_NEAR } from "../components/BoundsPanel";
import { BoundingVolume } from "./BoundingVolume";
import { cameraForward, closestLineParam, frustumSegmentCorners, viewAxisVolume } from "./math";

const AXIS_COLORS = ["#ff6b6b", "#51cf66", "#4dabf7"];
const ACTIVE_COLOR = "#ffffff";
const MAX_VIEW_AXIS_FAR = 1000;
const round3 = (x: number) => Math.round(x * 1e3) / 1e3;

// ---------------------------------------------------------------------------
// 1-D drag along a world-space line

interface AxisDragSpec {
  /** Line `origin + t * dir` (dir unit length) the handle slides on. */
  origin: Vector3;
  dir: Vector3;
  /** Current parameter of the handle on that line. */
  value: number;
  onValue: (t: number) => void;
  onEnd?: () => void;
}

/** Returns a pointer-down handler factory that drags a value along a line. */
function useAxisDrag() {
  const get = useThree((s) => s.get);
  const cleanupRef = useRef<(() => void) | null>(null);
  // Abort an in-progress drag if the gizmo unmounts.
  useEffect(() => () => cleanupRef.current?.(), []);

  return useCallback(
    (e: ThreeEvent<PointerEvent>, spec: AxisDragSpec) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      cleanupRef.current?.();
      const { gl, controls } = get();
      const el = gl.domElement;
      const orbit = controls as unknown as { enabled: boolean } | null;
      const wasEnabled = orbit?.enabled ?? false;
      if (orbit) orbit.enabled = false;
      try {
        el.setPointerCapture(e.pointerId);
      } catch {
        /* pointer already released */
      }
      // Grab offset so the handle does not jump to the projected pointer.
      const t0 = closestLineParam(spec.origin, spec.dir, e.ray.origin, e.ray.direction);
      const offset = t0 === null ? 0 : spec.value - t0;
      const raycaster = new Raycaster();
      const ndc = new Vector2();

      const move = (ev: PointerEvent) => {
        if (ev.pointerId !== e.pointerId) return;
        const rect = el.getBoundingClientRect();
        ndc.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
        raycaster.setFromCamera(ndc, get().camera);
        const t = closestLineParam(spec.origin, spec.dir, raycaster.ray.origin, raycaster.ray.direction);
        if (t !== null && Number.isFinite(t)) spec.onValue(t + offset);
      };
      const cleanup = () => {
        el.removeEventListener("pointermove", move);
        el.removeEventListener("pointerup", up);
        el.removeEventListener("pointercancel", up);
        try {
          if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
        } catch {
          /* ignore */
        }
        if (orbit) orbit.enabled = wasEnabled;
        cleanupRef.current = null;
      };
      const up = (ev: PointerEvent) => {
        if (ev.pointerId !== e.pointerId) return;
        cleanup();
        spec.onEnd?.();
      };
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", up);
      el.addEventListener("pointercancel", up);
      cleanupRef.current = cleanup;
    },
    [get],
  );
}

/** Sets the canvas cursor while hovering a handle. */
function useCursor() {
  const gl = useThree((s) => s.gl);
  return useCallback((c: string | null) => { gl.domElement.style.cursor = c ?? ""; }, [gl]);
}

// ---------------------------------------------------------------------------
// Handle mesh with a constant on-screen size

type HandleShape = "sphere" | "cube" | "octahedron";

interface HandleProps {
  position: Vector3 | Vec3;
  color: string;
  shape?: HandleShape;
  /** Radius as a fraction of the visible half-height at the handle's depth. */
  screenSize?: number;
  active?: boolean;
  onPointerDown?: (e: ThreeEvent<PointerEvent>) => void;
  onClick?: (e: ThreeEvent<MouseEvent>) => void;
  onHover?: (hovered: boolean) => void;
  meshRef?: (m: Mesh | null) => void;
}

function Handle({ position, color, shape = "sphere", screenSize = 0.028, active = false, onPointerDown, onClick, onHover, meshRef }: HandleProps) {
  const local = useRef<Mesh | null>(null);
  const [hovered, setHovered] = useState(false);
  const setCursor = useCursor();
  const tmp = useMemo(() => new Vector3(), []);
  // Do not leave a "grab" cursor behind when a hovered handle unmounts.
  useEffect(() => () => setCursor(null), [setCursor]);

  useFrame(({ camera }) => {
    const m = local.current;
    if (!m) return;
    m.getWorldPosition(tmp);
    const d = camera.position.distanceTo(tmp);
    const fov = camera instanceof PerspectiveCamera ? camera.fov : 40;
    const k = (active || hovered ? 1.3 : 1) * screenSize;
    m.scale.setScalar(Math.max(1e-4, d * Math.tan(MathUtils.degToRad(fov) / 2) * k));
  });

  const hover = (on: boolean) => {
    setHovered(on);
    onHover?.(on);
    setCursor(on ? (active ? "grabbing" : "grab") : null);
  };

  return (
    <mesh
      ref={(m) => {
        local.current = m;
        meshRef?.(m);
      }}
      position={position instanceof Vector3 ? position.toArray() : position}
      renderOrder={20}
      onPointerDown={onPointerDown}
      onClick={onClick}
      onPointerOver={(e) => {
        e.stopPropagation();
        hover(true);
      }}
      onPointerOut={() => hover(false)}
    >
      {shape === "sphere" && <sphereGeometry args={[1, 20, 14]} />}
      {shape === "cube" && <boxGeometry args={[1.5, 1.5, 1.5]} />}
      {shape === "octahedron" && <octahedronGeometry args={[1.3]} />}
      <meshBasicMaterial
        color={active ? ACTIVE_COLOR : color}
        depthTest={false}
        depthWrite={false}
        transparent
        opacity={active || hovered ? 1 : 0.85}
        toneMapped={false}
      />
    </mesh>
  );
}

// ---------------------------------------------------------------------------
// Box mode: six face handles + centre translate handle

interface BoxBoundsGizmoProps {
  volume: BV;
  onChange: (v: BV) => void;
}

export function BoxBoundsGizmo({ volume, onChange }: BoxBoundsGizmoProps) {
  const latest = useRef(volume);
  latest.current = volume;
  const beginDrag = useAxisDrag();
  const [activeFace, setActiveFace] = useState<string | null>(null);
  const [hoverFace, setHoverFace] = useState<string | null>(null);
  const [moveMode, setMoveMode] = useState(false);
  const [center, setCenter] = useState<Mesh | null>(null);

  const mid = volume.min.map((x, i) => (x + volume.max[i]) / 2) as Vec3;

  const faces = useMemo(() => {
    const mid = volume.min.map((x, i) => (x + volume.max[i]) / 2) as Vec3;
    const out: { key: string; axis: number; side: "min" | "max"; pos: Vec3 }[] = [];
    for (let axis = 0; axis < 3; axis++) {
      for (const side of ["min", "max"] as const) {
        const pos = [...mid] as Vec3;
        pos[axis] = volume[side][axis];
        out.push({ key: `${side}${axis}`, axis, side, pos });
      }
    }
    return out;
  }, [volume]);

  const highlightKey = activeFace ?? hoverFace;
  const highlight = useMemo(() => {
    const f = faces.find((x) => x.key === highlightKey);
    if (!f) return null;
    const [j, k] = [0, 1, 2].filter((i) => i !== f.axis);
    const corner = (a: number, b: number) => {
      const p = new Vector3();
      p.setComponent(f.axis, f.pos[f.axis]);
      p.setComponent(j, a ? volume.max[j] : volume.min[j]);
      p.setComponent(k, b ? volume.max[k] : volume.min[k]);
      return p;
    };
    return { points: [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1), corner(0, 0)], color: AXIS_COLORS[f.axis] };
  }, [faces, highlightKey, volume]);

  const startFaceDrag = (e: ThreeEvent<PointerEvent>, axis: number, side: "min" | "max", key: string, pos: Vec3) => {
    // The handle slides on the line through the face centre along the face normal;
    // with the axis component of the origin zeroed, the line parameter is the coordinate.
    const origin = new Vector3(...pos).setComponent(axis, 0);
    const dir = new Vector3().setComponent(axis, 1);
    setActiveFace(key);
    beginDrag(e, {
      origin,
      dir,
      value: volume[side][axis],
      onValue: (t) => {
        const v = latest.current;
        const next: BV = { min: [...v.min] as Vec3, max: [...v.max] as Vec3 };
        if (side === "min") next.min[axis] = round3(Math.min(t, v.max[axis] - MIN_BOX_SIZE));
        else next.max[axis] = round3(Math.max(t, v.min[axis] + MIN_BOX_SIZE));
        if (next[side][axis] !== v[side][axis]) onChange(next);
      },
      onEnd: () => setActiveFace(null),
    });
  };

  return (
    <>
      {highlight && <Line points={highlight.points} color={highlight.color} lineWidth={2.5} depthTest={false} renderOrder={19} />}
      {faces.map((f) => (
        <Handle
          key={f.key}
          position={f.pos}
          color={AXIS_COLORS[f.axis]}
          shape="cube"
          active={activeFace === f.key}
          onPointerDown={(e) => startFaceDrag(e, f.axis, f.side, f.key, f.pos)}
          onClick={(e) => e.stopPropagation()}
          onHover={(on) => setHoverFace((h) => (on ? f.key : h === f.key ? null : h))}
        />
      ))}
      <Handle
        position={mid}
        color="#e9ecef"
        shape="octahedron"
        screenSize={0.022}
        active={moveMode}
        meshRef={setCenter}
        onClick={(e) => {
          e.stopPropagation();
          if (e.delta <= 3) setMoveMode((m) => !m);
        }}
      />
      {moveMode && center && (
        <TransformControls
          object={center}
          mode="translate"
          size={0.7}
          onObjectChange={() => {
            const v = latest.current;
            const p = center.position;
            const half = v.max.map((x, i) => (x - v.min[i]) / 2);
            const c = [p.x, p.y, p.z];
            onChange({
              min: c.map((x, i) => round3(x - half[i])) as Vec3,
              max: c.map((x, i) => round3(x + half[i])) as Vec3,
            });
          }}
        />
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Unbounded single-view mode: frustum segment of camera 0 between near and far

interface ViewAxisBoundsProps {
  camera: Camera;
  near: number;
  far: number;
  color?: string;
  /** Also draw the axis-aligned box the backend derives from the segment. */
  showBox?: boolean;
  /** Draggable near / far handles (setup mode). */
  editable?: boolean;
  onChange?: (near: number, far: number) => void;
}

export function ViewAxisBounds({ camera, near, far, color = "#5c7cfa", showBox = true, editable = false, onChange }: ViewAxisBoundsProps) {
  const latest = useRef({ near, far });
  latest.current = { near, far };
  const beginDrag = useAxisDrag();
  const [active, setActive] = useState<"near" | "far" | null>(null);
  const valid = near > 0 && near < far;

  const geo = useMemo(() => {
    if (!valid) return null;
    const p = frustumSegmentCorners(camera, near, far);
    const segments = [
      p[0], p[1], p[1], p[2], p[2], p[3], p[3], p[0],
      p[4], p[5], p[5], p[6], p[6], p[7], p[7], p[4],
      p[0], p[4], p[1], p[5], p[2], p[6], p[3], p[7],
    ];
    const origin = new Vector3(...camera.position);
    const dir = cameraForward(camera);
    return {
      segments,
      origin,
      dir,
      nearPt: origin.clone().addScaledVector(dir, near),
      farPt: origin.clone().addScaledVector(dir, far),
      box: viewAxisVolume(camera, near, far),
    };
  }, [camera, near, far, valid]);

  if (!geo) return null;

  const startDrag = (e: ThreeEvent<PointerEvent>, which: "near" | "far") => {
    setActive(which);
    beginDrag(e, {
      origin: geo.origin,
      dir: geo.dir,
      value: which === "near" ? near : far,
      onValue: (t) => {
        const cur = latest.current;
        if (which === "near") {
          const n = round3(MathUtils.clamp(t, MIN_VIEW_AXIS_NEAR, cur.far - MIN_VIEW_AXIS_GAP));
          if (n !== cur.near) onChange?.(n, cur.far);
        } else {
          const f = round3(MathUtils.clamp(t, cur.near + MIN_VIEW_AXIS_GAP, MAX_VIEW_AXIS_FAR));
          if (f !== cur.far) onChange?.(cur.near, f);
        }
      },
      onEnd: () => setActive(null),
    });
  };

  return (
    <>
      <Line points={geo.segments} segments color={color} lineWidth={1.5} />
      {showBox && <BoundingVolume volume={geo.box} color={color} dashed lineWidth={1} />}
      {editable && (
        <>
          <Line points={[geo.nearPt, geo.farPt]} color="#e9ecef" lineWidth={1} dashed dashSize={0.1} gapSize={0.08} transparent opacity={0.7} />
          {(["near", "far"] as const).map((which) => {
            const pt = which === "near" ? geo.nearPt : geo.farPt;
            return (
              <group key={which}>
                <Handle
                  position={pt}
                  color={which === "near" ? "#ffd43b" : "#f783ac"}
                  active={active === which}
                  onPointerDown={(e) => startDrag(e, which)}
                  onClick={(e) => e.stopPropagation()}
                />
                <Html position={pt.toArray()} style={{ pointerEvents: "none" }} zIndexRange={[10, 0]}>
                  <div className="gizmo-label">
                    {which} {(which === "near" ? near : far).toFixed(2)}
                  </div>
                </Html>
              </group>
            );
          })}
        </>
      )}
    </>
  );
}
