// TransformControls on one assembly piece (Results-mode editing). The gizmo drives an
// empty proxy object placed at the piece's pose; every change is turned back into an
// ObjectInstance (translation / rotation_matrix / rotation6d, scale unchanged).
import { useMemo, useRef, useState, type MutableRefObject } from "react";
import { TransformControls } from "@react-three/drei";
import type { Group } from "three";
import type { ObjectInstance } from "../api/types";
import { instancePose, withPose } from "./math";

export type GizmoMode = "translate" | "rotate";

interface PieceGizmoProps {
  object: ObjectInstance;
  mode: GizmoMode;
  onChange: (next: ObjectInstance) => void;
  /** Set while the gizmo is being dragged (and briefly after), so a release is not a "click on empty space". */
  busyRef: MutableRefObject<boolean>;
}

export function PieceGizmo({ object, mode, onChange, busyRef }: PieceGizmoProps) {
  const [proxy, setProxy] = useState<Group | null>(null);
  const pose = useMemo(() => instancePose(object), [object]);
  const latest = useRef(object);
  latest.current = object;
  const releaseTimer = useRef<number | undefined>(undefined);

  return (
    <>
      <group ref={setProxy} position={pose.position} quaternion={pose.quaternion} />
      {proxy && (
        <TransformControls
          object={proxy}
          mode={mode}
          size={0.8}
          onMouseDown={() => {
            window.clearTimeout(releaseTimer.current);
            busyRef.current = true;
          }}
          onMouseUp={() => {
            releaseTimer.current = window.setTimeout(() => (busyRef.current = false), 150);
          }}
          onObjectChange={() => onChange(withPose(latest.current, proxy.position, proxy.quaternion))}
        />
      )}
    </>
  );
}
