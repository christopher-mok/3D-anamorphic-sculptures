// Renders an Assembly: one GLB load per model, instanced per mesh part.
// Optional editing support: click picking (instanceId -> object index), selection
// highlight (tint + yellow boxes) and a lock cue (blue tint + wireframe shell).
import { Suspense, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { useGLTF } from "@react-three/drei";
import type { ThreeEvent } from "@react-three/fiber";
import {
  Box3, BufferGeometry, Color, Float32BufferAttribute, IcosahedronGeometry, InstancedMesh,
  Material, Matrix4, Mesh, MeshBasicMaterial, MeshStandardMaterial, Vector3,
} from "three";
import type { Assembly, ObjectInstance } from "../api/types";
import { ErrorBoundary } from "../utils/ErrorBoundary";
import { colorForMesh, instanceMatrix } from "./math";

export const SELECTED_COLOR = "#ffd43b";
export const LOCKED_COLOR = "#4dabf7";

interface AssemblyMeshesProps {
  assembly: Assembly;
  /** Resolves a model name to its GLB url. */
  meshUrlFor: (name: string) => string;
  silhouette: boolean;
  showBoxes: boolean;
  /** Indices into `assembly.objects` drawn highlighted. */
  selected?: ReadonlySet<number>;
  /** When set, clicking a piece calls this with its index (additive = Shift held). */
  onPick?: (index: number, additive: boolean) => void;
}

interface InstanceGroup {
  name: string;
  instances: ObjectInstance[];
  /** Index in assembly.objects of each instance. */
  indices: number[];
}

export function AssemblyMeshes({ assembly, meshUrlFor, silhouette, showBoxes, selected, onPick }: AssemblyMeshesProps) {
  // Group instances by model so every GLB is loaded exactly once.
  const groups = useMemo(() => {
    const map = new Map<string, InstanceGroup>();
    (assembly.objects ?? []).forEach((o, index) => {
      const key = o.mesh_name ?? String(o.mesh_id);
      const g = map.get(key);
      if (g) {
        g.instances.push(o);
        g.indices.push(index);
      } else map.set(key, { name: key, instances: [o], indices: [index] });
    });
    return [...map.values()];
  }, [assembly]);

  return (
    <group>
      {groups.map((g) => {
        const url = meshUrlFor(g.name);
        const common = { group: g, silhouette, showBoxes, selected, onPick };
        const fallback = <PlaceholderInstances {...common} />;
        return (
          <ErrorBoundary key={g.name} resetKey={url} fallback={fallback} label={`mesh ${g.name}`}>
            <Suspense fallback={fallback}>
              <ModelInstances url={url} color={colorForMesh(g.name)} {...common} />
            </Suspense>
          </ErrorBoundary>
        );
      })}
    </group>
  );
}

// ---------------------------------------------------------------------------

interface Part {
  geometry: BufferGeometry;
  /** Transform of the mesh node inside the GLB scene (model space). */
  matrix: Matrix4;
}

interface InstancesCommon {
  group: InstanceGroup;
  silhouette: boolean;
  showBoxes: boolean;
  selected?: ReadonlySet<number>;
  onPick?: (index: number, additive: boolean) => void;
}

const WHITE = new Color("#ffffff");

function useInstanceMaterial(color: Color, silhouette: boolean, wireframe = false): Material {
  const material = useMemo<Material>(
    () =>
      silhouette
        ? new MeshBasicMaterial({ color: 0x000000 })
        : new MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.05, wireframe }),
    [color, silhouette, wireframe],
  );
  useEffect(() => () => material.dispose(), [material]);
  return material;
}

function ModelInstances({ url, color, ...common }: InstancesCommon & { url: string; color: Color }) {
  const gltf = useGLTF(url);

  // Collect every mesh primitive with its node transform; compute the model-space bbox.
  const { parts, localBox } = useMemo(() => {
    gltf.scene.updateMatrixWorld(true);
    const parts: Part[] = [];
    const localBox = new Box3();
    gltf.scene.traverse((o) => {
      const mesh = o as Mesh;
      if (!mesh.isMesh || !mesh.geometry) return;
      const geometry = mesh.geometry;
      if (!geometry.attributes.normal) geometry.computeVertexNormals();
      if (!geometry.boundingBox) geometry.computeBoundingBox();
      const matrix = mesh.matrixWorld.clone();
      if (geometry.boundingBox) localBox.union(geometry.boundingBox.clone().applyMatrix4(matrix));
      parts.push({ geometry, matrix });
    });
    if (localBox.isEmpty()) localBox.set(new Vector3(-1, -1, -1), new Vector3(1, 1, 1));
    return { parts, localBox };
  }, [gltf]);

  return <InstancesView {...common} parts={parts} localBox={localBox} baseColor={color} wireframe={false} />;
}

/** Shown while a GLB loads or if it fails: the radius-1 bounding sphere of each instance. */
const placeholderGeometry = new IcosahedronGeometry(1, 1);
const PLACEHOLDER_PARTS: Part[] = [{ geometry: placeholderGeometry, matrix: new Matrix4() }];
const PLACEHOLDER_BOX = new Box3(new Vector3(-1, -1, -1), new Vector3(1, 1, 1));
const PLACEHOLDER_COLOR = new Color("#868e96");

function PlaceholderInstances(common: InstancesCommon) {
  return (
    <InstancesView
      {...common}
      parts={PLACEHOLDER_PARTS}
      localBox={PLACEHOLDER_BOX}
      baseColor={PLACEHOLDER_COLOR}
      wireframe={!common.silhouette}
    />
  );
}

const LOCK_SHELL = new Matrix4().makeScale(1.02, 1.02, 1.02);

function InstancesView({
  group, silhouette, showBoxes, selected, onPick, parts, localBox, baseColor, wireframe,
}: InstancesCommon & { parts: Part[]; localBox: Box3; baseColor: Color; wireframe: boolean }) {
  // White material; the per-instance colour carries the type colour and the edit cues.
  const material = useInstanceMaterial(WHITE, silhouette, wireframe);
  const lockMaterial = useMemo(
    () => new MeshBasicMaterial({ color: LOCKED_COLOR, wireframe: true, transparent: true, opacity: 0.85, depthWrite: false }),
    [],
  );
  useEffect(() => () => lockMaterial.dispose(), [lockMaterial]);

  const { instances, indices } = group;
  const matrices = useMemo(() => instances.map(instanceMatrix), [instances]);
  const isSelected = useMemo(() => indices.map((i) => !!selected?.has(i)), [indices, selected]);

  const colors = useMemo(() => {
    const lock = new Color(LOCKED_COLOR);
    const sel = new Color(SELECTED_COLOR);
    return instances.map((o, k) => {
      const c = baseColor.clone();
      if (o.locked) c.lerp(lock, 0.55);
      if (isSelected[k]) c.lerp(sel, 0.7);
      return c;
    });
  }, [instances, isSelected, baseColor]);

  const lockedMatrices = useMemo(
    () => (silhouette ? [] : matrices.filter((_, k) => instances[k].locked).map((m) => m.clone().multiply(LOCK_SHELL))),
    [matrices, instances, silhouette],
  );
  const selectedMatrices = useMemo(() => matrices.filter((_, k) => isSelected[k]), [matrices, isSelected]);

  const onClick = useMemo(
    () =>
      onPick
        ? (e: ThreeEvent<MouseEvent>) => {
            // Ignore the click that ends an orbit drag.
            if (e.delta > 4 || e.instanceId === undefined) return;
            const index = indices[e.instanceId];
            if (index === undefined) return;
            e.stopPropagation();
            onPick(index, e.nativeEvent.shiftKey);
          }
        : undefined,
    [onPick, indices],
  );

  return (
    <>
      {parts.map((p, i) => (
        <PartInstances
          key={i}
          geometry={p.geometry}
          partMatrix={p.matrix}
          matrices={matrices}
          material={material}
          colors={colors}
          onClick={onClick}
        />
      ))}
      {lockedMatrices.length > 0 &&
        parts.map((p, i) => (
          <PartInstances
            key={`lock-${i}`}
            geometry={p.geometry}
            partMatrix={p.matrix}
            matrices={lockedMatrices}
            material={lockMaterial}
          />
        ))}
      {showBoxes && <InstanceBoxes box={localBox} matrices={matrices} color={silhouette ? "#e03131" : baseColor} />}
      {selectedMatrices.length > 0 && <InstanceBoxes box={localBox} matrices={selectedMatrices} color={SELECTED_COLOR} opacity={1} />}
    </>
  );
}

function PartInstances({
  geometry, partMatrix, matrices, material, colors, onClick,
}: {
  geometry: BufferGeometry;
  partMatrix: Matrix4;
  matrices: Matrix4[];
  material: Material;
  colors?: Color[];
  onClick?: (e: ThreeEvent<MouseEvent>) => void;
}) {
  const ref = useRef<InstancedMesh>(null);
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const tmp = new Matrix4();
    matrices.forEach((m, i) => mesh.setMatrixAt(i, tmp.multiplyMatrices(m, partMatrix)));
    mesh.instanceMatrix.needsUpdate = true;
    if (colors) {
      colors.forEach((c, i) => mesh.setColorAt(i, c));
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
    mesh.computeBoundingSphere();
    mesh.computeBoundingBox();
    // Re-run when R3F re-creates the InstancedMesh (args: geometry / material / count change).
  }, [matrices, partMatrix, geometry, material, colors]);

  // `args` changes (count / material) re-create the InstancedMesh; geometry is owned by the GLTF cache.
  return (
    <instancedMesh
      ref={ref}
      args={[geometry, material, Math.max(1, matrices.length)]}
      count={matrices.length}
      frustumCulled={false}
      dispose={null}
      onClick={onClick}
    />
  );
}

/** Oriented bounding box of each instance, merged into one LineSegments. */
function InstanceBoxes({
  box, matrices, color, opacity = 0.8,
}: { box: Box3; matrices: Matrix4[]; color: Color | string; opacity?: number }) {
  const geometry = useMemo(() => {
    const { min, max } = box;
    const corners = [0, 1, 2, 3, 4, 5, 6, 7].map(
      (i) => new Vector3(i & 1 ? max.x : min.x, i & 2 ? max.y : min.y, i & 4 ? max.z : min.z),
    );
    const edges: [number, number][] = [
      [0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    const positions: number[] = [];
    const p = new Vector3();
    for (const m of matrices) {
      const world = corners.map((c) => p.copy(c).applyMatrix4(m).toArray());
      for (const [a, b] of edges) positions.push(...world[a], ...world[b]);
    }
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(positions, 3));
    return g;
  }, [box, matrices]);
  useEffect(() => () => geometry.dispose(), [geometry]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={color} transparent opacity={opacity} />
    </lineSegments>
  );
}
