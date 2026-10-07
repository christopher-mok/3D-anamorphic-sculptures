// Renders an Assembly: one GLB load per model, instanced per mesh part.
import { Suspense, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { useGLTF } from "@react-three/drei";
import {
  Box3, BufferGeometry, Color, Float32BufferAttribute, IcosahedronGeometry, InstancedMesh,
  Material, Matrix4, Mesh, MeshBasicMaterial, MeshStandardMaterial, Vector3,
} from "three";
import type { Assembly, ObjectInstance } from "../api/types";
import { ErrorBoundary } from "../utils/ErrorBoundary";
import { colorForMesh, instanceMatrix } from "./math";

interface AssemblyMeshesProps {
  assembly: Assembly;
  /** Resolves a model name to its GLB url. */
  meshUrlFor: (name: string) => string;
  silhouette: boolean;
  showBoxes: boolean;
}

export function AssemblyMeshes({ assembly, meshUrlFor, silhouette, showBoxes }: AssemblyMeshesProps) {
  // Group instances by model so every GLB is loaded exactly once.
  const groups = useMemo(() => {
    const map = new Map<string, ObjectInstance[]>();
    for (const o of assembly.objects ?? []) {
      const key = o.mesh_name ?? String(o.mesh_id);
      const list = map.get(key);
      if (list) list.push(o);
      else map.set(key, [o]);
    }
    return [...map.entries()];
  }, [assembly]);

  return (
    <group>
      {groups.map(([name, instances]) => {
        const url = meshUrlFor(name);
        const color = colorForMesh(name);
        const fallback = (
          <PlaceholderInstances instances={instances} silhouette={silhouette} showBoxes={showBoxes} />
        );
        return (
          <ErrorBoundary key={name} resetKey={url} fallback={fallback} label={`mesh ${name}`}>
            <Suspense fallback={fallback}>
              <ModelInstances
                url={url}
                instances={instances}
                color={color}
                silhouette={silhouette}
                showBoxes={showBoxes}
              />
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

function ModelInstances({
  url, instances, color, silhouette, showBoxes,
}: { url: string; instances: ObjectInstance[]; color: Color; silhouette: boolean; showBoxes: boolean }) {
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

  const material = useInstanceMaterial(color, silhouette);
  const matrices = useMemo(() => instances.map(instanceMatrix), [instances]);

  return (
    <>
      {parts.map((p, i) => (
        <PartInstances key={i} geometry={p.geometry} partMatrix={p.matrix} matrices={matrices} material={material} />
      ))}
      {showBoxes && <InstanceBoxes box={localBox} matrices={matrices} color={silhouette ? "#e03131" : color} />}
    </>
  );
}

function PartInstances({
  geometry, partMatrix, matrices, material,
}: { geometry: BufferGeometry; partMatrix: Matrix4; matrices: Matrix4[]; material: Material }) {
  const ref = useRef<InstancedMesh>(null);
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const tmp = new Matrix4();
    matrices.forEach((m, i) => mesh.setMatrixAt(i, tmp.multiplyMatrices(m, partMatrix)));
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();
    // Re-run when R3F re-creates the InstancedMesh (args: geometry / material / count change).
  }, [matrices, partMatrix, geometry, material]);

  // `args` changes (count / material) re-create the InstancedMesh; geometry is owned by the GLTF cache.
  return (
    <instancedMesh
      ref={ref}
      args={[geometry, material, Math.max(1, matrices.length)]}
      count={matrices.length}
      frustumCulled={false}
      dispose={null}
    />
  );
}

/** Oriented bounding box of each instance, merged into one LineSegments. */
function InstanceBoxes({ box, matrices, color }: { box: Box3; matrices: Matrix4[]; color: Color | string }) {
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
      <lineBasicMaterial color={color} transparent opacity={0.8} />
    </lineSegments>
  );
}

/** Shown while a GLB loads or if it fails: the radius-1 bounding sphere of each instance. */
const placeholderGeometry = new IcosahedronGeometry(1, 1);

function PlaceholderInstances({
  instances, silhouette, showBoxes,
}: { instances: ObjectInstance[]; silhouette: boolean; showBoxes: boolean }) {
  const material = useInstanceMaterial(useMemo(() => new Color("#868e96"), []), silhouette, !silhouette);
  const matrices = useMemo(() => instances.map(instanceMatrix), [instances]);
  const identity = useMemo(() => new Matrix4(), []);
  const box = useMemo(() => new Box3(new Vector3(-1, -1, -1), new Vector3(1, 1, 1)), []);
  return (
    <>
      <PartInstances geometry={placeholderGeometry} partMatrix={identity} matrices={matrices} material={material} />
      {showBoxes && <InstanceBoxes box={box} matrices={matrices} color="#868e96" />}
    </>
  );
}
