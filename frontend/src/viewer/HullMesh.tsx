// Visual hull preview (OBJ at preprocessing.hull_url), assumed to be in world coordinates.
import { useEffect, useMemo } from "react";
import { useLoader } from "@react-three/fiber";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { DoubleSide, Mesh, MeshBasicMaterial, MeshStandardMaterial, type Material } from "three";

export type HullStyle = "wireframe" | "translucent";

export function HullMesh({ url, style }: { url: string; style: HullStyle }) {
  const obj = useLoader(OBJLoader, url);

  const material = useMemo<Material>(
    () =>
      style === "wireframe"
        ? new MeshBasicMaterial({ color: "#74c0fc", wireframe: true, transparent: true, opacity: 0.25, depthWrite: false })
        : new MeshStandardMaterial({
            color: "#74c0fc", transparent: true, opacity: 0.18, side: DoubleSide, depthWrite: false, roughness: 1,
          }),
    [style],
  );
  useEffect(() => () => material.dispose(), [material]);

  // Clone so the cached loader result is never mutated.
  const scene = useMemo(() => {
    const clone = obj.clone(true);
    clone.traverse((o) => {
      const m = o as Mesh;
      if (m.isMesh) {
        m.material = material;
        m.renderOrder = 10;
      }
    });
    return clone;
  }, [obj, material]);

  return <primitive object={scene} />;
}
