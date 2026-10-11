// DOM overlay drawn on top of the canvas in target view: the centered square of
// side min(width, height) is exactly the square optimization image (docs/api.md).
import type { CSSProperties } from "react";

export type OverlayBlend = "normal" | "multiply" | "difference";

interface TargetOverlayProps {
  width: number;
  height: number;
  imageUrl: string | null;
  showImage: boolean;
  opacity: number;
  blend: OverlayBlend;
}

export function TargetOverlay({ width, height, imageUrl, showImage, opacity, blend }: TargetOverlayProps) {
  const side = Math.min(width, height);
  if (side <= 0) return null;
  const style: CSSProperties = {
    width: side,
    height: side,
    left: (width - side) / 2,
    top: (height - side) / 2,
  };
  return (
    <div className="target-square" style={style}>
      {showImage && imageUrl && (
        <img
          src={imageUrl}
          alt="target overlay"
          style={{ opacity, mixBlendMode: blend, objectFit: "contain" }}
          draggable={false}
        />
      )}
    </div>
  );
}
