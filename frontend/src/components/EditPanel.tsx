// Floating "Edit assembly" panel (Results mode): selection, lock / unlock, delete,
// move / rotate gizmo, and "Rerun with locked pieces".
import { useState } from "react";
import type { ObjectInstance } from "../api/types";
import type { GizmoMode } from "../viewer/PieceGizmo";
import { ErrorText, Segmented, Spinner } from "./common";

export type GizmoChoice = "off" | GizmoMode;

interface EditPanelProps {
  methodLabel: string;
  objects: ObjectInstance[];
  selection: number[];
  edited: boolean;
  /** Editing allowed at all (finished result). */
  canEdit: boolean;
  /** Why picking is off (e.g. Target View); null when picking works. */
  pickHint: string | null;
  gizmo: GizmoChoice;
  onGizmo: (g: GizmoChoice) => void;
  onLockSelected: (locked: boolean) => void;
  onDeleteSelected: () => void;
  onClearSelection: () => void;
  onLockAll: (locked: boolean) => void;
  onReset: () => void;
  keepUnlocked: boolean;
  onKeepUnlocked: (v: boolean) => void;
  applyCurrent: boolean;
  onApplyCurrent: (v: boolean) => void;
  rerunBlockers: string[];
  onRerun: () => void;
  rerunning: boolean;
  rerunError: string | null;
}

function nameSummary(objects: ObjectInstance[], selection: number[]): string {
  const counts = new Map<string, number>();
  for (const i of selection) {
    const n = objects[i]?.mesh_name ?? "?";
    counts.set(n, (counts.get(n) ?? 0) + 1);
  }
  return [...counts.entries()].map(([n, c]) => (c > 1 ? `${n} ×${c}` : n)).join(", ");
}

export function EditPanel(p: EditPanelProps) {
  const [open, setOpen] = useState(true);
  const lockedCount = p.objects.reduce((n, o) => n + (o.locked ? 1 : 0), 0);
  const sel = p.selection.filter((i) => i >= 0 && i < p.objects.length);
  const selLocked = sel.filter((i) => p.objects[i].locked).length;
  const single = sel.length === 1 ? p.objects[sel[0]] : null;
  const canMove = !!single && !single.locked && !p.pickHint;

  if (!open) {
    return (
      <button type="button" className="edit-panel-collapsed" onClick={() => setOpen(true)} title="Edit assembly / rerun with locked pieces">
        Edit ▸ {p.edited && <span className="badge badge-edited">edited</span>}
        {lockedCount > 0 && <span className="muted small"> · {lockedCount} locked</span>}
      </button>
    );
  }

  return (
    <div className="edit-panel" onPointerDown={(e) => e.stopPropagation()}>
      <div className="edit-header">
        <strong>Edit · {p.methodLabel}</strong>
        {p.edited && <span className="badge badge-edited" title="Local edits (not saved; lost on reload)">edited</span>}
        <span className="spacer" />
        <button type="button" className="small-btn" onClick={() => setOpen(false)} title="Collapse">
          ◂
        </button>
      </div>
      <div className="muted small">
        {p.objects.length} pieces · {lockedCount} locked
      </div>

      {!p.canEdit ? (
        <div className="hint small">Editing becomes available when this method has finished.</div>
      ) : (
        <>
          <div className="hint small">
            {p.pickHint ?? "Click a piece to select · Shift+click to add / remove · click empty space to clear."}
          </div>

          <div className="edit-selection">
            <div>
              <strong>{sel.length}</strong> selected
              {sel.length > 0 && (
                <span className="muted" title={nameSummary(p.objects, sel)}>
                  {" "}· {nameSummary(p.objects, sel)}
                  {selLocked > 0 && ` · ${selLocked} locked`}
                </span>
              )}
            </div>
            <div className="edit-buttons">
              <button type="button" className="small-btn" disabled={sel.length === 0 || selLocked === sel.length} onClick={() => p.onLockSelected(true)}>
                Lock
              </button>
              <button type="button" className="small-btn" disabled={selLocked === 0} onClick={() => p.onLockSelected(false)}>
                Unlock
              </button>
              <button type="button" className="small-btn danger-text" disabled={sel.length === 0} onClick={p.onDeleteSelected}>
                Delete
              </button>
              <button type="button" className="small-btn" disabled={sel.length === 0} onClick={p.onClearSelection}>
                Clear
              </button>
            </div>
            <div className="edit-buttons" title={single?.locked ? "Unlock the piece to move it" : sel.length > 1 ? "Select a single piece to move it" : undefined}>
              <span className="muted small">Gizmo</span>
              <Segmented<GizmoChoice>
                value={canMove ? p.gizmo : "off"}
                onChange={p.onGizmo}
                options={[
                  { value: "off", label: "Off" },
                  { value: "translate", label: "Move", disabled: !canMove },
                  { value: "rotate", label: "Rotate", disabled: !canMove },
                ]}
              />
            </div>
          </div>

          <div className="edit-buttons">
            <button type="button" className="small-btn" disabled={lockedCount === p.objects.length} onClick={() => p.onLockAll(true)}>
              Lock all
            </button>
            <button type="button" className="small-btn" disabled={lockedCount === 0} onClick={() => p.onLockAll(false)}>
              Unlock all
            </button>
            <button type="button" className="small-btn" disabled={!p.edited} onClick={p.onReset}>
              Reset edits
            </button>
          </div>
        </>
      )}

      <div className="edit-rerun">
        <label className="checkbox small" title="keep_unlocked: unlocked pieces are the warm start (they may move or be replaced); otherwise only the locked pieces are kept">
          <input type="checkbox" checked={p.keepUnlocked} onChange={(e) => p.onKeepUnlocked(e.target.checked)} />
          Keep unlocked pieces as starting point
        </label>
        <label className="checkbox small" title="Merge the current Design options / bounds (and Custom JSON) as for a new run; otherwise reuse this job's own overrides">
          <input type="checkbox" checked={p.applyCurrent} onChange={(e) => p.onApplyCurrent(e.target.checked)} />
          Use current design options &amp; bounds
        </label>
        <button
          type="button"
          className="primary"
          disabled={p.rerunBlockers.length > 0 || p.rerunning}
          onClick={p.onRerun}
          title="Same models, targets, cameras, methods and preset as this job, optimizing around the locked pieces"
        >
          {p.rerunning ? <Spinner label="Submitting…" /> : `Rerun with ${lockedCount} locked piece${lockedCount === 1 ? "" : "s"}`}
        </button>
        {p.rerunBlockers.length > 0 && (
          <ul className="blockers">
            {p.rerunBlockers.map((b) => (
              <li key={b}>{b}</li>
            ))}
          </ul>
        )}
        <ErrorText>{p.rerunError}</ErrorText>
      </div>
    </div>
  );
}
