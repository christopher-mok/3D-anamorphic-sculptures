// Final comparison of the methods (shown once `comparison` is non-null).
import { trimSlash } from "../api/client";
import type { Comparison, JobStatus, MethodName } from "../api/types";
import { METHOD_LABELS } from "../api/types";
import { fmt, fmtInt, fmtPct, fmtTime } from "../utils/format";

interface ComparisonTableProps {
  job: JobStatus;
  comparison: Comparison;
  selectedMethod: MethodName | null;
  onSelect: (m: MethodName) => void;
}

/** Result directory of a method: from progress, else `<output_dir_url>/<method>`. */
function resultDir(job: JobStatus, m: MethodName): string | null {
  const dir = job.methods?.[m]?.result_dir_url;
  if (dir) return trimSlash(dir);
  return job.output_dir_url ? `${trimSlash(job.output_dir_url)}/${m}` : null;
}

export function ComparisonTable({ job, comparison, selectedMethod, onSelect }: ComparisonTableProps) {
  const rows = comparison.rows ?? [];
  const showView2 = rows.some((r) => (r.view_iou?.length ?? 0) > 1) || (job.request?.cameras?.length ?? 1) > 1;
  const out = job.output_dir_url ? trimSlash(job.output_dir_url) : null;

  return (
    <div className="comparison">
      <div className="comparison-header">
        <h2>Comparison</h2>
        <span className="muted small">
          selection metric: <span className="mono">{comparison.selection_metric || "—"}</span>
          {comparison.best_method && <> · best: <strong>{METHOD_LABELS[comparison.best_method]}</strong></>}
        </span>
        <span className="spacer" />
        {out && (
          <span className="small">
            <a href={`${out}/comparison.json`} target="_blank" rel="noreferrer">comparison.json</a>
            {" · "}
            <a href={`${out}/comparison.csv`} target="_blank" rel="noreferrer">comparison.csv</a>
          </span>
        )}
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>Mean IoU</th>
              <th>Min-view IoU</th>
              <th>View 1 IoU</th>
              {showView2 && <th>View 2 IoU</th>}
              <th>Spill</th>
              <th>Coverage</th>
              <th>Collisions</th>
              <th>Containment viol.</th>
              <th title="Piece-type randomness: normalized entropy of the per-type counts">Randomness</th>
              <th>Objects</th>
              <th>Runtime</th>
              <th>Downloads</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const best = r.method === comparison.best_method;
              const dir = resultDir(job, r.method);
              return (
                <tr
                  key={r.method}
                  className={`${best ? "best" : ""} ${selectedMethod === r.method ? "selected" : ""}`}
                  onClick={() => onSelect(r.method)}
                  title="Show this result in the 3D viewer"
                >
                  <td>
                    {best && <span className="star" title="best method">★ </span>}
                    {METHOD_LABELS[r.method] ?? r.method}
                  </td>
                  <td>{fmt(r.mean_iou)}</td>
                  <td>{fmt(r.min_view_iou)}</td>
                  <td>{fmt(r.view_iou?.[0])}</td>
                  {showView2 && <td>{fmt(r.view_iou?.[1])}</td>}
                  <td>{fmtPct(r.spill, 2)}</td>
                  <td>{fmtPct(r.coverage, 1)}</td>
                  <td>{fmtInt(r.collisions)}</td>
                  <td>{fmtInt(r.containment_violations)}</td>
                  <td title={r.type_counts ? Object.entries(r.type_counts).map(([k, v]) => `${k}: ${v}`).join(", ") : undefined}>{r.randomness == null ? "–" : r.randomness.toFixed(2)}</td>
                  <td>{fmtInt(r.object_count)}</td>
                  <td>{fmtTime(r.runtime_s)}</td>
                  <td onClick={(e) => e.stopPropagation()}>
                    {dir ? (
                      <>
                        <a href={`${dir}/result.json`} target="_blank" rel="noreferrer">result.json</a>
                        {" · "}
                        <a href={`${dir}/assembly.glb`} download>assembly.glb</a>
                      </>
                    ) : (
                      "—"
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
