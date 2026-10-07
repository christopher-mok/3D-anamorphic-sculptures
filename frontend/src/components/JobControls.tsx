// Step 7: start / cancel optimization and reopen previous jobs.
import type { JobStatus } from "../api/types";
import { METHOD_SHORT } from "../api/types";
import { fmtDate } from "../utils/format";
import { ErrorText, Section, Spinner } from "./common";

interface JobControlsProps {
  canStart: boolean;
  startBlockers: string[];
  submitting: boolean;
  jobActive: boolean;
  onStart: () => void;
  onCancel: () => void;
  cancelling: boolean;
  error: string | null;
  jobs: JobStatus[];
  currentJobId: string | null;
  onOpenJob: (id: string) => void;
  onRefreshJobs: () => void;
}

function jobLabel(j: JobStatus): string {
  const methods = (j.request?.methods ?? []).map((m) => METHOD_SHORT[m] ?? m).join(", ");
  return `${fmtDate(j.created_at)} · ${j.status} · ${methods || "?"} · ${j.job_id.slice(0, 8)}`;
}

export function JobControls(p: JobControlsProps) {
  return (
    <Section title="5 · Run">
      <div className="run-buttons">
        <button type="button" className="primary" disabled={!p.canStart || p.submitting || p.jobActive} onClick={p.onStart}>
          {p.submitting ? <Spinner label="Starting…" /> : "Start optimization"}
        </button>
        {p.jobActive && (
          <button type="button" className="danger" onClick={p.onCancel} disabled={p.cancelling}>
            {p.cancelling ? "Cancelling…" : "Cancel"}
          </button>
        )}
      </div>
      {!p.canStart && p.startBlockers.length > 0 && (
        <ul className="blockers">
          {p.startBlockers.map((b) => (
            <li key={b}>{b}</li>
          ))}
        </ul>
      )}
      <ErrorText>{p.error}</ErrorText>

      <div className="field-row">
        <label>Previous jobs</label>
        <div className="inline-form">
          <select value={p.currentJobId ?? ""} onChange={(e) => e.target.value && p.onOpenJob(e.target.value)}>
            <option value="">{p.jobs.length ? "— open a job —" : "— none —"}</option>
            {p.currentJobId && !p.jobs.some((j) => j.job_id === p.currentJobId) && (
              <option value={p.currentJobId}>{p.currentJobId}</option>
            )}
            {p.jobs.map((j) => (
              <option key={j.job_id} value={j.job_id}>
                {jobLabel(j)}
              </option>
            ))}
          </select>
          <button type="button" className="small-btn" onClick={p.onRefreshJobs} title="Refresh job list">
            ↻
          </button>
        </div>
      </div>
    </Section>
  );
}
