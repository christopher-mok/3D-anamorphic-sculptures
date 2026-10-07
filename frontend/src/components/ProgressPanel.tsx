// Live progress: job status, preprocessing diagnostics and one card per method.
import { withCacheBust } from "../api/client";
import type { ConnectionMode } from "../api/ws";
import type { JobStatus, MethodName, MethodProgress } from "../api/types";
import { ALL_METHODS, METHOD_LABELS } from "../api/types";
import { fmt, fmtInt, fmtPct, fmtTime, isNum } from "../utils/format";
import { ErrorText, Section, Spinner, StatusBadge } from "./common";

interface ProgressPanelProps {
  job: JobStatus | null;
  connection: ConnectionMode;
  lastUpdate: number;
  streamError: string | null;
  /** Target image url per view (preprocessed masks when available). */
  targetUrls: (string | null)[];
  displayedMethod: MethodName | null;
  onShowMethod: (m: MethodName) => void;
}

/** Methods present in a job, in request order. */
export function jobMethods(job: JobStatus): MethodName[] {
  const requested = job.request?.methods ?? [];
  const present = Object.keys(job.methods ?? {}) as MethodName[];
  const order = [...requested, ...ALL_METHODS];
  return order.filter((m, i) => order.indexOf(m) === i && (requested.includes(m) || present.includes(m)));
}

export function ProgressPanel(p: ProgressPanelProps) {
  const { job } = p;
  if (!job) {
    return (
      <Section title="Progress">
        {p.connection === "connecting" || p.connection === "polling" ? (
          <Spinner label="Connecting to job…" />
        ) : (
          <div className="hint">Start an optimization or open a previous job to see live progress.</div>
        )}
        <ErrorText>{p.streamError}</ErrorText>
      </Section>
    );
  }

  const numViews = Math.max(job.request?.cameras?.length ?? 0, job.request?.targets?.length ?? 0, 1);

  return (
    <Section
      title="Progress"
      right={<span className={`conn conn-${p.connection}`}>{p.connection}</span>}
    >
      <div className="job-line">
        <StatusBadge status={job.status} />
        <span className="muted small mono">{job.job_id}</span>
        {(job.status === "queued" || job.status === "preprocessing" || job.status === "running") && <Spinner />}
      </div>
      <ErrorText>{job.error}</ErrorText>
      <ErrorText>{p.streamError}</ErrorText>

      <PreprocessingInfo job={job} />

      {jobMethods(job).map((m) => (
        <MethodCard
          key={m}
          method={m}
          progress={job.methods?.[m]}
          numViews={numViews}
          targetUrls={p.targetUrls}
          lastUpdate={p.lastUpdate}
          displayed={p.displayedMethod === m}
          onShow={() => p.onShowMethod(m)}
        />
      ))}
    </Section>
  );
}

function PreprocessingInfo({ job }: { job: JobStatus }) {
  const pre = job.preprocessing;
  if (!pre) {
    if (job.status === "queued" || job.status === "preprocessing")
      return <div className="hint"><Spinner label="Preprocessing (targets, visual hull)…" /></div>;
    return null;
  }
  const bounds = pre.strict_coverage_upper_bound ?? [];
  return (
    <div className="prep">
      <div className="prep-row">
        <span className="muted">Hull voxels</span>
        <span>{fmtInt(pre.hull_voxels)}</span>
      </div>
      {bounds.map((b, i) => (
        <div className="prep-row" key={i}>
          <span className="muted">Strict coverage bound · view {i + 1}</span>
          <span className={isNum(b) && b < 0.9 ? "amber" : ""}>{fmtPct(b)}</span>
        </div>
      ))}
      {(pre.warnings ?? []).length > 0 && (
        <ul className="warnings">
          {pre.warnings.map((w, i) => (
            <li key={i}>⚠ {w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

interface MethodCardProps {
  method: MethodName;
  progress: MethodProgress | undefined;
  numViews: number;
  targetUrls: (string | null)[];
  lastUpdate: number;
  displayed: boolean;
  onShow: () => void;
}

function MethodCard({ method, progress, numViews, targetUrls, lastUpdate, displayed, onShow }: MethodCardProps) {
  const status = progress?.status ?? "pending";
  const views = Array.from({ length: numViews }, (_, i) => i);
  const m = progress?.metrics ?? null;
  const viewIou = m?.view_iou?.length ? m.view_iou : progress?.view_iou ?? [];
  const canShow = !!progress && (!!progress.assembly || !!progress.result_dir_url);

  return (
    <div className={`method-card ${displayed ? "displayed" : ""}`}>
      <div className="method-header">
        <strong>{METHOD_LABELS[method] ?? method}</strong>
        <StatusBadge status={status} />
        {status === "running" && <Spinner />}
        <span className="spacer" />
        <button type="button" className="small-btn" disabled={!canShow} onClick={onShow}>
          {displayed ? "Shown in 3D" : "Show in 3D"}
        </button>
      </div>
      {progress && (
        <div className="stats">
          <Stat label="Phase" value={progress.phase || "—"} />
          <Stat label="Iteration" value={fmtInt(progress.iteration)} />
          <Stat label="Runtime" value={fmtTime(m?.runtime_s ?? progress.runtime_s)} />
          <Stat label="Loss" value={fmt(m?.loss ?? progress.loss, 4)} />
          <Stat label="Objects" value={fmtInt(m?.object_count ?? progress.object_count)} />
          <Stat label="Best min-IoU" value={fmt(progress.best_metric, 3)} />
          {views.map((v) => (
            <Stat key={v} label={`IoU view ${v + 1}`} value={fmt(viewIou[v], 3)} />
          ))}
          {m && <Stat label="Mean IoU" value={fmt(m.mean_iou, 3)} />}
        </div>
      )}
      <ErrorText>{progress?.error}</ErrorText>
      {progress && (progress.preview_urls?.length ?? 0) > 0 && (
        <div className="previews">
          {views.map((v) => {
            const preview = progress.preview_urls[v];
            const target = targetUrls[v];
            return (
              <div className="preview-pair" key={v}>
                <figure>
                  {target ? <img src={target} alt={`target ${v + 1}`} /> : <div className="img-missing" />}
                  <figcaption>Target {v + 1}</figcaption>
                </figure>
                <figure>
                  {preview ? (
                    <img src={withCacheBust(preview, lastUpdate)} alt={`render ${v + 1}`} />
                  ) : (
                    <div className="img-missing" />
                  )}
                  <figcaption>Render {v + 1}</figcaption>
                </figure>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}
