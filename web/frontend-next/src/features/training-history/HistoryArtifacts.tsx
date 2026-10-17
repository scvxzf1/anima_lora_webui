import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "../../api/client";
import { Download, FileText } from "lucide-react";
import "./HistoryArtifacts.css";

type Artifact = { key: string; state: "available" | "missing" | "blocked" | "unreadable"; name?: string; size_bytes?: number; message?: string };
const labels: Record<string, string> = {
  "config-snapshot": "实际配置快照", "runtime-config": "运行配置", "original-config": "原始配置副本",
  "dataset-config": "数据集配置", logs: "完整日志", metrics: "训练指标", system: "系统采样",
};

export function HistoryArtifacts({ taskId }: { taskId: string }) {
  const base = `/api/training/history/${encodeURIComponent(taskId)}/artifacts`;
  const query = useQuery({
    queryKey: ["history-artifacts", taskId],
    queryFn: ({ signal }) => apiRequest<{ artifacts: Artifact[] }>(base, { signal }),
    retry: false,
  });
  return <section className="history-overview-section history-artifacts" aria-label="结果文件">
    <div className="toolbar"><h2>结果文件</h2><button type="button" disabled={query.isFetching} onClick={() => query.refetch()}>重新检查</button></div>
    {query.isPending && <p role="status">正在检查文件可用性</p>}
    {query.error && <p role="alert">{query.error.message}</p>}
    {query.error && query.data && <p className="data-scope">以下为上次检查结果，当前文件可用性尚未确认。</p>}
    {query.data?.artifacts.length === 0 && <p className="empty-state">此记录没有可列出的结果文件。</p>}
    <dl className="history-artifact-grid">{query.data?.artifacts.map((entry) => <div key={entry.key} data-state={entry.state}>
      <dt><FileText size={14} aria-hidden="true" />{labels[entry.key] || entry.key}</dt>
      <dd>{entry.state === "available" ? <>
        <a href={`${base}/${encodeURIComponent(entry.key)}?download=1`} download title={entry.name}>{entry.name}<Download size={14} aria-hidden="true" /></a>
        <small>{formatBytes(entry.size_bytes ?? 0)}</small>
      </> : <span>{entry.message || "不可用"}</span>}</dd>
    </div>)}</dl>
  </section>;
}

function formatBytes(bytes: number) {
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}
