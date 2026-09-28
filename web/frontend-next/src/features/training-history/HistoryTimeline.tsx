import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { fetchHistoryTimeline, historyKeys } from "./api";
import { orderHistoryTimeline, timelineLogText } from "./timelineData";
import "./HistoryTimeline.css";

export function HistoryTimeline({ taskIds, onClose }: { taskIds: string[]; onClose: () => void }) {
  const ids = [...new Set(taskIds.filter(Boolean))];
  const query = useQuery({
    queryKey: [...historyKeys.list, "timeline", ...ids],
    queryFn: ({ signal }) => fetchHistoryTimeline(ids, signal),
    enabled: ids.length >= 2,
    retry: false,
  });
  const data = query.data ? orderHistoryTimeline(query.data, ids) : undefined;
  const loss = (data?.metrics || []).filter((point) => typeof point.loss === "number" && Number.isFinite(point.loss));
  const logs = data?.logs || [];
  const availableMetricCount = (data?.segments || []).reduce((sum, segment) => sum + Number(segment.metric_count || 0), 0);
  const availableLogCount = (data?.segments || []).reduce((sum, segment) => sum + Number(segment.log_count || 0), 0);
  const visibleMetricsByTask = new Map<string, number>();
  const visibleLogsByTask = new Map<string, number>();
  for (const point of data?.metrics || []) {
    const id = String(point.source_task_id || "");
    visibleMetricsByTask.set(id, (visibleMetricsByTask.get(id) || 0) + 1);
  }
  for (const log of logs) {
    const id = String(log.source_task_id || "");
    visibleLogsByTask.set(id, (visibleLogsByTask.get(id) || 0) + 1);
  }
  return <section className="history-timeline" aria-label="训练合并时间线">
    <header><div><p className="eyebrow">SEQUENTIAL HISTORY</p><h2>合并查看</h2><p>按已选顺序串接训练阶段；与并列任务对比独立。</p></div>
      <button type="button" title="关闭合并查看" aria-label="关闭合并查看" onClick={onClose}><X size={16} /></button></header>
    {ids.length < 2 ? <p className="history-error" role="alert">至少选择两个训练任务。</p> : null}
    {query.isPending && ids.length >= 2 ? <p role="status">正在读取合并时间线…</p> : null}
    {query.error || (query.data && query.data.ok === false) ? <p className="history-error" role="alert">{query.error?.message || query.data?.error || "读取合并时间线失败。"} <button type="button" disabled={query.isFetching} onClick={() => void query.refetch()}>重试</button></p> : null}
    {data && data.ok !== false ? <>
      {(availableMetricCount > (data.metrics?.length || 0) || availableLogCount > logs.length) && (
        <p className="history-notice" role="status">
          服务端限制了时间线记录数：指标 {data.metrics?.length || 0} / {availableMetricCount}，日志 {logs.length} / {availableLogCount}。较早阶段可能不完整。
        </p>
      )}
      <ol className="history-timeline-stages">{(data.segments || []).map((segment, index) => <li key={segment.task?.id || index}>
        <strong>{index + 1}. {segment.task?.label || segment.task?.name || segment.task?.id || "训练任务"}</strong>
        <span>{visibleMetricsByTask.get(String(segment.task?.id || "")) || 0}/{segment.metric_count || 0} 个指标 · {visibleLogsByTask.get(String(segment.task?.id || "")) || 0}/{segment.log_count || 0} 行日志{segment.start_display_step != null ? ` · 训练步数 ${segment.start_display_step}–${segment.end_display_step}` : ""}</span>
      </li>)}</ol>
      <section className="history-timeline-chart" aria-label="串接 Loss 曲线">
        <h3>Loss · 串接指标序号</h3>
        {loss.length ? <TimelineChart points={loss} /> : <p>所选任务没有可绘制的 Loss 指标。</p>}
      </section>
      <section className="history-timeline-logs"><h3>聚合日志 <small>{logs.length} 行 · 已过滤进度事件</small></h3>
        {logs.length ? <pre>{logs.map(timelineLogText).join("\n")}</pre> : <p>所选任务没有非进度日志。</p>}
      </section>
    </> : null}
  </section>;
}

function TimelineChart({ points }: { points: Array<Record<string, unknown>> }) {
  const values = points.map((point) => Number(point.loss));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const coords = points.map((point, index) => `${index / Math.max(1, points.length - 1) * 1000},${190 - (Number(point.loss) - min) / span * 170}`).join(" ");
  const separators = points.flatMap((point, index) => point.stage_break_before ? [index / Math.max(1, points.length - 1) * 1000] : []);
  return <svg viewBox="0 0 1000 220" role="img" aria-label={`合并 Loss 曲线，共 ${points.length} 个点`}>
    {separators.map((x, index) => <g key={`${x}-${index}`}><line x1={x} x2={x} y1="10" y2="195" /><text x={x + 5} y="16">阶段分界</text></g>)}
    <polyline points={coords} />
    <text x="8" y="212">1</text><text x="940" y="212">{points.length}</text>
  </svg>;
}
