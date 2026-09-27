import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowRight, Square } from "lucide-react";
import { fetchGpus, fetchTrainingLogs, fetchTrainingMetrics, fetchTrainingStatus, liveMonitorKeys, stopTraining, type GpuInventory, type TrainingStatus } from "./api";
import { TrainingMetricsCharts } from "../../components/TrainingMetricsCharts";
import { LogViewer } from "../../components/LogViewer";
import { QueryFeedback } from "../../components/QueryFeedback";
import { InlineConfirmDialog } from "../../components/InlineConfirmDialog";
import { MonitorSummary, GpuDetails } from "./MonitorSummary";
import { useMonitorRefresh } from "./useMonitorRefresh";
import "./LiveMonitorPage.css";

const RUNNING_STATES = new Set(["running", "training", "compiling", "caching", "saving"]);
const STATE_LABELS: Record<string, string> = {
  idle: "空闲", running: "运行中", training: "训练中", compiling: "编译中",
  caching: "缓存中", saving: "保存中", error: "异常", unavailable: "不可用",
};

export function LiveMonitorPage() {
  const [params] = useSearchParams();
  const [stopTarget, setStopTarget] = useState<string | null>(null);
  const sourceTask = params.get("from_task");
  const client = useQueryClient();
  const statusQuery = useQuery({
    queryKey: liveMonitorKeys.status,
    queryFn: ({ signal }) => fetchTrainingStatus(signal),
    refetchInterval: (query) => query.state.error ? false : 2000,
    retry: false,
  });
  const status = statusQuery.data;
  const taskId = status?.task_id;
  const metricsQuery = useQuery({
    queryKey: [...liveMonitorKeys.metrics, taskId || "none"],
    queryFn: ({ signal }) => fetchTrainingMetrics(taskId, signal),
    enabled: Boolean(taskId) && !statusQuery.error, refetchInterval: 5000, retry: false,
  });
  const logsQuery = useQuery({
    queryKey: [...liveMonitorKeys.logs, taskId || "none"],
    queryFn: ({ signal }) => fetchTrainingLogs(300, taskId, signal),
    enabled: Boolean(taskId) && !statusQuery.error, refetchInterval: 2000, retry: false,
  });
  const gpusQuery = useQuery({
    queryKey: liveMonitorKeys.gpus,
    queryFn: ({ signal }) => fetchGpus(signal),
    refetchInterval: RUNNING_STATES.has(status?.status || "") ? 2000 : 10000, retry: false,
  });
  const stop = useMutation({
    mutationFn: (clickedTaskId: string) => stopTraining(clickedTaskId),
    onSettled: () => client.invalidateQueries({ queryKey: liveMonitorKeys.status }),
    retry: false,
  });
  const connection = useMonitorRefresh();
  const running = RUNNING_STATES.has(status?.status || "");
  const state = statusQuery.error ? "状态待确认" : !status ? "读取中" : STATE_LABELS[status.status || ""] || "未知";
  useEffect(() => {
    if (stopTarget && (stopTarget !== taskId || !running || statusQuery.error)) setStopTarget(null);
  }, [stopTarget, taskId, running, statusQuery.error]);

  return <div className="monitor-shell"><main className="monitor-page">
    <header className="monitor-header">
      <div className="monitor-header-main">
        <h1>当前监控</h1>
        {taskId ? <p className="monitor-identity"><Link to={`/history/${encodeURIComponent(taskId)}`}>{taskId}</Link>
          {status?.variant ? ` · ${status.variant}` : ""}{status?.preset ? ` · ${status.preset}` : ""}
          {status?.latest_progress?.label ? ` · ${String(status.latest_progress.label)}` : ""}
        </p> : null}
      </div>
      <div className="monitor-header-actions">
        <span className="monitor-state" role="status" aria-label="当前任务状态" data-state={statusQuery.error ? "unknown" : running ? "running" : status?.status === "error" ? "error" : "idle"}>{state}</span>
        {taskId ? <button type="button" className="monitor-stop" disabled={!running || Boolean(statusQuery.error) || stop.isPending}
          onClick={() => setStopTarget(taskId)}><Square size={14} />{stop.isPending ? "正在停止" : "停止训练"}</button> : null}
      </div>
    </header>
    <div className="monitor-connection-row">
      <p role="status" aria-label="实时连接状态" data-connection={connection.status}>{connection.status === "open" ? "实时连接已建立" : connection.status === "connecting"
        ? "正在连接；通过快照刷新" : "实时连接已断开；通过快照刷新"}</p>
      {statusQuery.dataUpdatedAt > 0 ? <span>任务状态最近成功读取 <time dateTime={new Date(statusQuery.dataUpdatedAt).toISOString()}>{new Date(statusQuery.dataUpdatedAt).toLocaleTimeString()}</time></span> : null}
    </div>
    {(statusQuery.error || statusQuery.isPending) ? <QueryFeedback query={statusQuery} label="任务状态" hasData={Boolean(status)} /> : null}
    {statusQuery.error && taskId ? <p className="monitor-stale" role="status">
      任务状态未确认。以下为上次成功读取的任务快照，停止操作暂不可用。
    </p> : null}
    {sourceTask && status && !statusQuery.error && sourceTask !== taskId && <p className="data-scope" role="status">
      来源任务 {sourceTask} 已不是当前监控对象。{taskId ? `当前显示 ${taskId}。` : "当前没有已确认的任务。"}
      <Link to={`/history/${encodeURIComponent(sourceTask)}`}>返回来源任务</Link>
    </p>}
    {stop.error ? <section className="monitor-error" role="alert"><h2>停止训练失败</h2>
      <p>目标任务：{stop.variables}</p><p>{stop.error.message}</p>
    </section> : null}
    {status && taskId ? <MonitorSummary status={status} /> : status ? <section className="monitor-idle" aria-label="空闲状态">
      <h2>{statusQuery.error ? "任务状态待确认。" : running ? "任务身份未确认，控制暂不可用。" : "暂无当前任务。"}</h2>
      {!running && !statusQuery.error ? <nav aria-label="空闲时操作"><Link to="/training">训练配置 <ArrowRight size={14} /></Link><Link to="/queue">训练队列 <ArrowRight size={14} /></Link></nav> : null}
    </section> : null}
    {taskId ? <div className="monitor-workspace" data-has-charts={status?.job === "training"}>
      {status?.job === "training" ? <section className="monitor-chart-panel" aria-label="训练趋势">
        <QueryFeedback query={metricsQuery} label="指标" hasData={metricsQuery.data !== undefined} />
        {metricsQuery.data ? <TrainingMetricsCharts key={taskId} points={metricsQuery.data} total={status?.metric_count} /> : null}
      </section> : null}
      <GpuPanel query={gpusQuery} status={status} confirmed={Boolean(status) && !statusQuery.error} />
      <section className="monitor-log-panel" aria-label="实时日志">
        <header><h2>实时日志</h2></header>
        <QueryFeedback query={logsQuery} label="日志" hasData={Boolean(logsQuery.data)} />
        {logsQuery.data ? <LogViewer key={taskId} lines={logsQuery.data.records} total={status?.log_count} /> : null}
      </section>
    </div> : <div className="monitor-idle-device"><GpuPanel query={gpusQuery} status={status} confirmed={Boolean(status) && !statusQuery.error} /></div>}
    {stopTarget && stopTarget === taskId && running && !statusQuery.error ? <InlineConfirmDialog
      title="停止训练"
      message={`确定停止任务 ${stopTarget} 吗？`}
      confirmLabel="停止训练"
      danger
      onCancel={() => setStopTarget(null)}
      onConfirm={() => {
        if (taskId === stopTarget && running && !stop.isPending) stop.mutate(stopTarget);
        setStopTarget(null);
      }}
    /> : null}
  </main></div>;
}

function GpuPanel({ query, status, confirmed }: {
  query: UseQueryResult<GpuInventory, Error>; status?: TrainingStatus; confirmed: boolean;
}) {
  const running = confirmed && RUNNING_STATES.has(status?.status || "");
  const selected = running && Array.isArray(status?.gpu_whitelist) ? status.gpu_whitelist : [];
  const sampledAt = query.data?.sampled_at;
  const unavailable = Boolean(query.error || query.data?.stale);
  return <section className="monitor-gpu-panel" aria-label="设备信息">
    <header><h2>设备信息</h2>{typeof sampledAt === "number" && Number.isFinite(sampledAt) && !unavailable ? <time dateTime={new Date(sampledAt * 1000).toISOString()}>
      采样于 {new Date(sampledAt * 1000).toLocaleTimeString()}
    </time> : null}</header>
    <QueryFeedback query={query} label="GPU" hasData={Boolean(query.data)} />
    {query.data?.stale ? <p className="monitor-gpu-warning" role="status">GPU 采样暂不可用</p> : null}
    {running && selected.some((index) => !query.data?.gpus?.some((gpu) => gpu.index === index)) && !unavailable ?
      <p className="monitor-gpu-warning" role="status">当前任务所选 GPU 未全部出现在设备列表中</p> : null}
    <div className="monitor-gpu-list">
      {!unavailable && query.data?.gpus?.map((gpu, index) => <GpuDetails key={String(gpu.uuid ?? gpu.index ?? index)} gpu={gpu}
        participation={!confirmed || running && !selected.length ? "参与状态未确认" : !running ? "当前无任务" : selected.includes(Number(gpu.index)) ? "当前任务已选" : "未选用"} />)}
      {query.data && !unavailable && !query.data.gpus?.length ? <p className="monitor-empty">未检测到 GPU 信息</p> : null}
    </div>
  </section>;
}
