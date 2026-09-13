import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { Square } from "lucide-react";
import { fetchGpus, fetchTrainingLogs, fetchTrainingMetrics, fetchTrainingStatus, liveMonitorKeys, stopTraining } from "./api";
import { TrainingMetricsCharts } from "../../components/TrainingMetricsCharts";
import { LogViewer } from "../../components/LogViewer";
import { QueryFeedback } from "../../components/QueryFeedback";
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
  const sourceTask = params.get("from_task");
  const client = useQueryClient();
  const statusQuery = useQuery({
    queryKey: liveMonitorKeys.status,
    queryFn: ({ signal }) => fetchTrainingStatus(signal),
    refetchInterval: 2000, retry: false,
  });
  const status = statusQuery.data;
  const taskId = status?.task_id;
  const metricsQuery = useQuery({
    queryKey: [...liveMonitorKeys.metrics, taskId || "none"],
    queryFn: ({ signal }) => fetchTrainingMetrics(taskId, signal),
    enabled: Boolean(taskId), refetchInterval: 5000, retry: false,
  });
  const logsQuery = useQuery({
    queryKey: [...liveMonitorKeys.logs, taskId || "none"],
    queryFn: ({ signal }) => fetchTrainingLogs(300, taskId, signal),
    enabled: Boolean(taskId), refetchInterval: 2000, retry: false,
  });
  const gpusQuery = useQuery({
    queryKey: liveMonitorKeys.gpus,
    queryFn: ({ signal }) => fetchGpus(signal),
    refetchInterval: 10000, retry: false,
  });
  const stop = useMutation({
    mutationFn: (clickedTaskId: string) => stopTraining(clickedTaskId),
    onSettled: () => client.invalidateQueries({ queryKey: liveMonitorKeys.status }),
    retry: false,
  });
  const connection = useMonitorRefresh();
  const running = RUNNING_STATES.has(status?.status || "");
  const state = statusQuery.error ? "状态待确认" : !status ? "读取中" : STATE_LABELS[status.status || ""] || "未知";

  return <div className="monitor-shell"><main className="monitor-page">
    <header className="monitor-header">
      <div>
        <h1>当前监控</h1>
        <p>{connection.status === "open" ? "实时连接已建立" : connection.status === "connecting"
          ? "正在连接；通过快照刷新" : "实时连接已断开；通过快照刷新"}</p>
        {taskId ? <p className="monitor-identity"><Link to={`/history/${encodeURIComponent(taskId)}`}>{taskId}</Link>
          {status?.variant ? ` · ${status.variant}` : ""}{status?.preset ? ` · ${status.preset}` : ""}
          {status?.latest_progress?.label ? ` · ${String(status.latest_progress.label)}` : ""}
        </p> : null}
      </div>
      <div className="monitor-header-actions">
        <span className="monitor-state" data-state={statusQuery.error ? "unknown" : running ? "running" : "idle"}>{state}</span>
        <button type="button" className="monitor-stop" disabled={!taskId || !running || Boolean(statusQuery.error) || stop.isPending}
          onClick={() => {
            if (taskId && window.confirm(`确定停止任务 ${taskId} 吗？`)) stop.mutate(taskId);
          }}><Square size={14} />{stop.isPending ? "正在停止" : "停止训练"}</button>
      </div>
    </header>
    <QueryFeedback query={statusQuery} label="任务状态" hasData={Boolean(status)} />
    {sourceTask && status && !statusQuery.error && sourceTask !== taskId && <p className="data-scope" role="status">
      来源任务 {sourceTask} 已不是当前监控对象。{taskId ? `当前显示 ${taskId}。` : "当前没有已确认的任务。"}
      <Link to={`/history/${encodeURIComponent(sourceTask)}`}>返回来源任务</Link>
    </p>}
    {stop.error ? <section className="monitor-error" role="alert"><h2>停止训练失败</h2>
      <p>目标任务：{stop.variables}</p><p>{stop.error.message}</p>
    </section> : null}
    {status && taskId ? <MonitorSummary status={status} /> : status ? <p role="status">
      {running ? "任务身份未确认，控制暂不可用。" : "暂无当前任务。"}
    </p> : null}
    {taskId ? <>
      <QueryFeedback query={metricsQuery} label="指标" hasData={metricsQuery.data !== undefined} />
      {metricsQuery.data && status?.job === "training" ? <TrainingMetricsCharts key={taskId} points={metricsQuery.data} total={status?.metric_count} /> : null}
    </> : null}
    <section className="monitor-grid">
      <div className="monitor-log-panel">
        <header><h2>实时日志</h2></header>
        {taskId ? <>
          <QueryFeedback query={logsQuery} label="日志" hasData={Boolean(logsQuery.data)} />
          {logsQuery.data ? <LogViewer key={taskId} lines={logsQuery.data.records} total={status?.log_count} /> : null}
        </> : <p className="monitor-empty">{statusQuery.error ? "任务身份待确认" : status ? "暂无当前任务日志" : "等待任务状态"}</p>}
      </div>
      <div className="monitor-gpu-panel">
        <header><h2>设备信息</h2></header>
        <QueryFeedback query={gpusQuery} label="GPU" hasData={Boolean(gpusQuery.data)} />
        <div className="monitor-gpu-list">
          {gpusQuery.data?.gpus?.map((gpu, index) => <GpuDetails key={String(gpu.index ?? index)} gpu={gpu} />)}
          {gpusQuery.data && !gpusQuery.data.gpus?.length ? <p className="monitor-empty">未检测到 GPU 信息</p> : null}
        </div>
      </div>
    </section>
  </main></div>;
}
