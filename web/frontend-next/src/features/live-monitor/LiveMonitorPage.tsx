import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';
import { Topbar } from '../../app/Topbar';
import { useWebSocket } from '../../app/useWebSocket';
import {
  fetchGpus,
  fetchTrainingLogs,
  fetchTrainingMetrics,
  fetchTrainingStatus,
  liveMonitorKeys,
  stopTraining,
  type TrainingStatus,
} from './api';
import './LiveMonitorPage.css';

const RUNNING_STATES = new Set(['running', 'training', 'compiling', 'caching', 'saving']);
const LOG_LIMIT = 300;

function isRunning(status?: string) {
  return Boolean(status && RUNNING_STATES.has(status));
}

function asNumber(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatValue(value: unknown) {
  const number = asNumber(value);
  if (number === null) return 'N/A';
  return Math.abs(number) >= 100 ? number.toFixed(0) : number.toFixed(3);
}

function stateLabel(status?: string) {
  const labels: Record<string, string> = {
    idle: '空闲',
    running: '运行中',
    training: '训练中',
    compiling: '编译中',
    caching: '缓存中',
    saving: '保存中',
    error: '异常',
    unavailable: '不可用',
  };
  return labels[status || ''] || status || '未知';
}

export function LiveMonitorPage() {
  const queryClient = useQueryClient();
  const statusQuery = useQuery({
    queryKey: liveMonitorKeys.status,
    queryFn: ({ signal }) => fetchTrainingStatus(signal),
    refetchInterval: 2000,
  });
  const metricsQuery = useQuery({
    queryKey: liveMonitorKeys.metrics,
    queryFn: ({ signal }) => fetchTrainingMetrics(signal),
    refetchInterval: 5000,
  });
  const logsQuery = useQuery({
    queryKey: liveMonitorKeys.logs,
    queryFn: ({ signal }) => fetchTrainingLogs(LOG_LIMIT, signal),
    refetchInterval: 2000,
  });
  const gpusQuery = useQuery({
    queryKey: liveMonitorKeys.gpus,
    queryFn: ({ signal }) => fetchGpus(signal),
    refetchInterval: 10000,
  });

  const stop = useMutation({
    mutationFn: stopTraining,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: liveMonitorKeys.status });
    },
  });

  useWebSocket('/ws/training', (message) => {
    if (message.type === 'progress') {
      const progress = (message.progress && typeof message.progress === 'object'
        ? message.progress
        : message) as Record<string, unknown>;
      queryClient.setQueryData<TrainingStatus>(liveMonitorKeys.status, (current) => ({
        ...(current || {}),
        status: 'running',
        latest_progress: { ...((current?.latest_progress || {}) as Record<string, unknown>), ...progress },
      }));
    } else if (message.type === 'status') {
      queryClient.setQueryData<TrainingStatus>(liveMonitorKeys.status, (current) => ({
        ...(current || {}),
        status: String(message.state || message.status || current?.status || ''),
        last_log_line: String(message.message || current?.last_log_line || ''),
      }));
    } else if (message.type === 'log') {
      const record = message as { id?: number } & Record<string, unknown>;
      queryClient.setQueryData<{ records: { id?: number }[] }>(liveMonitorKeys.logs, (current) => {
        const records = current?.records || [];
        if (record.id != null && records.some((item) => item.id === record.id)) return current;
        return { records: [...records, record].slice(-LOG_LIMIT) };
      });
    } else if (message.type === 'system') {
      queryClient.setQueryData<TrainingStatus>(liveMonitorKeys.status, (current) => ({
        ...(current || {}),
        latest_system: { ...((current?.latest_system || {}) as Record<string, unknown>), ...message },
      }));
    } else if (message.type === 'metrics') {
      queryClient.invalidateQueries({ queryKey: liveMonitorKeys.metrics });
    }
  });

  const status = statusQuery.data;
  const progress = (status?.latest_progress || {}) as TrainingStatus['latest_progress'];
  const system = status?.latest_system || {};
  const metric = status?.latest_metric || {};
  const logs = logsQuery.data?.records || [];
  const gpus = gpusQuery.data?.gpus || [];

  const step = asNumber(progress?.current ?? metric?.step ?? 0) || 0;
  const total = asNumber(progress?.total ?? metric?.total ?? 0) || 0;
  const progressPct = total > 0 ? Math.min(100, (step / total) * 100) : 0;
  const running = isRunning(status?.status);

  const summary = useMemo(() => ({
    loss: progress?.loss ?? metric?.loss,
    lr: progress?.lr ?? metric?.lr,
    rate: progress?.rate ?? metric?.rate,
    vram: system.vram_used_gb,
    vramTotal: system.vram_total_gb,
    gpuTemp: system.gpu_temp,
    gpuUtil: system.gpu_util,
  }), [progress, metric, system]);

  return (
    <div className="monitor-shell">
      <Topbar />
      <main className="monitor-page">
        <header className="monitor-header">
          <div>
            <p className="eyebrow">LIVE TRAINING</p>
            <h1>当前监控</h1>
            <p>查看正在运行任务的进度、指标和日志；无任务时保持空闲。</p>
          </div>
          <div className="monitor-header-actions">
            <span className="monitor-state" data-state={running ? 'running' : 'idle'}>
              {stateLabel(status?.status)}
            </span>
            <button
              type="button"
              className="monitor-stop"
              disabled={!running || stop.isPending}
              onClick={() => {
                if (window.confirm('确定停止当前训练吗？')) stop.mutate();
              }}
            >
              {stop.isPending ? '正在停止' : '停止训练'}
            </button>
          </div>
        </header>

        {statusQuery.error || metricsQuery.error || logsQuery.error ? (
          <section className="monitor-error" role="alert">
            <h2>无法读取训练监控</h2>
            <p>{(statusQuery.error || metricsQuery.error || logsQuery.error)?.message}</p>
          </section>
        ) : (
          <>
            <section className="monitor-progress-card" aria-label="训练进度">
              <div className="monitor-progress-copy">
                <div><span>进度</span><strong>{progressPct.toFixed(1)}%</strong></div>
                <div><span>步数</span><strong>{step}/{total || '—'}</strong></div>
                <div><span>任务目录</span><strong title={status?.output_dir || ''}>{status?.output_dir || '暂无正在运行的任务目录'}</strong></div>
              </div>
              <div className="monitor-progress-track"><div style={{ width: `${progressPct}%` }} /></div>
              {status?.anomaly_message ? <p className="monitor-anomaly" role="status">{status.anomaly_message}</p> : null}
            </section>

            <section className="monitor-metrics" aria-label="实时指标">
              <MetricCard label="Loss" value={formatValue(summary.loss)} />
              <MetricCard label="学习率" value={formatValue(summary.lr)} />
              <MetricCard label="速度" value={summary.rate ? String(summary.rate) : 'N/A'} />
              <MetricCard label="显存" value={summary.vram != null ? `${Number(summary.vram).toFixed(1)} / ${summary.vramTotal != null ? `${Number(summary.vramTotal).toFixed(1)} GB` : 'GB'}` : 'N/A'} />
              <MetricCard label="GPU 温度" value={summary.gpuTemp != null ? `${Number(summary.gpuTemp).toFixed(0)}°C` : 'N/A'} />
              <MetricCard label="GPU 利用率" value={summary.gpuUtil != null ? `${Number(summary.gpuUtil).toFixed(0)}%` : 'N/A'} />
            </section>

            <section className="monitor-grid">
              <div className="monitor-log-panel">
                <header><div><span className="eyebrow">日志</span><h2>实时日志</h2></div><span>{logs.length} 行</span></header>
                <LogPanel logs={logs} />
              </div>
              <div className="monitor-gpu-panel">
                <header><div><span className="eyebrow">GPU</span><h2>设备状态</h2></div><span>{gpus.length} 张</span></header>
                <div className="monitor-gpu-list">
                  {gpus.length ? gpus.map((gpu, index) => (
                    <dl key={index} className="monitor-gpu-card">
                      {Object.entries(gpu).slice(0, 8).map(([key, value]) => (
                        <div key={key}><dt>{key}</dt><dd>{String(value)}</dd></div>
                      ))}
                    </dl>
                  )) : <p className="monitor-empty">暂无 GPU 信息</p>}
                </div>
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return <div className="monitor-metric"><span>{label}</span><strong>{value}</strong></div>;
}

function LogPanel({ logs }: { logs: { id?: number; line?: string; type?: string }[] }) {
  if (!logs.length) {
    return <p className="monitor-empty">暂无日志，任务启动后会显示在这里。</p>;
  }
  return (
    <div className="monitor-log-list" role="log" aria-live="polite">
      {logs.map((log, index) => (
        <div key={log.id || index} className="monitor-log-line" data-type={log.type || 'info'}>
          {log.line || String(log)}
        </div>
      ))}
    </div>
  );
}
