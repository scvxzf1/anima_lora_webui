import { apiRequest } from '../../api/client';

export type TrainingStatus = {
  status?: string;
  variant?: string;
  preset?: string;
  job?: string;
  output_dir?: string;
  task_id?: string;
  last_log_line?: string;
  last_log_id?: number;
  log_count?: number;
  metric_count?: number;
  latest_progress?: Record<string, unknown> & {
    current?: number;
    total?: number;
    loss?: number;
    lr?: number;
    rate?: string;
  };
  latest_metric?: Record<string, unknown>;
  latest_system?: Record<string, unknown> & {
    vram_used_gb?: number;
    vram_total_gb?: number;
    gpu_temp?: number;
    gpu_util?: number;
  };
  error_hint?: string;
  anomaly_message?: string;
};

export type LogRecord = {
  id?: number;
  ts?: number;
  line?: string;
  type?: string;
  level?: string;
};

export type GpuInfo = Record<string, unknown>;

export const liveMonitorKeys = {
  status: ['live-monitor', 'status'] as const,
  metrics: ['live-monitor', 'metrics'] as const,
  logs: ['live-monitor', 'logs'] as const,
  gpus: ['live-monitor', 'gpus'] as const,
};

export function fetchTrainingStatus(signal?: AbortSignal) {
  return apiRequest<TrainingStatus>('/api/training/status', { signal });
}

export function fetchTrainingMetrics(signal?: AbortSignal) {
  return apiRequest<Record<string, unknown>[]>('/api/training/metrics', { signal });
}

export function fetchTrainingLogs(limit = 300, signal?: AbortSignal) {
  return apiRequest<{ records: LogRecord[] }>(`/api/training/logs?limit=${limit}`, { signal });
}

export function fetchGpus(signal?: AbortSignal) {
  return apiRequest<{ ok?: boolean; gpus?: GpuInfo[] }>('/api/training/gpus', { signal });
}

export function stopTraining() {
  return apiRequest<{ ok?: boolean; message?: string }>('/api/training/stop', { method: 'POST' });
}
