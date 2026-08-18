import { apiRequest } from '../../api/client';

export type HistoryTaskSummary = {
  id?: string;
  name?: string;
  job?: string;
  state?: string;
  archived?: boolean;
  group?: string;
  started_at?: number;
  started_at_text?: string;
  finished_at?: number;
  finished_at_text?: string;
  history_run_label?: string;
  history_source_config_file?: string;
  history_group_key?: string;
  run_dir?: string;
  output_dir?: string;
  training_output_dir?: string;
  log_count?: number;
  metric_count?: number;
  variant?: string;
  training_variant?: string;
  base_compute?: string;
  precision_preference?: string;
};

export type HistoryLogRecord = {
  id?: number;
  ts?: number;
  line?: string;
  type?: string;
};

export type HistoryMetricPoint = Record<string, unknown>;

export type HistoryTaskDetail = {
  ok?: boolean;
  task?: HistoryTaskSummary;
  logs?: HistoryLogRecord[];
  metrics?: HistoryMetricPoint[];
  system?: Record<string, unknown>[];
  limits?: Record<string, number | boolean>;
  config_toml?: string;
};

export type HistoryBatchPayload = {
  action: 'archive' | 'unarchive' | 'set_group' | 'delete';
  task_ids: string[];
  group?: string;
  delete_runtime_dirs?: boolean;
  confirmed?: boolean;
};

export const historyKeys = {
  list: ['training-history', 'list'] as const,
  detail: (taskId: string) => ['training-history', 'detail', taskId] as const,
};

export function fetchHistoryTasks(limit = 200, signal?: AbortSignal) {
  return apiRequest<{ ok?: boolean; tasks?: HistoryTaskSummary[] }>(
    `/api/training/history?limit=${limit}&include_archived=1`,
    { signal },
  );
}

export function fetchHistoryTaskDetail(taskId: string, signal?: AbortSignal) {
  return apiRequest<HistoryTaskDetail>(`/api/training/history/${encodeURIComponent(taskId)}`, { signal });
}

export function batchUpdateHistoryTasks(payload: HistoryBatchPayload) {
  return apiRequest<{ ok?: boolean; message?: string }>('/api/training/history/batch', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
