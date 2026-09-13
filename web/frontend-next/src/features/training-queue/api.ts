import { apiRequest } from '../../api/client';

export type QueueItemState = 'queued' | 'running' | 'done' | 'error' | 'canceled';

export type QueueItem = {
  id?: string;
  state?: string;
  kind?: string;
  requires_preprocess?: boolean;
  variant?: string;
  preset?: string;
  methods_subdir?: string;
  runtime_config_file?: string;
  source_config_file?: string;
  gpu_whitelist?: string[];
  resume_info?: { checkpoint?: string; checkpoint_name?: string } & Record<string, unknown>;
  retry_of?: string;
  attempt?: number;
  max_attempts?: number;
  history_task_ids?: string[];
  message?: string;
  created_at?: number;
  created_at_text?: string;
  started_at?: number | null;
  started_at_text?: string;
  finished_at?: number | null;
  finished_at_text?: string;
};

export type QueueSnapshot = {
  ok?: boolean;
  revision?: string;
  current_task_id?: string;
  paused?: boolean;
  failure_policy?: string;
  auto_retry?: boolean;
  max_attempts?: number;
  retry_backoff_sec?: number;
  status?: string;
  current_item_id?: string;
  summary?: {
    total?: number;
    queued?: number;
    running?: number;
    done?: number;
    error?: number;
    canceled?: number;
  };
  items?: QueueItem[];
  message?: string;
};

export type QueueSettingsPayload = {
  paused?: boolean;
  failure_policy?: string;
  auto_retry?: boolean;
  max_attempts?: number;
  retry_backoff_sec?: number;
};

export const queueKeys = {
  snapshot: ['training-queue', 'snapshot'] as const,
};

export function fetchQueueSnapshot(signal?: AbortSignal) {
  return apiRequest<QueueSnapshot>('/api/training/queue', { signal });
}

export function saveQueueSettings(payload: QueueSettingsPayload) {
  return apiRequest<QueueSnapshot>('/api/training/queue/settings', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function moveQueueItem(itemId: string, direction: 'up' | 'down' | 'top' | 'bottom') {
  return apiRequest<QueueSnapshot>(`/api/training/queue/${encodeURIComponent(itemId)}/move`, {
    method: 'POST',
    body: JSON.stringify({ direction }),
  });
}

export function retryQueueItem(itemId: string) {
  return apiRequest<QueueSnapshot>(`/api/training/queue/${encodeURIComponent(itemId)}/retry`, {
    method: 'POST',
  });
}

export function cancelQueueItem(itemId: string, deleteRuntime = false) {
  return apiRequest<QueueSnapshot>(`/api/training/queue/${encodeURIComponent(itemId)}`, {
    method: 'DELETE',
    body: JSON.stringify({ delete_runtime: deleteRuntime }),
  });
}

function bulkQueueCommand(action: string, revision: string) {
  if (!revision) throw new Error('缺少队列快照版本，请刷新后重试');
  return apiRequest<QueueSnapshot>(`/api/training/queue/${action}`, { method: 'POST', body: JSON.stringify({ expected_revision: revision }) });
}

export function cancelAllQueueItems(revision: string) {
  return bulkQueueCommand('cancel-all', revision);
}

export function abortQueueAfterCurrent(revision: string) {
  return bulkQueueCommand('abort-after-current', revision);
}

export function forceAbortQueue(revision: string) {
  return bulkQueueCommand('force-abort', revision);
}

export function cancelWaitingQueueItems(revision: string) {
  return bulkQueueCommand('cancel-waiting', revision);
}

export function clearCompletedQueueItems(revision: string) {
  return bulkQueueCommand('clear-completed', revision);
}

export function clearCanceledQueueItems(revision: string) {
  return bulkQueueCommand('clear-canceled', revision);
}
