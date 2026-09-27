import { apiRequest } from "../../api/client";

export type HistoryResumeSource = {
  source_task_id?: string;
  source_task_name?: string;
  checkpoint?: string;
  checkpoint_name?: string;
  checkpoint_kind?: string;
  checkpoint_epoch?: number | null;
  checkpoint_step?: number | null;
  target_total_steps?: number | null;
  remaining_steps?: number | null;
};

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
  history_group_label?: string;
  methods_subdir?: string;
  preset?: string;
  run_dir?: string;
  output_dir?: string;
  training_output_dir?: string;
  log_count?: number;
  metric_count?: number;
  variant?: string;
  gpu_whitelist?: number[];
  training_variant?: string;
  base_compute?: string;
  precision_preference?: string;
  preprocess_precision_preference?: string;
  model_family?: string;
  preprocess_precision?: string;
  block_swap_precision?: string;
  source_task_id?: string;
  source_task_name?: string;
  message?: string;
  returncode?: number;
  sample_dir?: string;
  source_image_dir?: string;
  resized_image_dir?: string;
  lora_cache_dir?: string;
  dataset_cache_dir?: string;
  resume_from?: HistoryResumeSource | string | null;
  linked_preprocess_task?: HistoryTaskSummary;
  last_step?: number;
  final_loss?: number;
  loss_preview?: number[];
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
  action: "archive" | "unarchive" | "set_group" | "delete";
  task_ids: string[];
  group?: string;
  delete_runtime_dirs?: boolean;
  confirmed?: boolean;
};

export const historyKeys = {
  list: ["training-history", "list"] as const,
  detail: (taskId: string) => ["training-history", "detail", taskId] as const,
  collections: ["training-history", "collections"] as const,
};

export type HistoryCollections = {
  collection_order: string[];
  config_group_order: Record<string, string[]>;
};
export const fetchHistoryCollections = (signal?: AbortSignal) =>
  apiRequest<HistoryCollections>("/api/training/history/collections/settings", {
    signal,
  });
export const saveHistoryCollections = (payload: HistoryCollections) =>
  apiRequest<HistoryCollections>("/api/training/history/collections/settings", {
    method: "PUT",
    body: JSON.stringify(payload),
  });

export function fetchHistoryTasks(limit = 200, signal?: AbortSignal, search = "", cursor: string | number = 0) {
  const params = new URLSearchParams({ limit: String(limit), include_archived: "1" });
  if (cursor) params.set("cursor", String(cursor));
  if (search.trim()) params.set("q", search.trim());
  return apiRequest<{ ok?: boolean; tasks?: HistoryTaskSummary[]; total?: number; next_cursor?: string | number | null; search?: string }>(
    `/api/training/history?${params.toString()}`,
    { signal },
  );
}

export function fetchHistoryTaskDetail(taskId: string, signal?: AbortSignal) {
  return apiRequest<HistoryTaskDetail>(
    `/api/training/history/${encodeURIComponent(taskId)}`,
    { signal },
  );
}

export function batchUpdateHistoryTasks(payload: HistoryBatchPayload) {
  return apiRequest<{ ok?: boolean; message?: string }>(
    "/api/training/history/batch",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

export type ResumeCheckpoint = {
  path: string;
  name: string;
  step?: number;
  remaining_steps?: number;
  target_total_steps?: number;
  estimate_error?: string;
  resume_available?: boolean;
  unavailable_reason?: string;
  state_complete?: boolean;
  state_integrity?: {
    ok?: boolean;
    missing?: string[];
    scheduler_required?: boolean;
    train_state?: boolean;
    model?: boolean;
    optimizer?: boolean;
    scheduler?: boolean;
    random_state?: boolean;
  };
};
export const fetchResumeOptions = (taskId: string, signal?: AbortSignal) =>
  apiRequest<{
    checkpoints: ResumeCheckpoint[];
    default_checkpoint: string;
    message: string;
  }>(`/api/training/history/${encodeURIComponent(taskId)}/resume-options`, {
    signal,
  });
export function resumeHistoryTask(
  taskId: string,
  checkpoint: string,
  queue: boolean,
  appendSteps?: number,
) {
  return apiRequest<{ ok: boolean; message?: string }>(
    queue ? "/api/training/queue/resume" : "/api/training/resume",
    {
      method: "POST",
      body: JSON.stringify({
        task_id: taskId,
        checkpoint,
        ...(appendSteps
          ? { duration_overrides: { max_train_steps: appendSteps } }
          : {}),
      }),
    },
  );
}
export type HistoryImage = {
  file: string;
  name: string;
  sample?: Record<string, unknown>;
  width?: number;
  height?: number;
};
export type HistoryImageListing = {
  images: HistoryImage[];
  count?: number;
  total?: number;
  offset?: number;
  next_offset?: number | null;
  directory_exists?: boolean;
  message?: string;
};
export type HistoryWeight = {
  file: string;
  name: string;
  size_bytes: number;
  scope_label: string;
};
export const fetchHistoryImages = (taskId: string, signal?: AbortSignal, limit = 120, offset = 0) =>
  apiRequest<HistoryImageListing>(
    `/api/preview/images?${new URLSearchParams({ task_id: taskId, source: "training", limit: String(limit), offset: String(offset) })}`,
    { signal },
  );
export const fetchHistoryWeights = (taskId: string, signal?: AbortSignal, limit = 200, offset = 0, sort = "recent") =>
  apiRequest<{ weights: HistoryWeight[]; count?: number; total?: number; next_offset?: number | null; directory_exists?: boolean; message?: string }>(
    `/api/preview/weights?${new URLSearchParams({ task_id: taskId, limit: String(limit), offset: String(offset), sort })}`,
    { signal },
  );
export const historyAssetUrl = (taskId: string, file: string, weight = false) =>
  `/api/preview/${weight ? "weight" : "image"}?${new URLSearchParams({ task_id: taskId, file })}`;
