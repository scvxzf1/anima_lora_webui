import { apiRequest } from "../../api/client";
import type { HistoryTaskSummary } from "../training-history/api";

export type PreviewSource = "training" | "inference" | "custom";
export type PreviewSettings = {
  training_dir?: string;
  inference_dir?: string;
  custom_dir?: string;
  effective_training_dir?: string;
  defaults?: Record<string, string>;
  revision?: string;
};
export type PreviewImage = {
  file: string;
  name: string;
  url?: string;
  width?: number;
  height?: number;
  size_bytes?: number;
  sample?: Record<string, unknown>;
  source_task?: { label?: string };
};
export type PreviewWeight = {
  file: string;
  abs_path?: string;
  name: string;
  size_bytes?: number;
  epoch?: number;
  steps?: number;
  scope_label?: string;
  mtime_text?: string;
  download_url?: string;
  source_task?: { id?: string };
};
export type PreviewGroup = {
  key: string;
  historyGroupKey: string;
  methodsSubdir: string;
  variant: string;
  preset: string;
  label: string;
  tasks: HistoryTaskSummary[];
};

export const previewKeys = {
  settings: ["preview-workspace", "settings"] as const,
  tasks: ["preview-workspace", "tasks"] as const,
  assets: (source: PreviewSource, scope: string, days: string) =>
    ["preview-workspace", "assets", source, scope, days] as const,
};

export const fetchPreviewSettings = (signal?: AbortSignal) =>
  apiRequest<PreviewSettings>("/api/preview/settings", { signal });

export const savePreviewSettings = (settings: Pick<PreviewSettings, "training_dir" | "inference_dir" | "custom_dir" | "revision">) =>
  apiRequest<PreviewSettings & { ok?: boolean; message?: string }>("/api/preview/settings", {
    method: "PUT", body: JSON.stringify(settings),
  });

export const fetchPreviewTasks = (cursor = "", signal?: AbortSignal) => {
  const params = new URLSearchParams({ limit: "100", include_archived: "1" });
  if (cursor) params.set("cursor", cursor);
  return apiRequest<{ tasks?: HistoryTaskSummary[]; next_cursor?: string | null }>(`/api/training/history?${params}`, { signal });
};

export function makePreviewGroups(tasks: HistoryTaskSummary[]): PreviewGroup[] {
  const groups = new Map<string, PreviewGroup>();
  for (const task of tasks) {
    if (task.job !== "training") continue;
    const historyGroupKey = String(task.history_group_key || "").trim();
    const methodsSubdir = String(task.methods_subdir || "").trim();
    const variant = String(task.variant || "").trim();
    const preset = String(task.preset || "default").trim() || "default";
    const key = historyGroupKey || `${methodsSubdir}::${variant}::${preset}`;
    if (!key || (!historyGroupKey && (!methodsSubdir || !variant))) continue;
    const group = groups.get(key) || {
      key, historyGroupKey, methodsSubdir, variant, preset,
      label: task.history_group_label || task.history_source_config_file || `${methodsSubdir} / ${variant} / ${preset}`,
      tasks: [],
    };
    group.tasks.push(task);
    groups.set(key, group);
  }
  return [...groups.values()].sort((a, b) => a.label.localeCompare(b.label, "zh-CN"));
}

function scopeParams(scope: string, taskId: string, group?: PreviewGroup) {
  const params = new URLSearchParams();
  if (scope === "task" && taskId) params.set("task_id", taskId);
  if (scope === "group" && group) {
    params.set("mode", "config_group");
    if (group.historyGroupKey) params.set("group_key", group.historyGroupKey);
    params.set("methods_subdir", group.methodsSubdir);
    params.set("variant", group.variant);
    params.set("preset", group.preset);
  }
  return params;
}

export function fetchPreviewImages(source: PreviewSource, scope: string, taskId: string, group: PreviewGroup | undefined, days: string, pageParam: number | string = 0, signal?: AbortSignal) {
  const params = scopeParams(scope, taskId, group);
  params.set("source", source);
  params.set("limit", "200");
  params.set("days", days);
  if (scope === "group") {
    if (pageParam) params.set("cursor", String(pageParam));
  } else if (pageParam) params.set("offset", String(pageParam));
  return apiRequest<{ images: PreviewImage[]; count?: number; total?: number; next_offset?: number | null; next_cursor?: string | null; directory?: string; message?: string }>(
    `/api/preview/images?${params}`, { signal },
  );
}

export function fetchPreviewWeights(scope: string, taskId: string, group: PreviewGroup | undefined, offset = 0, signal?: AbortSignal) {
  const params = scopeParams(scope, taskId, group);
  if (scope !== "group") {
    params.set("limit", "100");
    params.set("sort", "recent");
    if (offset) params.set("offset", String(offset));
  }
  return apiRequest<{ weights: PreviewWeight[]; total?: number; next_offset?: number | null; truncated?: boolean; directory?: string; message?: string }>(`/api/preview/weights?${params}`, { signal });
}

export function previewImageUrl(file: string, taskId?: string) {
  const params = new URLSearchParams({ file });
  if (taskId) params.set("task_id", taskId);
  return `/api/preview/image?${params}`;
}

export function previewWeightUrl(file: string, taskId?: string) {
  const params = new URLSearchParams({ file });
  if (taskId) params.set("task_id", taskId);
  return `/api/preview/weight?${params}`;
}

export function deletePreviewImages(source: PreviewSource, files: string[], taskId?: string) {
  const params = new URLSearchParams();
  if (source === "training" && taskId) params.set("task_id", taskId);
  return apiRequest<{ ok: boolean; deleted_count?: number; blocked_count?: number; message?: string }>(
    `/api/preview/images${params.size ? `?${params}` : ""}`,
    { method: "DELETE", body: JSON.stringify({ source, files }) },
  );
}
