import { apiRequest } from "../../api/client";

export type RuntimeDeleteBlocker = {
  id?: string;
  path?: string;
  reason: string;
};
export type RuntimeDeleteTask = {
  id: string;
  name?: string;
  job?: string;
  state?: string;
  started_at_text?: string;
  run_dir?: string;
  output_dir?: string;
};
export type RuntimeDeleteDir = {
  path: string;
  status: "ready" | "missing" | string;
};
export type RuntimeDeletePreview = {
  ok: true;
  dry_run: true;
  tasks: RuntimeDeleteTask[];
  runtime_dirs: RuntimeDeleteDir[];
  blocked: RuntimeDeleteBlocker[];
  task_count: number;
  runtime_dir_count: number;
};
export type RuntimeDeleteResult = {
  ok: true;
  dry_run: false;
  deleted_task_ids?: string[];
  deleted_runtime_dirs?: string[];
  runtime_cleanup_errors?: Record<string, string>;
  preview?: RuntimeDeletePreview;
  message?: string;
};

const endpoint = "/api/training/history/batch";

export function previewRuntimeDelete(taskIds: string[]) {
  return apiRequest<RuntimeDeletePreview>(endpoint, {
    method: "POST",
    body: JSON.stringify({
      action: "delete",
      task_ids: taskIds,
      delete_runtime_dirs: true,
      dry_run: true,
    }),
  });
}

export function confirmRuntimeDelete(taskIds: string[]) {
  return apiRequest<RuntimeDeleteResult>(endpoint, {
    method: "POST",
    body: JSON.stringify({
      action: "delete",
      task_ids: taskIds,
      delete_runtime_dirs: true,
      confirmed: true,
    }),
  });
}
