import { apiRequest } from "../../api/client";

export type GlobalSettings = Record<string, unknown> & {
  ok?: boolean;
  message?: string;
  requires_reload?: boolean;
  revision?: string;
  defaults?: Record<string, unknown>;
  path_overrides?: Record<string, string>;
  effective_paths?: Record<string, string>;
};
export type ModelConfigItem = {
  id: string;
  name: string;
  model_family: string;
  pretrained_model_name_or_path: string;
  qwen3: string;
  vae: string;
  complete?: boolean;
  training_tasks?: string[];
  capability_labels?: string[];
};
export type ModelConfigResponse = {
  ok?: boolean;
  items: ModelConfigItem[];
  default_id: string;
  revision: string;
  groups?: { id: string; label: string; item_ids: string[] }[];
};

export type ModelPathStatus = {
  path: string;
  kind: "empty" | "file" | "directory" | "missing";
  size_bytes?: number;
};

export type SettingsDirectoryEntry = {
  name: string;
  path: string;
  kind: "directory";
};

export type SettingsDirectoryPage = {
  ok?: boolean;
  roots: { name: string; path: string }[];
  directory: string;
  parent: string | null;
  entries: SettingsDirectoryEntry[];
  offset: number;
  next_offset: number | null;
  total: number;
  truncated: boolean;
};

export const settingsKeys = {
  global: ["settings", "global"] as const,
  models: ["settings", "models"] as const,
};

export const fetchGlobalSettings = (signal?: AbortSignal) =>
  apiRequest<GlobalSettings>("/api/settings/global", { signal });
export const saveGlobalSettings = (payload: Record<string, unknown>) =>
  apiRequest<GlobalSettings>("/api/settings/global", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
export const fetchModelConfigs = (signal?: AbortSignal) =>
  apiRequest<ModelConfigResponse>("/api/settings/model-configs", { signal });
export const saveModelConfigs = (payload: ModelConfigResponse) =>
  apiRequest<ModelConfigResponse>("/api/settings/model-configs", {
    method: "PUT",
    body: JSON.stringify(payload),
  });

export const fetchModelPathStatus = (path: string, signal?: AbortSignal) =>
  apiRequest<ModelPathStatus>(
    `/api/settings/model-path-status?path=${encodeURIComponent(path)}`,
    { signal },
  );

export const fetchSettingsDirectories = (
  directory?: string,
  offset = 0,
  signal?: AbortSignal,
) => {
  const params = new URLSearchParams({ offset: String(offset), limit: "100" });
  if (directory) params.set("directory", directory);
  return apiRequest<SettingsDirectoryPage>(
    `/api/settings/directories?${params.toString()}`,
    { signal },
  );
};
