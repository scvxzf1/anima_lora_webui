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
