import { apiRequest } from "../../api/client";
import type { DatasetPreviewResponse } from "../dataset-editor/types";

const ROOT = "/api/captioning";
export const captioningKeys = {
  profiles: ["captioning", "profiles"] as const,
  jobs: ["captioning", "jobs"] as const,
  prompts: ["captioning", "prompts"] as const,
  job: (id: string) => ["captioning", "job", id] as const,
};
export type CaptionProfile = {
  id: string;
  name: string;
  provider: string;
  kind: string;
  available: boolean;
  status: string;
  config: Record<string, unknown>;
  api_key_hint: string;
  api_key_configured: boolean;
};
export type ProviderType = { id: string; label: string; kind: string };
export type Profiles = {
  profiles: CaptionProfile[];
  active_profile_id: string;
  provider_types: ProviderType[];
};
export type CaptionItem = {
  id: string;
  name: string;
  file: string;
  url: string;
  thumbnail_url?: string;
  state: string;
  caption: string;
  proposed_caption: string;
  error?: string;
  commit_error?: string;
};
export type CaptionJob = {
  id: string;
  state: string;
  settings?: { provider?: string };
  dataset_file: string;
  dataset_index: number;
  profile_name: string;
  profile_id: string;
  total: number;
  completed: number;
  failed: number;
  created_at_text?: string;
  items: CaptionItem[];
  error?: string;
};
export type JobSnapshot = { ok: true; job: CaptionJob };
export type PromptPreset = {
  id: string;
  name: string;
  system_prompt: string;
  user_prompt: string;
  builtin?: boolean;
};
export type CreateJob = {
  dataset_file: string;
  dataset_index: number;
  source: string;
  profile_id: string;
  user_prompt: string;
  system_prompt: string;
  items: { file: string }[];
};
export const captionActive = (state?: string) =>
  state === "queued" || state === "running";
function post<T>(path: string, body?: unknown, method = "POST") {
  return apiRequest<T>(ROOT + path, {
    method,
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
}
export const fetchCaptionProfiles = (signal?: AbortSignal) =>
  apiRequest<Profiles>(`${ROOT}/profiles`, { signal });
export const fetchCaptionJobs = (signal?: AbortSignal) =>
  apiRequest<{ jobs: CaptionJob[] }>(`${ROOT}/jobs`, { signal });
export const clearCaptionJobs = (job_ids: string[]) =>
  post<{ removed: string[]; skipped: string[] }>("/jobs/cleanup", { job_ids });
export const fetchCaptionJob = (id: string, signal?: AbortSignal) =>
  apiRequest<JobSnapshot>(`${ROOT}/jobs/${encodeURIComponent(id)}`, { signal });
export const createCaptionJob = (payload: CreateJob) =>
  post<JobSnapshot>("/jobs", payload);
export const cancelCaptionJob = (id: string) =>
  post<JobSnapshot>(`/jobs/${encodeURIComponent(id)}/cancel`);
export const rerunCaptionJob = (
  id: string,
  profile_id: string,
  item_ids: string[],
) =>
  post<JobSnapshot>(`/jobs/${encodeURIComponent(id)}/rerun`, {
    profile_id,
    item_ids,
  });
export const updateCaptionItem = (
  job: string,
  id: string,
  proposed_caption: string,
) =>
  post<JobSnapshot>(
    `/jobs/${encodeURIComponent(job)}/items/${encodeURIComponent(id)}`,
    { proposed_caption },
    "PATCH",
  );
export const commitCaptionJob = (job: string, item_ids: string[]) =>
  post<{
    written: number;
    conflicts: number;
    skipped: number;
    errors: { file: string; error: string }[];
    job: CaptionJob;
  }>(`/jobs/${encodeURIComponent(job)}/commit`, { item_ids });
export const fetchCaptionPrompts = (signal?: AbortSignal) =>
  apiRequest<{ presets: PromptPreset[] }>(`${ROOT}/prompt-presets`, { signal });
export const saveCaptionPrompt = (
  payload: Omit<PromptPreset, "id">,
  id?: string,
) =>
  post<{ presets: PromptPreset[]; preset: PromptPreset }>(
    `/prompt-presets${id ? `/${encodeURIComponent(id)}` : ""}`,
    payload,
    id ? "PUT" : "POST",
  );
export const deleteCaptionPrompt = (id: string) =>
  post<{ presets: PromptPreset[] }>(
    `/prompt-presets/${encodeURIComponent(id)}`,
    undefined,
    "DELETE",
  );
export const saveCaptionProfile = (
  payload: Record<string, unknown>,
  id?: string,
) =>
  post<Profiles>(
    `/profiles${id ? `/${encodeURIComponent(id)}` : ""}`,
    payload,
    id ? "PUT" : "POST",
  );
export const activateCaptionProfile = (id: string) =>
  post<Profiles>(`/profiles/${encodeURIComponent(id)}/activate`);
export const deleteCaptionProfile = (id: string) =>
  post<Profiles>(`/profiles/${encodeURIComponent(id)}`, undefined, "DELETE");
export const testCaptionProvider = (profile_id: string) =>
  post<{ ok: boolean; message?: string }>("/test", {
    mode: "ping",
    profile_id,
  });
export const fetchCaptionLogs = (job: string, signal?: AbortSignal) =>
  apiRequest<{ lines: { sequence: number; message: string; level: string }[] }>(
    `${ROOT}/logs?${new URLSearchParams({ job_id: job, limit: "300" })}`,
    { signal },
  );
export const scanCaptionImages = (
  file: string,
  datasetIndex: number,
  source: string,
  offset: number,
  signal?: AbortSignal,
) =>
  apiRequest<DatasetPreviewResponse>(
    `/api/config/dataset-presets/images?${new URLSearchParams({ file, dataset_index: String(datasetIndex), source, limit: "60", offset: String(offset) })}`,
    { signal },
  );

export function captionImageUrl(value: string) {
  try {
    const url = new URL(value, window.location.origin);
    return url.origin === window.location.origin &&
      url.pathname.startsWith("/api/config/dataset-presets/")
      ? url.pathname + url.search
      : "";
  } catch {
    return "";
  }
}
