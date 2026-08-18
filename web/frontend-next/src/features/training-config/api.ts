import { apiRequest } from '../../api/client';
import type { TrainingConfigFile } from '../../api/trainingContext';

export type RawTrainingConfigResponse = {
  file: string;
  content: string;
  meta: TrainingConfigFile;
};

export type RawPatchResponse = {
  ok: true;
  file: string;
  message: string;
  content: string;
  changed: string[];
  warnings: string[];
};

export type RawSaveAsResponse = {
  ok: true;
  file: string;
  message: string;
  warnings: string[];
};

export type PreflightCheck = {
  level: 'ok' | 'warning' | 'error' | string;
  key: string;
  message: string;
  path?: string;
};

export type TrainingPreflightResponse = {
  ok: boolean;
  variant: string;
  preset: string;
  methods_subdir: string;
  summary: { errors: number; warnings: number; checks: number };
  checks: PreflightCheck[];
  errors: PreflightCheck[];
  warnings: PreflightCheck[];
};

export const trainingConfigKeys = {
  all: ['training-config'] as const,
  raw: (file: string) => [...trainingConfigKeys.all, 'raw', file] as const,
};

export function fetchRawTrainingConfig(file: string, signal?: AbortSignal) {
  const query = new URLSearchParams({ file });
  return apiRequest<RawTrainingConfigResponse>(`/api/config/raw?${query.toString()}`, { signal });
}

export function previewTrainingConfigPatch(file: string, values: Record<string, unknown>) {
  return apiRequest<RawPatchResponse>('/api/config/raw/patch-preview', {
    method: 'POST',
    body: JSON.stringify({ file, values }),
  });
}

export function saveTrainingConfigPatch(file: string, values: Record<string, unknown>) {
  return apiRequest<RawPatchResponse>('/api/config/raw', {
    method: 'PATCH',
    body: JSON.stringify({ file, values }),
  });
}

export function saveTrainingConfigAs(file: string, content: string) {
  return apiRequest<RawSaveAsResponse>('/api/config/raw/save-as', {
    method: 'POST',
    body: JSON.stringify({ file, content }),
  });
}

export async function runTrainingPreflight(file: TrainingConfigFile, preset: string) {
  const response = await fetch('/api/training/preflight', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      variant: file.method || 'lora',
      preset,
      methods_subdir: file.methods_subdir || 'gui-methods',
      config_file: file.path,
    }),
  });
  const payload = await response.json() as TrainingPreflightResponse & { error?: string };
  if (!response.ok) throw new Error(payload.error || payload.errors?.[0]?.message || '训练预检测请求失败');
  return payload;
}
