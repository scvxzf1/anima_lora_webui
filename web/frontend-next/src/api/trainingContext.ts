import { apiRequest } from './client';

export type TrainingConfigFile = {
  path: string;
  label?: string;
  filename?: string;
  method?: string;
  methods_subdir?: string;
  trainable?: boolean;
  readonly?: boolean;
  locked?: boolean;
};

export type TrainingConfigGroup = {
  id: string;
  label: string;
  kind?: string;
  methods_subdir?: string;
  files: TrainingConfigFile[];
};

export type TrainingPresets = string[];

export type MergedTrainingConfig = Record<string, unknown> & {
  max_train_steps?: number;
};

export const trainingContextKeys = {
  all: ['training-context'] as const,
  files: () => [...trainingContextKeys.all, 'files'] as const,
  presets: () => [...trainingContextKeys.all, 'presets'] as const,
  merged: (file: string, preset: string) => (
    [...trainingContextKeys.all, 'merged', file, preset] as const
  ),
};

export function fetchTrainingConfigGroups(signal?: AbortSignal) {
  return apiRequest<TrainingConfigGroup[]>('/api/config/file-groups?kind=training', { signal });
}

export function fetchTrainingPresets(signal?: AbortSignal) {
  return apiRequest<TrainingPresets>('/api/presets', { signal });
}

export function fetchMergedTrainingConfig(
  file: TrainingConfigFile,
  preset: string,
  signal?: AbortSignal,
) {
  const query = new URLSearchParams({
    variant: file.method || 'lora',
    preset: preset || 'default',
    methods_subdir: file.methods_subdir || 'gui-methods',
    config_file: file.path,
  });
  return apiRequest<MergedTrainingConfig>(`/api/config/merged?${query.toString()}`, { signal });
}
