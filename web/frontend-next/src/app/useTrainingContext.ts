import { useQuery } from '@tanstack/react-query';
import { useEffect, useMemo } from 'react';

import {
  fetchMergedTrainingConfig,
  fetchTrainingConfigGroups,
  fetchTrainingPresets,
  trainingContextKeys,
} from '../api/trainingContext';
import { useTrainingContextStore } from './trainingContextStore';

export function useTrainingContext() {
  const selection = useTrainingContextStore();
  const groupsQuery = useQuery({
    queryKey: trainingContextKeys.files(),
    queryFn: ({ signal }) => fetchTrainingConfigGroups(signal),
  });
  const presetsQuery = useQuery({
    queryKey: trainingContextKeys.presets(),
    queryFn: ({ signal }) => fetchTrainingPresets(signal),
  });
  const files = useMemo(() => (groupsQuery.data || []).flatMap((group) => (
    (group.files || [])
      .filter((file) => file.trainable !== false)
      .map((file) => ({
        ...file,
        methods_subdir: file.methods_subdir || group.methods_subdir,
      }))
  )), [groupsQuery.data]);
  const presets = presetsQuery.data || [];
  const selectedFile = files.find((file) => file.path === selection.configFile)
    || files.find((file) => file.path === 'configs/imported/lora.toml' && !file.locked)
    || files.find((file) => !file.locked)
    || files.find((file) => file.path === 'configs/gui-methods/lora.toml')
    || files[0];
  const selectedPreset = presets.includes(selection.preset)
    ? selection.preset
    : presets.includes('default') ? 'default' : presets[0] || 'default';

  useEffect(() => {
    if (selectedFile && selectedFile.path !== selection.configFile) {
      selection.selectConfigFile(selectedFile.path);
    }
  }, [selectedFile?.path, selection.configFile, selection.selectConfigFile]);

  useEffect(() => {
    if (selectedPreset !== selection.preset) selection.selectPreset(selectedPreset);
  }, [selectedPreset, selection.preset, selection.selectPreset]);

  const mergedQuery = useQuery({
    queryKey: trainingContextKeys.merged(selectedFile?.path || '', selectedPreset),
    queryFn: ({ signal }) => fetchMergedTrainingConfig(selectedFile!, selectedPreset, signal),
    enabled: Boolean(selectedFile),
  });

  return {
    files,
    presets,
    selectedFile,
    selectedPreset,
    selectConfigFile: selection.selectConfigFile,
    selectPreset: selection.selectPreset,
    mergedConfig: mergedQuery.data,
    maxTrainSteps: positiveSteps(mergedQuery.data?.max_train_steps),
    isPending: groupsQuery.isPending || presetsQuery.isPending || mergedQuery.isPending,
    error: groupsQuery.error || presetsQuery.error || mergedQuery.error,
  };
}

function positiveSteps(value: unknown) {
  const steps = Math.round(Number(value) || 0);
  return steps > 0 ? steps : 0;
}

export type TrainingContextController = ReturnType<typeof useTrainingContext>;
