import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef } from "react";

import {
  fetchMergedTrainingConfig,
  fetchTrainingConfigGroups,
  fetchTrainingPresets,
  trainingContextKeys,
} from "../api/trainingContext";
import { useTrainingContextStore } from "./trainingContextStore";

export function useTrainingContext(options: { loadMergedConfig?: boolean; retryMergedConfig?: boolean; deferFallback?: boolean } = {}) {
  const loadMergedConfig = options.loadMergedConfig ?? true;
  const selection = useTrainingContextStore();
  const groupsQuery = useQuery({
    queryKey: trainingContextKeys.files(),
    queryFn: ({ signal }) => fetchTrainingConfigGroups(signal),
  });
  const presetsQuery = useQuery({
    queryKey: trainingContextKeys.presets(),
    queryFn: ({ signal }) => fetchTrainingPresets(signal),
  });
  const files = useMemo(
    () =>
      (groupsQuery.data || []).flatMap((group) =>
        (group.files || [])
          .filter((file) => file.trainable !== false)
          .map((file) => ({
            ...file,
            methods_subdir: file.methods_subdir || group.methods_subdir,
          })),
      ),
    [groupsQuery.data],
  );
  const presets = presetsQuery.data || [];
  const listedSelectedFile = files.find((file) => file.path === selection.configFile);
  const lastSelectedFile = useRef(listedSelectedFile);
  if (listedSelectedFile) lastSelectedFile.current = listedSelectedFile;
  const selectedFileAvailable = Boolean(listedSelectedFile);
  const selectedFile =
    listedSelectedFile ||
    (options.deferFallback && lastSelectedFile.current?.path === selection.configFile
      ? lastSelectedFile.current
      : undefined) ||
    (!options.deferFallback
      ? files.find(
          (file) => file.path === "configs/imported/lora.toml" && !file.locked,
        ) ||
        files.find((file) => !file.locked) ||
        files.find((file) => file.path === "configs/gui-methods/lora.toml") ||
        files[0]
      : undefined);
  const selectedPreset = options.deferFallback && !presets.includes(selection.preset)
    ? selection.preset
    : presets.includes(selection.preset)
    ? selection.preset
    : presets.includes("default")
      ? "default"
      : presets[0] || "default";

  useEffect(() => {
    if (!options.deferFallback && selectedFile && selectedFile.path !== selection.configFile) {
      selection.selectConfigFile(selectedFile.path);
    }
  }, [options.deferFallback, selectedFile?.path, selection.configFile, selection.selectConfigFile]);

  useEffect(() => {
    if (options.deferFallback) return;
    if (selectedPreset !== selection.preset)
      selection.selectPreset(selectedPreset);
  }, [options.deferFallback, selectedPreset, selection.preset, selection.selectPreset]);

  const mergedQuery = useQuery({
    queryKey: trainingContextKeys.merged(
      selectedFile?.path || "",
      selectedPreset,
    ),
    queryFn: ({ signal }) =>
      fetchMergedTrainingConfig(selectedFile!, selectedPreset, signal),
    enabled: loadMergedConfig && Boolean(selectedFile),
    retry: options.retryMergedConfig,
  });

  return {
    groups: groupsQuery.data || [],
    files,
    presets,
    selectedFile,
    selectedFileId: selection.configFile,
    selectedFileAvailable,
    selectedPreset,
    selectConfigFile: selection.selectConfigFile,
    selectPreset: selection.selectPreset,
    mergedConfig: mergedQuery.data,
    mergedConfigPending: mergedQuery.isPending || mergedQuery.isFetching,
    mergedConfigUpdatedAt: mergedQuery.dataUpdatedAt,
    mergedConfigIsFetching: mergedQuery.isFetching,
    listsRefetching: groupsQuery.isFetching || presetsQuery.isFetching,
    listsPending: groupsQuery.isPending || presetsQuery.isPending,
    refetchFiles: groupsQuery.refetch,
    refetchPresets: presetsQuery.refetch,
    mergedConfigError: mergedQuery.error,
    refetchMergedConfig: mergedQuery.refetch,
    maxTrainSteps: positiveSteps(mergedQuery.data?.max_train_steps),
    isPending:
      groupsQuery.isPending || presetsQuery.isPending || (loadMergedConfig && mergedQuery.isPending),
    error: groupsQuery.error || presetsQuery.error || (loadMergedConfig && mergedQuery.error),
  };
}

function positiveSteps(value: unknown) {
  const steps = Math.round(Number(value) || 0);
  return steps > 0 ? steps : 0;
}

export type TrainingContextController = ReturnType<typeof useTrainingContext>;
