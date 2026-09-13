import { useMutation, useQueryClient } from '@tanstack/react-query';

import { datasetKeys, placeDatasetGroup, placeDatasetPreset } from './api';
import type { DatasetLibraryResponse } from './types';

export type DatasetOrderingCommand =
  | { type: 'group'; groupId: string; index: number }
  | { type: 'preset'; file: string; groupId: string; order: string[] };

export function useDatasetLibraryOrdering(onNotice: (message: string) => void) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (command: DatasetOrderingCommand) => (
      command.type === 'group'
        ? placeDatasetGroup(command.groupId, command.index)
        : placeDatasetPreset(command.file, command.groupId, command.order)
    ),
    retry: false,
    onMutate: async (command) => {
      onNotice('');
      await queryClient.cancelQueries({ queryKey: datasetKeys.library() });
      const previous = queryClient.getQueryData<DatasetLibraryResponse>(datasetKeys.library());
      if (previous && command.type === 'preset') {
        const preset = previous.groups.flatMap((group) => group.files).find((file) => file.path === command.file);
        if (preset) queryClient.setQueryData<DatasetLibraryResponse>(datasetKeys.library(), {
          ...previous,
          groups: previous.groups.map((group) => {
            const remaining = group.files.filter((file) => file.path !== command.file);
            if (group.id !== command.groupId) return { ...group, files: remaining };
            const byPath = new Map([...remaining, preset].map((file) => [file.path, file]));
            return { ...group, files: command.order.flatMap((path) => byPath.has(path) ? [byPath.get(path)!] : []) };
          }),
        });
      }
      return { previous };
    },
    onError: (_error, _command, context) => {
      if (context?.previous) queryClient.setQueryData(datasetKeys.library(), context.previous);
    },
    onSuccess: async (result) => {
      onNotice(result.message || '预设库顺序已更新');
    },
    // Do not make the mutation pending for the duration of a full library refetch.
    // The optimistic ordering is already authoritative for this interaction; refresh
    // asynchronously so large libraries and cover queries cannot block dragging.
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: datasetKeys.library() });
    },
  });
}
