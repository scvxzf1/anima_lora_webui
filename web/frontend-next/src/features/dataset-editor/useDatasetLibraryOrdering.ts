import { useMutation, useQueryClient } from '@tanstack/react-query';

import { datasetKeys, placeDatasetGroup, placeDatasetPreset } from './api';

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
    onMutate: () => onNotice(''),
    onSuccess: async (result) => {
      onNotice(result.message || '预设库顺序已更新');
      await queryClient.invalidateQueries({ queryKey: datasetKeys.library() });
    },
  });
}
