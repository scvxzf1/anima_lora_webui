import { queryOptions } from '@tanstack/react-query';

import { apiRequest } from '../../api/client';
import type {
  ApplyDatasetPresetResponse,
  CreateDatasetGroupResponse,
  DatasetPresetMutationResponse,
  DatasetPreviewResponse,
  DatasetPresetWritePayload,
  DeleteDatasetPresetResponse,
  DatasetLibraryResponse,
  DatasetPresetResponse,
  DeleteDatasetGroupResponse,
  PlaceDatasetLibraryItemResponse,
  RenameDatasetGroupResponse,
} from './types';

export const datasetKeys = {
  all: ['datasets'] as const,
  library: () => [...datasetKeys.all, 'library'] as const,
  preset: (file: string) => [...datasetKeys.all, 'preset', file] as const,
  preview: (file: string, datasetIndex: number) => (
    [...datasetKeys.all, 'preview', file, datasetIndex] as const
  ),
};

export function fetchDatasetLibrary(signal?: AbortSignal) {
  return apiRequest<DatasetLibraryResponse>('/api/config/dataset-presets', { signal });
}

export function fetchDatasetPreset(file: string, signal?: AbortSignal) {
  const query = new URLSearchParams({ file });
  return apiRequest<DatasetPresetResponse>(
    `/api/config/dataset-presets/read?${query.toString()}`,
    { signal },
  );
}

export function fetchDatasetPresetImages(
  file: string,
  datasetIndex: number,
  signal?: AbortSignal,
) {
  const query = new URLSearchParams({
    file,
    dataset_index: String(datasetIndex),
    source: 'source',
    limit: '120',
  });
  return apiRequest<DatasetPreviewResponse>(
    `/api/config/dataset-presets/images?${query.toString()}`,
    { signal },
  );
}

export function createDatasetGroup(label: string) {
  return apiRequest<CreateDatasetGroupResponse>('/api/config/file-groups', {
    method: 'POST',
    body: JSON.stringify({ label, kind: 'dataset' }),
  });
}

export function renameDatasetGroup(groupId: string, label: string) {
  return apiRequest<RenameDatasetGroupResponse>(
    `/api/config/file-groups/${encodeURIComponent(groupId)}`,
    {
      method: 'PATCH',
      body: JSON.stringify({ label }),
    },
  );
}

export function deleteDatasetGroup(groupId: string) {
  return apiRequest<DeleteDatasetGroupResponse>(
    `/api/config/file-groups/${encodeURIComponent(groupId)}`,
    { method: 'DELETE' },
  );
}

export function placeDatasetPreset(file: string, groupId: string, order: string[]) {
  return apiRequest<PlaceDatasetLibraryItemResponse>('/api/config/file-groups/place', {
    method: 'POST',
    body: JSON.stringify({ target: 'file', file, group: groupId, order }),
  });
}

export function placeDatasetGroup(groupId: string, index: number) {
  return apiRequest<PlaceDatasetLibraryItemResponse>('/api/config/file-groups/place', {
    method: 'POST',
    body: JSON.stringify({ target: 'group', group: groupId, scope: 'dataset', index }),
  });
}

export function saveDatasetPreset(
  file: string,
  payload: DatasetPresetWritePayload,
  overwrite: boolean,
) {
  return apiRequest<DatasetPresetMutationResponse>('/api/config/dataset-presets', {
    method: 'PUT',
    body: JSON.stringify({ file, ...payload, overwrite }),
  });
}

export function saveDatasetPresetAs(name: string, payload: DatasetPresetWritePayload) {
  return apiRequest<DatasetPresetMutationResponse>('/api/config/dataset-presets/save-as', {
    method: 'POST',
    body: JSON.stringify({ name, ...payload }),
  });
}

export function importDatasetPreset(name: string, content: string, overwrite = false) {
  return apiRequest<DatasetPresetMutationResponse>('/api/config/dataset-presets/import', {
    method: 'POST',
    body: JSON.stringify({ name, content, overwrite }),
  });
}

export function deleteDatasetPreset(file: string) {
  const query = new URLSearchParams({ file });
  return apiRequest<DeleteDatasetPresetResponse>(
    `/api/config/dataset-presets?${query.toString()}`,
    { method: 'DELETE' },
  );
}

export function applyDatasetPreset(datasetFile: string, trainFile: string) {
  return apiRequest<ApplyDatasetPresetResponse>('/api/config/dataset-presets/apply', {
    method: 'POST',
    body: JSON.stringify({ dataset_file: datasetFile, train_file: trainFile }),
  });
}

export const datasetLibraryQuery = queryOptions({
  queryKey: datasetKeys.library(),
  queryFn: ({ signal }) => fetchDatasetLibrary(signal),
});

export function datasetPresetQuery(file: string) {
  return queryOptions({
    queryKey: datasetKeys.preset(file),
    queryFn: ({ signal }) => fetchDatasetPreset(file, signal),
  });
}
