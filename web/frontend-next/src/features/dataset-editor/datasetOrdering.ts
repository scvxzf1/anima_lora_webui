import type { DatasetLibraryGroup, DatasetPresetSummary } from './types';

export function isSortableDatasetGroup(group: DatasetLibraryGroup, searchActive: boolean) {
  return Boolean(
    !searchActive
    && group.id
    && group.id !== 'unfiled_datasets'
    && group.kind === 'dataset'
    && group.movable
    && !group.locked
    && !group.group_locked
    && !group.user_group_locked
    && !group.system_locked,
  );
}

export function isSortableDatasetPreset(
  group: DatasetLibraryGroup,
  preset: DatasetPresetSummary,
  searchActive: boolean,
) {
  return !searchActive && isMovableDatasetPreset(group, preset);
}

export function isMovableDatasetPreset(group: DatasetLibraryGroup, preset: DatasetPresetSummary) {
  return Boolean(
    !preset.readonly
    && group.movable
    && !group.locked
    && !group.group_locked
    && !group.user_group_locked
    && !group.system_locked,
  );
}

export function isDatasetMoveTarget(group: DatasetLibraryGroup) {
  return Boolean(
    group.id
    && group.kind === 'dataset'
    && group.movable
    && !group.locked
    && !group.group_locked
    && !group.user_group_locked
    && !group.system_locked
  );
}

export function datasetMoveTargets(groups: DatasetLibraryGroup[], currentGroupId: string) {
  return groups.filter((group) => group.id !== currentGroupId && isDatasetMoveTarget(group));
}

export function movePath(paths: string[], path: string, nextIndex: number) {
  const currentIndex = paths.indexOf(path);
  if (currentIndex < 0) return paths;
  const boundedIndex = Math.max(0, Math.min(nextIndex, paths.length - 1));
  if (boundedIndex === currentIndex) return paths;
  const next = [...paths];
  next.splice(currentIndex, 1);
  next.splice(boundedIndex, 0, path);
  return next;
}

export function insertPath(paths: string[], path: string, index = paths.length) {
  const next = paths.filter((item) => item !== path);
  next.splice(Math.max(0, Math.min(index, next.length)), 0, path);
  return next;
}
