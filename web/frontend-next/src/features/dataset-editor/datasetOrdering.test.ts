import { describe, expect, it } from 'vitest';

import {
  datasetMoveTargets,
  insertPath,
  isSortableDatasetGroup,
  isSortableDatasetPreset,
  movePath,
} from './datasetOrdering';
import type { DatasetLibraryGroup, DatasetPresetSummary } from './types';

const preset: DatasetPresetSummary = { path: 'configs/datasets/alpha.toml' };
const group: DatasetLibraryGroup = {
  id: 'characters',
  label: '角色',
  kind: 'dataset',
  movable: true,
  files: [preset],
};

describe('dataset ordering domain', () => {
  it('disables drag sorting during search and for the unfiled group', () => {
    expect(isSortableDatasetGroup(group, false)).toBe(true);
    expect(isSortableDatasetGroup(group, true)).toBe(false);
    expect(isSortableDatasetGroup({ ...group, id: 'unfiled_datasets' }, false)).toBe(false);
    expect(isSortableDatasetPreset(group, preset, true)).toBe(false);
  });

  it('allows unfiled datasets as an explicit move target when it is writable', () => {
    const unfiled = { ...group, id: 'unfiled_datasets', label: '未分组', files: [] };
    expect(datasetMoveTargets([group, unfiled], group.id).map((item) => item.id)).toEqual(['unfiled_datasets']);
  });

  it('moves and inserts paths without duplicates', () => {
    expect(movePath(['a', 'b', 'c'], 'a', 2)).toEqual(['b', 'c', 'a']);
    expect(insertPath(['a', 'b'], 'a', 1)).toEqual(['b', 'a']);
    expect(insertPath(['a'], 'b')).toEqual(['a', 'b']);
  });
});
