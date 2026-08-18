import { useDroppable } from '@dnd-kit/core';
import { SortableContext, useSortable, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { ArrowDown, ArrowUp, GripVertical } from 'lucide-react';

import {
  datasetMoveTargets,
  insertPath,
  isDatasetMoveTarget,
  isMovableDatasetPreset,
  isSortableDatasetGroup,
  isSortableDatasetPreset,
  movePath,
} from './datasetOrdering';
import type { DatasetLibraryGroup, DatasetPresetSummary } from './types';

type Props = {
  group: DatasetLibraryGroup;
  groups: DatasetLibraryGroup[];
  selectedFile: string;
  searchActive: boolean;
  ordering: boolean;
  sortableGroupIndex: number;
  sortableGroupCount: number;
  presetDragging: boolean;
  onSelect: (file: string) => void;
  onGroupAction: (action: 'rename' | 'delete', group: DatasetLibraryGroup) => void;
  onPlaceGroup: (groupId: string, index: number) => void;
  onPlacePreset: (file: string, groupId: string, order: string[]) => void;
};

export function groupDragId(groupId: string) {
  return `group:${groupId}`;
}

export function presetDragId(path: string) {
  return `preset:${path}`;
}

export function groupDropId(groupId: string) {
  return `group-drop:${groupId}`;
}

export function datasetPresetName(preset: DatasetPresetSummary) {
  return preset.label || preset.filename || preset.path.split('/').pop() || preset.path;
}

export function SortableDatasetGroup({
  group,
  groups,
  selectedFile,
  searchActive,
  ordering,
  sortableGroupIndex,
  sortableGroupCount,
  presetDragging,
  onSelect,
  onGroupAction,
  onPlaceGroup,
  onPlacePreset,
}: Props) {
  const sortable = isSortableDatasetGroup(group, searchActive);
  const groupSort = useSortable({
    id: groupDragId(group.id),
    data: { type: 'group', groupId: group.id },
    disabled: !sortable || ordering,
  });
  const drop = useDroppable({
    id: groupDropId(group.id),
    data: { type: 'group-drop', groupId: group.id },
    disabled: !isDatasetMoveTarget(group) || ordering,
  });
  const style = {
    transform: CSS.Transform.toString(groupSort.transform),
    transition: groupSort.transition,
  };
  const sortDisabledReason = searchActive
    ? '搜索时不能调整分组顺序'
    : '该分组不能排序';

  return (
    <section
      ref={groupSort.setNodeRef}
      style={style}
      className="dataset-group"
      data-dragging={groupSort.isDragging}
    >
      <header>
        <div>
          <h3>{group.label}</h3>
          <span>{group.files.length} 个预设</span>
        </div>
        <div className="dataset-group-actions">
          {group.locked || group.group_locked ? <span className="badge">锁定</span> : null}
          <button
            ref={groupSort.setActivatorNodeRef}
            type="button"
            className="dataset-sort-button dataset-drag-handle"
            aria-label={`拖动排序分组 ${group.label}`}
            title={sortable ? '拖动或用键盘排序分组' : sortDisabledReason}
            disabled={!sortable || ordering}
            {...groupSort.attributes}
            {...groupSort.listeners}
          >
            <GripVertical size={15} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="dataset-sort-button"
            aria-label={`上移分组 ${group.label}`}
            title="上移分组"
            disabled={!sortable || sortableGroupIndex <= 0 || ordering}
            onClick={() => onPlaceGroup(group.id, sortableGroupIndex - 1)}
          >
            <ArrowUp size={15} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="dataset-sort-button"
            aria-label={`下移分组 ${group.label}`}
            title="下移分组"
            disabled={!sortable || sortableGroupIndex < 0 || sortableGroupIndex >= sortableGroupCount - 1 || ordering}
            onClick={() => onPlaceGroup(group.id, sortableGroupIndex + 1)}
          >
            <ArrowDown size={15} aria-hidden="true" />
          </button>
          {group.renamable ? (
            <button type="button" aria-label={`重命名分组 ${group.label}`} onClick={() => onGroupAction('rename', group)}>
              改名
            </button>
          ) : null}
          {group.deletable ? (
            <button type="button" className="danger-command" aria-label={`删除分组 ${group.label}`} onClick={() => onGroupAction('delete', group)}>
              删除
            </button>
          ) : null}
        </div>
      </header>
      <SortableContext items={group.files.map((preset) => presetDragId(preset.path))} strategy={verticalListSortingStrategy}>
        <div className="dataset-preset-list">
          {group.files.map((preset, index) => (
            <SortablePresetRow
              key={preset.path}
              group={group}
              groups={groups}
              preset={preset}
              index={index}
              selected={preset.path === selectedFile}
              searchActive={searchActive}
              ordering={ordering}
              onSelect={onSelect}
              onPlacePreset={onPlacePreset}
            />
          ))}
          <div
            ref={drop.setNodeRef}
            className="dataset-group-dropzone"
            data-over={drop.isOver}
            data-visible={presetDragging || group.files.length === 0}
          >
            {group.files.length === 0 ? '空分组，可将预设移到此处' : '拖到此组末尾'}
          </div>
        </div>
      </SortableContext>
    </section>
  );
}

type PresetRowProps = {
  group: DatasetLibraryGroup;
  groups: DatasetLibraryGroup[];
  preset: DatasetPresetSummary;
  index: number;
  selected: boolean;
  searchActive: boolean;
  ordering: boolean;
  onSelect: (file: string) => void;
  onPlacePreset: (file: string, groupId: string, order: string[]) => void;
};

function SortablePresetRow({
  group,
  groups,
  preset,
  index,
  selected,
  searchActive,
  ordering,
  onSelect,
  onPlacePreset,
}: PresetRowProps) {
  const sortable = isSortableDatasetPreset(group, preset, searchActive);
  const movable = isMovableDatasetPreset(group, preset);
  const targets = datasetMoveTargets(groups, group.id);
  const sort = useSortable({
    id: presetDragId(preset.path),
    data: { type: 'preset', groupId: group.id, file: preset.path },
    disabled: !sortable || ordering,
  });
  const name = datasetPresetName(preset);
  const paths = group.files.map((item) => item.path);
  const style = {
    transform: CSS.Transform.toString(sort.transform),
    transition: sort.transition,
  };
  const sortDisabledReason = searchActive
    ? '搜索时不能调整预设顺序'
    : '该预设不能排序';

  return (
    <div ref={sort.setNodeRef} style={style} className="dataset-preset-row" data-dragging={sort.isDragging}>
      <button
        type="button"
        className="dataset-preset"
        data-selected={selected}
        onClick={() => onSelect(preset.path)}
      >
        <span className="dataset-preset-title">
          <strong>{name}</strong>
          {preset.readonly ? <span className="badge">只读</span> : null}
        </span>
        <span className="dataset-preset-path">{preset.path}</span>
        <span className="dataset-preset-summary">
          {preset.summary?.dataset_count ?? 0} 组 · 重复 {preset.summary?.repeat_total ?? 0}
        </span>
      </button>
      {movable ? (
        <div className="dataset-preset-order-controls" aria-label={`${name} 排序与分组`}>
          <button
            ref={sort.setActivatorNodeRef}
            type="button"
            className="dataset-sort-button dataset-drag-handle"
            aria-label={`拖动排序预设 ${name}`}
            title={sortable ? '拖动或用键盘排序预设' : sortDisabledReason}
            disabled={!sortable || ordering}
            {...sort.attributes}
            {...sort.listeners}
          >
            <GripVertical size={15} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="dataset-sort-button"
            aria-label={`上移预设 ${name}`}
            title="上移预设"
            disabled={!sortable || index <= 0 || ordering}
            onClick={() => onPlacePreset(preset.path, group.id, movePath(paths, preset.path, index - 1))}
          >
            <ArrowUp size={15} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="dataset-sort-button"
            aria-label={`下移预设 ${name}`}
            title="下移预设"
            disabled={!sortable || index >= paths.length - 1 || ordering}
            onClick={() => onPlacePreset(preset.path, group.id, movePath(paths, preset.path, index + 1))}
          >
            <ArrowDown size={15} aria-hidden="true" />
          </button>
          <select
            className="dataset-move-select"
            aria-label={`移动 ${name} 到分组`}
            value=""
            disabled={!targets.length || ordering}
            onChange={(event) => {
              const target = groups.find((item) => item.id === event.target.value);
              if (target) {
                onPlacePreset(preset.path, target.id, insertPath(target.files.map((item) => item.path), preset.path));
              }
            }}
          >
            <option value="">移动到…</option>
            {targets.map((target) => <option key={target.id} value={target.id}>{target.label}</option>)}
          </select>
        </div>
      ) : null}
    </div>
  );
}
