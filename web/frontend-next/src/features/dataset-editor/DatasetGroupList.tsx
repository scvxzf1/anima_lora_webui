import {
  closestCenter,
  DndContext,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragStartEvent,
} from '@dnd-kit/core';
import { SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { useState } from 'react';

import { insertPath, isSortableDatasetGroup, movePath } from './datasetOrdering';
import {
  groupDragId,
  SortableDatasetGroup,
  datasetPresetName,
} from './SortableDatasetGroup';
import type { DatasetLibraryGroup } from './types';

type Props = {
  groups: DatasetLibraryGroup[];
  pending: boolean;
  selectedFile: string;
  searchActive: boolean;
  ordering: boolean;
  orderingError?: string;
  onSelect: (file: string) => void;
  onGroupAction: (action: 'rename' | 'delete', group: DatasetLibraryGroup) => void;
  onPlaceGroup: (groupId: string, index: number) => void;
  onPlacePreset: (file: string, groupId: string, order: string[]) => void;
};

type DragData = {
  type?: 'group' | 'preset' | 'group-drop';
  groupId?: string;
  file?: string;
};

export { datasetPresetName };

export function DatasetGroupList({
  groups,
  pending,
  selectedFile,
  searchActive,
  ordering,
  orderingError,
  onSelect,
  onGroupAction,
  onPlaceGroup,
  onPlacePreset,
}: Props) {
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );
  const [activeType, setActiveType] = useState<DragData['type']>();
  const sortableGroups = groups.filter((group) => isSortableDatasetGroup(group, searchActive));

  function handleDragStart(event: DragStartEvent) {
    setActiveType((event.active.data.current as DragData | undefined)?.type);
  }

  function handleDragEnd(event: DragEndEvent) {
    setActiveType(undefined);
    if (!event.over) return;
    const active = event.active.data.current as DragData | undefined;
    const over = event.over.data.current as DragData | undefined;
    if (!active?.type || !over?.groupId) return;

    if (active.type === 'group' && active.groupId) {
      const oldIndex = sortableGroups.findIndex((group) => group.id === active.groupId);
      const overIndex = sortableGroups.findIndex((group) => group.id === over.groupId);
      if (oldIndex >= 0 && overIndex >= 0 && oldIndex !== overIndex) {
        onPlaceGroup(active.groupId, overIndex);
      }
      return;
    }

    if (active.type !== 'preset' || !active.file || !active.groupId) return;
    const sourceGroup = groups.find((group) => group.id === active.groupId);
    const targetGroup = groups.find((group) => group.id === over.groupId);
    if (!sourceGroup || !targetGroup) return;

    const targetPaths = targetGroup.files.map((preset) => preset.path);
    let nextOrder: string[];
    if (over.type === 'preset' && over.file) {
      const overIndex = targetPaths.indexOf(over.file);
      if (overIndex < 0) return;
      if (sourceGroup.id === targetGroup.id) {
        nextOrder = movePath(targetPaths, active.file, overIndex);
      } else {
        const translatedTop = event.active.rect.current.translated?.top;
        const insertAfter = translatedTop != null
          && translatedTop > event.over.rect.top + event.over.rect.height / 2;
        nextOrder = insertPath(targetPaths, active.file, overIndex + (insertAfter ? 1 : 0));
      }
    } else {
      nextOrder = insertPath(targetPaths, active.file);
    }

    const unchanged = sourceGroup.id === targetGroup.id
      && nextOrder.every((path, index) => path === targetPaths[index]);
    if (!unchanged) onPlacePreset(active.file, targetGroup.id, nextOrder);
  }

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragStart={handleDragStart}
      onDragCancel={() => setActiveType(undefined)}
      onDragEnd={handleDragEnd}
    >
      <SortableContext items={groups.map((group) => groupDragId(group.id))} strategy={verticalListSortingStrategy}>
        <div className="dataset-group-list">
          {pending ? <p className="dataset-empty">正在读取预设库</p> : null}
          {!pending && groups.length === 0 ? <p className="dataset-empty">没有匹配的数据集预设</p> : null}
          {orderingError ? <p className="dataset-command-error" role="alert">{orderingError}</p> : null}
          {groups.map((group) => (
            <SortableDatasetGroup
              key={group.id}
              group={group}
              groups={groups}
              selectedFile={selectedFile}
              searchActive={searchActive}
              ordering={ordering}
              sortableGroupIndex={sortableGroups.findIndex((item) => item.id === group.id)}
              sortableGroupCount={sortableGroups.length}
              presetDragging={activeType === 'preset'}
              onSelect={onSelect}
              onGroupAction={onGroupAction}
              onPlaceGroup={onPlaceGroup}
              onPlacePreset={onPlacePreset}
            />
          ))}
        </div>
      </SortableContext>
    </DndContext>
  );
}
