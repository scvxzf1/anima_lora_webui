import {
  DndContext,
  DragOverlay,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragStartEvent,
} from '@dnd-kit/core';
import { SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { datasetCollision, datasetDropOrder, type DatasetDrop } from './datasetDrag';
import './DatasetDrag.css';
import { useDatasetGroupCollapse } from './useDatasetGroupCollapse';

import { isSortableDatasetGroup } from './datasetOrdering';
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
  const [activeFile, setActiveFile] = useState<string>();
  const [dropTarget, setDropTarget] = useState<DatasetDrop | null>(null);
  const collapse = useDatasetGroupCollapse(activeType === 'preset', searchActive, dropTarget?.groupId ?? null);
  const targetRef = useRef<DatasetDrop | null>(null);
  const collision = useMemo(() => datasetCollision((target) => { targetRef.current = target; }), []);
  function syncDropTarget() {
    const next = targetRef.current;
    setDropTarget((previous) => (
      previous?.groupId === next?.groupId
      && previous?.file === next?.file
      && previous?.position === next?.position
        ? previous
        : next
    ));
  }
  function resetDrag() {
    setActiveType(undefined);
    setActiveFile(undefined);
    setDropTarget(null);
    targetRef.current = null;
  }
  const sortableGroups = groups.filter((group) => isSortableDatasetGroup(group, searchActive));

  function handleDragStart(event: DragStartEvent) {
    setActiveType((event.active.data.current as DragData | undefined)?.type);
    setActiveFile(event.active.data.current?.file);
  }

  function handleDragEnd(event: DragEndEvent) {
    const target = targetRef.current;
    resetDrag();
    if (ordering || searchActive) return;
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
    const targetGroup = groups.find((group) => group.id === target?.groupId);
    if (!sourceGroup || !targetGroup) return;

    const targetPaths = targetGroup.files.map((preset) => preset.path);
    const nextOrder = target && datasetDropOrder(groups, active.file, target);
    if (!nextOrder) return;

    const unchanged = sourceGroup.id === targetGroup.id
      && nextOrder.every((path, index) => path === targetPaths[index]);
    if (!unchanged) onPlacePreset(active.file, targetGroup.id, nextOrder);
  }

  return (
    <DndContext
      sensors={sensors}
      autoScroll={{ threshold: { x: 0.05, y: 0.08 }, acceleration: 5 }}
      collisionDetection={collision}
      onDragStart={handleDragStart}
      onDragCancel={resetDrag}
      onDragMove={syncDropTarget}
      onDragOver={syncDropTarget}
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
              collapsed={collapse.isCollapsed(group.id)}
              temporaryExpanded={collapse.isTemporary(group.id)}
              collapseDisabled={Boolean(activeType) || searchActive}
              onToggle={() => collapse.toggle(group.id)}
              dropTarget={dropTarget}
              onSelect={onSelect}
              onGroupAction={onGroupAction}
              onPlaceGroup={onPlaceGroup}
              onPlacePreset={onPlacePreset}
            />
          ))}
        </div>
      </SortableContext>
      {createPortal(<DragOverlay dropAnimation={null}>
        {activeFile && <div className="dataset-drag-overlay">{datasetPresetName(groups.flatMap((group) => group.files).find((file) => file.path === activeFile) || { path: activeFile })}</div>}
      </DragOverlay>, document.body)}
    </DndContext>
  );
}
